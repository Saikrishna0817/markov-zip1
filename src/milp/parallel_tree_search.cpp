#include "markov_cero/milp/parallel_tree_search.hpp"

#include "markov_cero/lp/dual/dual_simplex.hpp"
#include "markov_cero/lp/reference/revised_simplex.hpp"
#include "markov_cero/milp/cuts.hpp"
#include "markov_cero/milp/heuristics.hpp"
#include "markov_cero/milp/strong_branching.hpp"
#include "markov_cero/transform/sparse_canonical_model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stop_token>
#include <thread>
#include <vector>

namespace markov_cero::milp {
namespace {

struct NodeLpResult {
    lp::reference::SolveStatus status{lp::reference::SolveStatus::infeasible};
    std::vector<double> primal;
    double objective{0.0};
    std::size_t iterations{0};
    std::optional<lp::dual::BasisState> basis;
};

NodeLpResult solve_node_lp(const model::Model& node_model, const ParallelOptions& options,
                           const std::optional<lp::dual::BasisState>& warm_start) {
    NodeLpResult res;
    try {
        const auto canon = transform::sparse_canonicalize(node_model, /*relax_integrality=*/true);
        const auto dense = canon.to_dense();
        lp::reference::Result sol;

        if (options.enable_warm_start && warm_start.has_value()) {
            lp::dual::Options dopts;
            dopts.iteration_limit = options.max_iterations;
            dopts.feasibility_tolerance = options.feasibility_tolerance;
            dopts.allow_cold_fallback = true;
            const auto dres = lp::dual::solve(dense, dopts, warm_start);
            sol = dres.solution;
            res.basis = dres.basis_state;
        } else {
            lp::reference::Options ropts;
            ropts.iteration_limit = options.max_iterations;
            ropts.feasibility_tolerance = options.feasibility_tolerance;
            sol = lp::reference::solve(dense, ropts);
            if (sol.basis.size() == dense.matrix.rows) {
                try {
                    res.basis = lp::dual::make_basis_state(dense, sol.basis);
                } catch (...) {
                }
            }
        }
        res.status = sol.status;
        res.iterations = sol.phase_one_iterations + sol.phase_two_iterations;
        if (res.status == lp::reference::SolveStatus::optimal) {
            res.primal = transform::reconstruct_primal(canon, sol.primal);
            res.objective = transform::reconstruct_objective(canon, sol.objective);
        }
    } catch (...) {
        res.status = lp::reference::SolveStatus::numerical_failure;
    }
    return res;
}

struct SharedPseudoCosts {
    std::mutex mutex;
    std::vector<VariablePseudoCost> costs;
};

void worker_loop(
    std::size_t thread_id, const model::Model& root_model, const ParallelOptions& options,
    ThreadSafeNodeQueue& queue, IncumbentManager& incumbent, std::atomic<std::size_t>& next_node_id,
    std::atomic<std::size_t>& total_nodes_explored, std::atomic<std::size_t>& total_lp_iterations,
    std::atomic<std::size_t>& total_heuristics_found, std::atomic<double>* worker_bounds,
    std::size_t num_threads, SharedPseudoCosts& shared_pseudo_costs,
    const std::chrono::steady_clock::time_point start_time, std::stop_token stop_token) {

    bool was_active = false;
    model::Model node_model = root_model;
    auto clear_bound = [&]() {
        worker_bounds[thread_id].store(std::numeric_limits<double>::infinity(),
                                       std::memory_order_relaxed);
    };

    while (!stop_token.stop_requested() && !queue.is_stopped()) {
        const auto now = std::chrono::steady_clock::now();
        const auto time_spent = std::chrono::duration<double>(now - start_time).count();
        if (time_spent > options.time_limit_seconds ||
            total_nodes_explored.load(std::memory_order_relaxed) >= options.max_nodes) {
            queue.request_stop();
            break;
        }

        const double current_incumbent =
            incumbent.best_incumbent_objective.load(std::memory_order_relaxed);
        const double prune_cutoff = current_incumbent - options.absolute_gap_tolerance;

        if (incumbent.has_incumbent()) {
            const double current_inc = incumbent.get_objective();
            const double tree_lb =
                compute_tree_lower_bound(queue, worker_bounds, num_threads, current_inc);
            if (tree_lb > -1e15) {
                const double gap =
                    std::abs(current_inc - tree_lb) / std::max(1.0, std::abs(current_inc));
                if (gap <= options.relative_gap_tolerance) {
                    queue.request_stop();
                    break;
                }
            }
        }

        bool became_active = false;
        auto node = queue.pop_node(was_active, prune_cutoff, became_active);
        was_active = became_active;

        if (!node) {
            break;
        }

        worker_bounds[thread_id].store(node->lower_bound, std::memory_order_relaxed);

        if (node->lower_bound >=
            incumbent.best_incumbent_objective.load(std::memory_order_relaxed) -
                options.absolute_gap_tolerance) {
            clear_bound();
            continue;
        }

        node_model.variable_lower = node->variable_lower;
        node_model.variable_upper = node->variable_upper;

        const auto node_lp_res = solve_node_lp(node_model, options, node->warm_basis);
        total_lp_iterations.fetch_add(node_lp_res.iterations, std::memory_order_relaxed);
        total_nodes_explored.fetch_add(1, std::memory_order_relaxed);

        if (node_lp_res.status == lp::reference::SolveStatus::infeasible) {
            clear_bound();
            continue;
        }
        if (node_lp_res.status != lp::reference::SolveStatus::optimal) {
            // Unresolved LP failure — do not treat as proven prune
            clear_bound();
            continue;
        }

        node->lower_bound = node_lp_res.objective;
        worker_bounds[thread_id].store(node->lower_bound, std::memory_order_relaxed);

        if (node_lp_res.objective >=
            incumbent.best_incumbent_objective.load(std::memory_order_relaxed) -
                options.absolute_gap_tolerance) {
            clear_bound();
            continue;
        }

        if (node->depth > 0) {
            const std::size_t b_var = node->branch_variable;
            const double delta_z = node_lp_res.objective - node->lower_bound;
            const double frac = node->branch_value - std::floor(node->branch_value);
            std::lock_guard<std::mutex> pc_lock(shared_pseudo_costs.mutex);
            if (node->is_down_branch) {
                shared_pseudo_costs.costs[b_var].record_down(delta_z, frac);
            } else {
                shared_pseudo_costs.costs[b_var].record_up(delta_z, 1.0 - frac);
            }
        }

        const auto fractional_vars = find_fractional_variables(
            node_lp_res.primal, root_model.variable_type, options.integrality_tolerance);

        if (fractional_vars.empty()) {
            if (incumbent.update_if_better(node_lp_res.objective, node_lp_res.primal,
                                           options.absolute_gap_tolerance)) {
                queue.prune(incumbent.best_incumbent_objective.load(std::memory_order_relaxed) -
                            options.absolute_gap_tolerance);
            }
            clear_bound();
            continue;
        }

        if (options.enable_heuristics &&
            total_nodes_explored.load(std::memory_order_relaxed) % 10 == 0) {
            const auto hr =
                simple_rounding(node_model, node_lp_res.primal, options.feasibility_tolerance,
                                options.integrality_tolerance);
            if (hr.found && incumbent.update_if_better(hr.objective, hr.primal,
                                                       options.absolute_gap_tolerance)) {
                total_heuristics_found.fetch_add(1, std::memory_order_relaxed);
                queue.prune(incumbent.best_incumbent_objective.load(std::memory_order_relaxed) -
                            options.absolute_gap_tolerance);
            }
        }

        std::vector<VariablePseudoCost> pc_snapshot;
        {
            std::lock_guard<std::mutex> pc_lock(shared_pseudo_costs.mutex);
            pc_snapshot = shared_pseudo_costs.costs;
        }
        const std::size_t branch_var =
            select_branching_variable(node_lp_res.primal, root_model.variable_type, pc_snapshot,
                                      options.branching_strategy, options.integrality_tolerance);

        if (branch_var >= root_model.matrix.column_count) {
            clear_bound();
            continue;
        }

        queue.push_branch_children(*node, branch_var, node_lp_res.primal[branch_var],
                                   node_lp_res.objective, node_lp_res.basis, next_node_id);
        clear_bound();
    }

    if (was_active) {
        queue.deactivate_worker();
    }
    clear_bound();
}

} // namespace

Result solve_parallel(const model::Model& model, const ParallelOptions& options) {
    const auto start_time = std::chrono::steady_clock::now();
    Result result;

    auto elapsed_ms = [&]() {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_time).count();
    };

    auto fail_early = [&](lp::reference::SolveStatus st, std::string msg) {
        result.status = st;
        result.message = std::move(msg);
        result.runtime_ms = elapsed_ms();
        return result;
    };

    try {
        model.validate();
    } catch (const std::exception& e) {
        return fail_early(lp::reference::SolveStatus::invalid_model, e.what());
    }

    bool has_discrete = false;
    for (const auto type : model.variable_type) {
        if (type != model::VariableType::continuous) {
            has_discrete = true;
            break;
        }
    }

    if (!has_discrete) {
        const auto canon = transform::sparse_canonicalize(model, /*relax_integrality=*/false);
        const auto dense = canon.to_dense();
        lp::reference::Options ropts;
        ropts.iteration_limit = options.max_iterations;
        ropts.feasibility_tolerance = options.feasibility_tolerance;
        const auto lpres = lp::reference::solve(dense, ropts);

        result.status = lpres.status;
        result.lp_iterations = lpres.phase_one_iterations + lpres.phase_two_iterations;
        result.nodes_explored = 1;
        if (result.status == lp::reference::SolveStatus::optimal) {
            result.primal = transform::reconstruct_primal(canon, lpres.primal);
            result.objective = transform::reconstruct_objective(canon, lpres.objective);
            result.best_bound = result.objective;
            result.relative_gap = 0.0;
            result.message = "pure continuous LP solved to optimality";
        } else {
            result.message = lpres.message;
        }
        result.runtime_ms = elapsed_ms();
        return result;
    }

    IncumbentManager incumbent;
    std::atomic<std::size_t> next_node_id{1};
    std::atomic<std::size_t> total_nodes_explored{1};
    std::atomic<std::size_t> total_lp_iterations{0};
    std::atomic<std::size_t> total_heuristics_found{0};
    std::size_t root_cuts_generated = 0;

    SharedPseudoCosts shared_pseudo_costs;
    shared_pseudo_costs.costs.resize(model.matrix.column_count);

    model::Model root_model = model;

    const auto root_lp = solve_node_lp(root_model, options, std::nullopt);
    total_lp_iterations.fetch_add(root_lp.iterations, std::memory_order_relaxed);

    if (root_lp.status == lp::reference::SolveStatus::infeasible) {
        return fail_early(lp::reference::SolveStatus::infeasible,
                          "root continuous relaxation is infeasible");
    }
    if (root_lp.status != lp::reference::SolveStatus::optimal) {
        return fail_early(root_lp.status, "root continuous relaxation failed: " +
                                              std::to_string(static_cast<int>(root_lp.status)));
    }

    double best_lower_bound = root_lp.objective;

    if (check_integer_feasibility(root_model, root_lp.primal, options.feasibility_tolerance,
                                  options.integrality_tolerance)) {
        result.status = lp::reference::SolveStatus::optimal;
        result.primal = root_lp.primal;
        result.objective = root_lp.objective;
        result.best_bound = root_lp.objective;
        result.relative_gap = 0.0;
        result.nodes_explored = 1;
        result.lp_iterations = total_lp_iterations.load(std::memory_order_relaxed);
        result.message = "root relaxation integer feasible (integer optimal)";
        result.runtime_ms = elapsed_ms();
        return result;
    }

    if (options.enable_heuristics) {
        const auto hr = simple_rounding(root_model, root_lp.primal, options.feasibility_tolerance,
                                        options.integrality_tolerance);
        if (hr.found &&
            incumbent.update_if_better(hr.objective, hr.primal, options.absolute_gap_tolerance)) {
            total_heuristics_found.fetch_add(1, std::memory_order_relaxed);
        }

        const auto fp =
            feasibility_pump(root_model, root_lp.primal, options.max_pump_iterations,
                             options.feasibility_tolerance, options.integrality_tolerance);
        if (fp.found &&
            incumbent.update_if_better(fp.objective, fp.primal, options.absolute_gap_tolerance)) {
            total_heuristics_found.fetch_add(1, std::memory_order_relaxed);
        }
    }

    std::optional<lp::dual::BasisState> current_basis = root_lp.basis;
    std::vector<double> current_primal = root_lp.primal;
    double current_obj = root_lp.objective;

    if (options.enable_cuts && root_lp.basis.has_value()) {
        try {
            const auto canon =
                transform::sparse_canonicalize(root_model, /*relax_integrality=*/true);
            std::vector<Cut> cuts = generate_gomory_cuts(root_model, current_primal, canon,
                                                         *root_lp.basis, options.max_cut_rounds);
            if (options.enable_mir_cuts) {
                const auto mir_cuts = generate_mir_cuts(root_model, current_primal, canon,
                                                        *root_lp.basis, options.max_cut_rounds);
                cuts.insert(cuts.end(), mir_cuts.begin(), mir_cuts.end());
            }
            cuts = filter_cuts(std::move(cuts), options.max_cut_rounds);
            if (!cuts.empty()) {
                add_cuts_to_model(root_model, cuts);
                root_cuts_generated = cuts.size();

                const auto cut_lp = solve_node_lp(root_model, options, root_lp.basis);
                total_lp_iterations.fetch_add(cut_lp.iterations, std::memory_order_relaxed);
                if (cut_lp.status == lp::reference::SolveStatus::optimal) {
                    current_primal = cut_lp.primal;
                    current_obj = cut_lp.objective;
                    current_basis = cut_lp.basis;
                    best_lower_bound = std::max(best_lower_bound, current_obj);

                    if (check_integer_feasibility(root_model, current_primal,
                                                  options.feasibility_tolerance,
                                                  options.integrality_tolerance)) {
                        incumbent.update_if_better(current_obj, current_primal,
                                                   options.absolute_gap_tolerance);
                    }
                }
            }
        } catch (...) {
        }
    }

    if (incumbent.has_incumbent() && best_lower_bound > -1e15) {
        const double gap = std::abs(incumbent.get_objective() - best_lower_bound) /
                           std::max(1.0, std::abs(incumbent.get_objective()));
        if (gap <= options.relative_gap_tolerance) {
            result.status = lp::reference::SolveStatus::optimal;
            result.primal = incumbent.get_primal();
            result.objective = incumbent.get_objective();
            result.best_bound = best_lower_bound;
            result.relative_gap = gap;
            result.nodes_explored = 1;
            result.lp_iterations = total_lp_iterations.load(std::memory_order_relaxed);
            result.cuts_generated = root_cuts_generated;
            result.heuristics_found = total_heuristics_found.load(std::memory_order_relaxed);
            result.message = "optimality gap closed at root node";
            result.runtime_ms = elapsed_ms();
            return result;
        }
    }

    if (options.enable_strong_branching && current_basis.has_value()) {
        try {
            StrongBranchingOptions sb_opts;
            sb_opts.integrality_tolerance = options.integrality_tolerance;
            sb_opts.feasibility_tolerance = options.feasibility_tolerance;
            sb_opts.update_pseudo_costs = true;
            std::vector<VariablePseudoCost> initial_pc = shared_pseudo_costs.costs;
            const auto sb_res = evaluate_strong_branching(root_model, current_primal, current_obj,
                                                          current_basis, sb_opts, &initial_pc);

            if (sb_res.subproblem_infeasible) {
                return fail_early(lp::reference::SolveStatus::infeasible,
                                  "proven infeasible by strong branching at root");
            }

            {
                std::lock_guard<std::mutex> lock(shared_pseudo_costs.mutex);
                shared_pseudo_costs.costs = std::move(initial_pc);
            }

            for (const auto& dr : sb_res.domain_reductions) {
                if (dr.variable_index < root_model.matrix.column_count) {
                    if (dr.new_lower.is_finite()) {
                        root_model.variable_lower[dr.variable_index] = dr.new_lower;
                    }
                    if (dr.new_upper.is_finite()) {
                        root_model.variable_upper[dr.variable_index] = dr.new_upper;
                    }
                }
            }
        } catch (...) {
        }
    }

    const std::size_t root_branch_var = select_branching_variable(
        current_primal, root_model.variable_type, shared_pseudo_costs.costs,
        options.branching_strategy, options.integrality_tolerance);

    if (root_branch_var >= root_model.matrix.column_count) {
        if (incumbent.has_incumbent()) {
            result.status = lp::reference::SolveStatus::optimal;
            result.primal = incumbent.get_primal();
            result.objective = incumbent.get_objective();
            result.best_bound = incumbent.get_objective();
            result.relative_gap = 0.0;
        } else {
            result.status = lp::reference::SolveStatus::infeasible;
            result.message = "no integer feasible solution found";
        }
        result.runtime_ms = elapsed_ms();
        return result;
    }

    ThreadSafeNodeQueue queue;
    BranchNode root_node;
    root_node.id = 0;
    root_node.depth = 0;
    root_node.variable_lower = root_model.variable_lower;
    root_node.variable_upper = root_model.variable_upper;

    queue.push_branch_children(root_node, root_branch_var, current_primal[root_branch_var],
                               current_obj, current_basis, next_node_id);

    const std::size_t num_threads = std::max<std::size_t>(1, options.num_threads);
    auto worker_bounds = std::make_unique<std::atomic<double>[]>(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
        worker_bounds[i].store(std::numeric_limits<double>::infinity(), std::memory_order_relaxed);
    }

    // Keep jthreads alive until workers finish; destruction alone requests stop
    // before the queue protocol can complete (F2).
    {
        std::vector<std::jthread> workers;
        workers.reserve(num_threads);
        for (std::size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([&, i](std::stop_token st) {
                worker_loop(i, root_model, options, queue, incumbent, next_node_id,
                            total_nodes_explored, total_lp_iterations, total_heuristics_found,
                            worker_bounds.get(), num_threads, shared_pseudo_costs, start_time, st);
            });
        }
        // Explicit join while owners are still in scope
        for (auto& w : workers) {
            if (w.joinable()) {
                w.join();
            }
        }
    }

    result.runtime_ms = elapsed_ms();
    result.nodes_explored = total_nodes_explored.load(std::memory_order_relaxed);
    result.lp_iterations = total_lp_iterations.load(std::memory_order_relaxed);
    result.cuts_generated = root_cuts_generated;
    result.heuristics_found = total_heuristics_found.load(std::memory_order_relaxed);

    const bool tree_done = queue.empty() && queue.active_workers() == 0;
    const bool hit_limits =
        result.nodes_explored >= options.max_nodes ||
        (std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count() >
         options.time_limit_seconds);
    const bool complete = tree_done && !hit_limits;

    if (incumbent.has_incumbent()) {
        result.primal = incumbent.get_primal();
        result.objective = incumbent.get_objective();
        const double final_lb =
            compute_tree_lower_bound(queue, worker_bounds.get(), num_threads, result.objective);
        result.best_bound = tree_done ? result.objective
                                     : (std::isfinite(final_lb) ? final_lb : result.objective);
        result.relative_gap = std::max(0.0, std::abs(result.objective - result.best_bound) /
                                                std::max(1.0, std::abs(result.objective)));
        if (complete) {
            result.status = lp::reference::SolveStatus::optimal;
            result.message = "parallel tree search MILP optimum";
        } else {
            result.status = lp::reference::SolveStatus::resource_limit;
            result.message = "incomplete parallel search – feasible incumbent returned";
        }
    } else {
        if (complete) {
            result.status = lp::reference::SolveStatus::infeasible;
            result.message = "MILP proved infeasible";
        } else {
            result.status = lp::reference::SolveStatus::resource_limit;
            result.message = "incomplete parallel search – no incumbent";
        }
    }

    return result;
}

} // namespace markov_cero::milp
