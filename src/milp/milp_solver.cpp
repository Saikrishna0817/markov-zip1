#include "markov_cero/milp/milp_solver.hpp"

#include "markov_cero/lp/dual/dual_simplex.hpp"
#include "markov_cero/lp/reference/revised_simplex.hpp"
#include "markov_cero/milp/branch_node.hpp"
#include "markov_cero/milp/cuts.hpp"
#include "markov_cero/milp/heuristics.hpp"
#include "markov_cero/milp/strong_branching.hpp"
#include "markov_cero/transform/sparse_canonical_model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <queue>

namespace markov_cero::milp {
namespace {

struct NodeLpResult {
    lp::reference::SolveStatus status{lp::reference::SolveStatus::infeasible};
    std::vector<double> primal;
    double objective{0.0};
    std::size_t iterations{0};
    std::optional<lp::dual::BasisState> basis;
};

NodeLpResult solve_node_lp(
    const model::Model& node_model,
    const Options& options,
    const std::optional<lp::dual::BasisState>& warm_start) {
    NodeLpResult res;
    try {
        const auto canon = transform::sparse_canonicalize(node_model, /*relax_integrality=*/true);
        const auto dense = canon.to_dense();

        if (options.enable_warm_start && warm_start.has_value()) {
            lp::dual::Options dopts;
            dopts.iteration_limit = options.max_iterations;
            dopts.feasibility_tolerance = options.feasibility_tolerance;
            dopts.allow_cold_fallback = true;
            const auto dres = lp::dual::solve(dense, dopts, warm_start);
            res.status = dres.solution.status;
            res.iterations = dres.solution.phase_one_iterations + dres.solution.phase_two_iterations;
            if (res.status == lp::reference::SolveStatus::optimal) {
                res.primal = transform::reconstruct_primal(canon, dres.solution.primal);
                res.objective = transform::reconstruct_objective(canon, dres.solution.objective);
                res.basis = dres.basis_state;
            }
        } else {
            lp::reference::Options ropts;
            ropts.iteration_limit = options.max_iterations;
            ropts.feasibility_tolerance = options.feasibility_tolerance;
            const auto rres = lp::reference::solve(dense, ropts);
            res.status = rres.status;
            res.iterations = rres.phase_one_iterations + rres.phase_two_iterations;
            if (res.status == lp::reference::SolveStatus::optimal) {
                res.primal = transform::reconstruct_primal(canon, rres.primal);
                res.objective = transform::reconstruct_objective(canon, rres.objective);
                if (rres.basis.size() == dense.matrix.rows) {
                    try {
                        res.basis = lp::dual::make_basis_state(dense, rres.basis);
                    } catch (...) {
                    }
                }
            }
        }
    } catch (...) {
        res.status = lp::reference::SolveStatus::numerical_failure;
    }
    return res;
}

} // namespace

Result solve(const model::Model& model, const Options& options) {
    const auto start_time = std::chrono::steady_clock::now();
    Result result;

    try {
        model.validate();
    } catch (const std::exception& e) {
        result.status = lp::reference::SolveStatus::invalid_model;
        result.message = e.what();
        return result;
    }

    // Check if model is purely continuous
    bool has_discrete = false;
    for (const auto type : model.variable_type) {
        if (type != model::VariableType::continuous) {
            has_discrete = true;
            break;
        }
    }

    if (!has_discrete) {
        // Pure continuous LP shortcut
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
        const auto elapsed = std::chrono::steady_clock::now() - start_time;
        result.runtime_ms = std::chrono::duration<double, std::milli>(elapsed).count();
        return result;
    }

    // Initialize search state
    std::size_t next_node_id = 1;
    double best_upper_bound = std::numeric_limits<double>::infinity();
    double best_lower_bound = -std::numeric_limits<double>::infinity();
    std::vector<double> best_primal;
    std::vector<VariablePseudoCost> pseudo_costs(model.matrix.column_count);

    model::Model root_model = model;

    // 1. Solve Root Continuous LP Relaxation
    const auto root_lp = solve_node_lp(root_model, options, std::nullopt);
    result.lp_iterations += root_lp.iterations;
    result.nodes_explored = 1;

    if (root_lp.status == lp::reference::SolveStatus::infeasible) {
        result.status = lp::reference::SolveStatus::infeasible;
        result.message = "root continuous relaxation is infeasible";
        const auto elapsed = std::chrono::steady_clock::now() - start_time;
        result.runtime_ms = std::chrono::duration<double, std::milli>(elapsed).count();
        return result;
    }
    if (root_lp.status != lp::reference::SolveStatus::optimal) {
        result.status = root_lp.status;
        result.message = "root continuous relaxation failed: " + std::to_string(static_cast<int>(root_lp.status));
        const auto elapsed = std::chrono::steady_clock::now() - start_time;
        result.runtime_ms = std::chrono::duration<double, std::milli>(elapsed).count();
        return result;
    }

    best_lower_bound = root_lp.objective;

    // Check if root continuous solution is integer feasible
    if (check_integer_feasibility(root_model, root_lp.primal, options.feasibility_tolerance, options.integrality_tolerance)) {
        result.status = lp::reference::SolveStatus::optimal;
        result.primal = root_lp.primal;
        result.objective = root_lp.objective;
        result.best_bound = root_lp.objective;
        result.relative_gap = 0.0;
        result.message = "root relaxation integer feasible (integer optimal)";
        const auto elapsed = std::chrono::steady_clock::now() - start_time;
        result.runtime_ms = std::chrono::duration<double, std::milli>(elapsed).count();
        return result;
    }

    // 2. Run Primal Heuristics at Root
    if (options.enable_heuristics) {
        const auto hr = simple_rounding(root_model, root_lp.primal, options.feasibility_tolerance, options.integrality_tolerance);
        if (hr.found && hr.objective < best_upper_bound) {
            best_upper_bound = hr.objective;
            best_primal = hr.primal;
            ++result.heuristics_found;
        }

        const auto fp = feasibility_pump(root_model, root_lp.primal, options.max_pump_iterations, options.feasibility_tolerance, options.integrality_tolerance);
        if (fp.found && fp.objective < best_upper_bound) {
            best_upper_bound = fp.objective;
            best_primal = fp.primal;
            ++result.heuristics_found;
        }
    }

    // 3. Generate Gomory Mixed-Integer and MIR Cuts at Root
    std::optional<lp::dual::BasisState> current_basis = root_lp.basis;
    std::vector<double> current_primal = root_lp.primal;
    double current_obj = root_lp.objective;

    if (options.enable_cuts && root_lp.basis.has_value()) {
        try {
            const auto canon = transform::sparse_canonicalize(root_model, /*relax_integrality=*/true);
            std::vector<Cut> cuts = generate_gomory_cuts(root_model, current_primal, canon, *root_lp.basis, options.max_cut_rounds);
            if (options.enable_mir_cuts) {
                const auto mir_cuts = generate_mir_cuts(root_model, current_primal, canon, *root_lp.basis, options.max_cut_rounds);
                cuts.insert(cuts.end(), mir_cuts.begin(), mir_cuts.end());
            }
            cuts = filter_cuts(std::move(cuts), options.max_cut_rounds);
            if (!cuts.empty()) {
                add_cuts_to_model(root_model, cuts);
                result.cuts_generated += cuts.size();

                // Re-solve root LP with cuts
                const auto cut_lp = solve_node_lp(root_model, options, root_lp.basis);
                result.lp_iterations += cut_lp.iterations;
                if (cut_lp.status == lp::reference::SolveStatus::optimal) {
                    current_primal = cut_lp.primal;
                    current_obj = cut_lp.objective;
                    current_basis = cut_lp.basis;
                    best_lower_bound = std::max(best_lower_bound, current_obj);

                    if (check_integer_feasibility(root_model, current_primal, options.feasibility_tolerance, options.integrality_tolerance)) {
                        if (current_obj < best_upper_bound) {
                            best_upper_bound = current_obj;
                            best_primal = current_primal;
                        }
                    }
                }
            }
        } catch (...) {
        }
    }

    // Check if root cuts closed the optimality gap
    if (!best_primal.empty() && best_lower_bound > -std::numeric_limits<double>::infinity()) {
        const double gap = std::abs(best_upper_bound - best_lower_bound) / std::max(1.0, std::abs(best_upper_bound));
        if (gap <= options.relative_gap_tolerance) {
            result.status = lp::reference::SolveStatus::optimal;
            result.primal = best_primal;
            result.objective = best_upper_bound;
            result.best_bound = best_lower_bound;
            result.relative_gap = gap;
            result.message = "optimality gap closed at root node";
            const auto elapsed = std::chrono::steady_clock::now() - start_time;
            result.runtime_ms = std::chrono::duration<double, std::milli>(elapsed).count();
            return result;
        }
    }

    // 4. Evaluate Strong Branching at Root to Initialize Pseudo-Costs and Tighten Bounds
    if (options.enable_strong_branching && current_basis.has_value()) {
        try {
            StrongBranchingOptions sb_opts;
            sb_opts.integrality_tolerance = options.integrality_tolerance;
            sb_opts.feasibility_tolerance = options.feasibility_tolerance;
            sb_opts.update_pseudo_costs = true;
            const auto sb_res = evaluate_strong_branching(
                root_model, current_primal, current_obj, current_basis, sb_opts, &pseudo_costs);

            if (sb_res.subproblem_infeasible) {
                result.status = lp::reference::SolveStatus::infeasible;
                result.message = "proven infeasible by strong branching at root";
                const auto elapsed = std::chrono::steady_clock::now() - start_time;
                result.runtime_ms = std::chrono::duration<double, std::milli>(elapsed).count();
                return result;
            }

            // Apply discovered domain reductions to root model
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

    // 5. Initialize Active Node Priority Queue
    std::priority_queue<std::shared_ptr<BranchNode>,
                        std::vector<std::shared_ptr<BranchNode>>,
                        NodeCompareBestBound> queue;

    auto root_node = std::make_shared<BranchNode>();
    root_node->id = 0;
    root_node->parent_id = 0;
    root_node->depth = 0;
    root_node->lower_bound = current_obj;
    root_node->variable_lower = root_model.variable_lower;
    root_node->variable_upper = root_model.variable_upper;
    root_node->warm_basis = current_basis;
    queue.push(root_node);

    // 6. Tree Search Loop
    while (!queue.empty() && result.nodes_explored < options.max_nodes) {
        const auto now = std::chrono::steady_clock::now();
        const auto time_spent = std::chrono::duration<double>(now - start_time).count();
        if (time_spent > options.time_limit_seconds) {
            break;
        }

        auto node = queue.top();
        queue.pop();

        // Bound pruning
        if (node->lower_bound >= best_upper_bound - options.absolute_gap_tolerance) {
            continue;
        }

        // Evaluate Node LP relaxation if not root
        NodeLpResult node_lp_res;
        if (node->id == 0) {
            node_lp_res.status = lp::reference::SolveStatus::optimal;
            node_lp_res.primal = current_primal;
            node_lp_res.objective = current_obj;
            node_lp_res.basis = current_basis;
        } else {
            model::Model node_model = root_model;
            node_model.variable_lower = node->variable_lower;
            node_model.variable_upper = node->variable_upper;

            node_lp_res = solve_node_lp(node_model, options, node->warm_basis);
            result.lp_iterations += node_lp_res.iterations;
            ++result.nodes_explored;

            if (node_lp_res.status != lp::reference::SolveStatus::optimal) {
                // Infeasible or numerical failure -> prune
                continue;
            }

            node->lower_bound = node_lp_res.objective;

            // Bound pruning after solving node LP
            if (node_lp_res.objective >= best_upper_bound - options.absolute_gap_tolerance) {
                continue;
            }

            // Update pseudo-cost of parent branch
            if (node->depth > 0) {
                const std::size_t b_var = node->branch_variable;
                const double delta_z = node_lp_res.objective - node->lower_bound;
                const double frac = node->branch_value - std::floor(node->branch_value);
                if (node->is_down_branch) {
                    pseudo_costs[b_var].record_down(delta_z, frac);
                } else {
                    pseudo_costs[b_var].record_up(delta_z, 1.0 - frac);
                }
            }
        }

        // Check integer feasibility of node solution
        const auto fractional_vars = find_fractional_variables(
            node_lp_res.primal, root_model.variable_type, options.integrality_tolerance);

        if (fractional_vars.empty()) {
            // Integer feasible incumbent found!
            if (node_lp_res.objective < best_upper_bound) {
                best_upper_bound = node_lp_res.objective;
                best_primal = node_lp_res.primal;
            }
            continue;
        }

        // Try quick simple rounding on fractional point
        if (options.enable_heuristics && result.nodes_explored % 5 == 0) {
            model::Model node_model = root_model;
            node_model.variable_lower = node->variable_lower;
            node_model.variable_upper = node->variable_upper;
            const auto hr = simple_rounding(node_model, node_lp_res.primal, options.feasibility_tolerance, options.integrality_tolerance);
            if (hr.found && hr.objective < best_upper_bound) {
                best_upper_bound = hr.objective;
                best_primal = hr.primal;
                ++result.heuristics_found;
            }
        }

        // 7. Branching Variable Selection
        std::size_t branch_var = root_model.matrix.column_count;
        if (options.branching_strategy == BranchingStrategy::strong_branching && node_lp_res.basis.has_value()) {
            try {
                StrongBranchingOptions sb_opts;
                sb_opts.integrality_tolerance = options.integrality_tolerance;
                sb_opts.feasibility_tolerance = options.feasibility_tolerance;
                sb_opts.update_pseudo_costs = true;
                model::Model current_node_model = root_model;
                current_node_model.variable_lower = node->variable_lower;
                current_node_model.variable_upper = node->variable_upper;
                const auto sb_res = evaluate_strong_branching(
                    current_node_model, node_lp_res.primal, node_lp_res.objective,
                    node_lp_res.basis, sb_opts, &pseudo_costs);
                if (sb_res.subproblem_infeasible) {
                    continue; // Prune node
                }
                branch_var = sb_res.best_variable;
            } catch (...) {
                branch_var = select_branching_variable(
                    node_lp_res.primal, root_model.variable_type, pseudo_costs,
                    options.branching_strategy, options.integrality_tolerance);
            }
        } else {
            branch_var = select_branching_variable(
                node_lp_res.primal, root_model.variable_type, pseudo_costs,
                options.branching_strategy, options.integrality_tolerance);
        }

        if (branch_var >= root_model.matrix.column_count) {
            continue;
        }

        const double branch_val = node_lp_res.primal[branch_var];
        const double floor_val = std::floor(branch_val);
        const double ceil_val = std::ceil(branch_val);

        // Child 1 (Down Branch): x_k <= floor_val
        bool down_valid = true;
        if (node->variable_lower[branch_var].is_finite() &&
            floor_val < node->variable_lower[branch_var].value - 1e-9) {
            down_valid = false;
        }
        if (down_valid) {
            auto down_child = std::make_shared<BranchNode>();
            down_child->id = next_node_id++;
            down_child->parent_id = node->id;
            down_child->depth = node->depth + 1;
            down_child->lower_bound = node_lp_res.objective; // parent lower bound is valid lower bound
            down_child->branch_variable = branch_var;
            down_child->branch_value = branch_val;
            down_child->is_down_branch = true;
            down_child->variable_lower = node->variable_lower;
            down_child->variable_upper = node->variable_upper;
            down_child->variable_upper[branch_var] = model::Bound::finite(floor_val);
            down_child->warm_basis = node_lp_res.basis;
            queue.push(down_child);
        }

        // Child 2 (Up Branch): x_k >= ceil_val
        bool up_valid = true;
        if (node->variable_upper[branch_var].is_finite() &&
            ceil_val > node->variable_upper[branch_var].value + 1e-9) {
            up_valid = false;
        }
        if (up_valid) {
            auto up_child = std::make_shared<BranchNode>();
            up_child->id = next_node_id++;
            up_child->parent_id = node->id;
            up_child->depth = node->depth + 1;
            up_child->lower_bound = node_lp_res.objective;
            up_child->branch_variable = branch_var;
            up_child->branch_value = branch_val;
            up_child->is_down_branch = false;
            up_child->variable_lower = node->variable_lower;
            up_child->variable_upper = node->variable_upper;
            up_child->variable_lower[branch_var] = model::Bound::finite(ceil_val);
            up_child->warm_basis = node_lp_res.basis;
            queue.push(up_child);
        }

        // Update global lower bound from active queue
        if (!queue.empty()) {
            best_lower_bound = std::min(best_upper_bound, queue.top()->lower_bound);
        }

        // Check relative optimality gap
        if (!best_primal.empty() && best_lower_bound > -std::numeric_limits<double>::infinity()) {
            const double gap = std::abs(best_upper_bound - best_lower_bound) / std::max(1.0, std::abs(best_upper_bound));
            if (gap <= options.relative_gap_tolerance) {
                break;
            }
        }
    }

    // 7. Assemble Final Result
    const auto end_time = std::chrono::steady_clock::now();
    result.runtime_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    if (!best_primal.empty()) {
        result.status = lp::reference::SolveStatus::optimal;
        result.primal = best_primal;
        result.objective = best_upper_bound;
        result.best_bound = (queue.empty() ? best_upper_bound : best_lower_bound);
        if (std::abs(result.best_bound) > 1e15) {
            result.best_bound = result.objective;
        }
        result.relative_gap = std::max(0.0, std::abs(result.objective - result.best_bound) / std::max(1.0, std::abs(result.objective)));
        result.message = "branch-and-cut MILP optimum";
    } else {
        result.status = lp::reference::SolveStatus::infeasible;
        result.message = "no integer feasible solution found";
    }

    return result;
}

} // namespace markov_cero::milp
