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

// -----------------------------------------------------------------------------
// ThreadSafeNodeQueue Implementation
// -----------------------------------------------------------------------------

void ThreadSafeNodeQueue::push(std::shared_ptr<BranchNode> node) {
    if (!node) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_) {
        return;
    }
    heap_.push_back(std::move(node));
    std::push_heap(heap_.begin(), heap_.end(), NodeCompareBestBound{});
    cv_.notify_one();
}

void ThreadSafeNodeQueue::push_children(std::shared_ptr<BranchNode> left, std::shared_ptr<BranchNode> right) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_) {
        return;
    }
    bool pushed = false;
    if (left) {
        heap_.push_back(std::move(left));
        std::push_heap(heap_.begin(), heap_.end(), NodeCompareBestBound{});
        pushed = true;
    }
    if (right) {
        heap_.push_back(std::move(right));
        std::push_heap(heap_.begin(), heap_.end(), NodeCompareBestBound{});
        pushed = true;
    }
    if (pushed) {
        cv_.notify_all();
    }
}

void ThreadSafeNodeQueue::prune_locked(double cutoff) {
    if (heap_.empty()) {
        return;
    }
    auto it = std::remove_if(heap_.begin(), heap_.end(), [cutoff](const std::shared_ptr<BranchNode>& node) {
        return !node || node->lower_bound >= cutoff;
    });
    if (it != heap_.end()) {
        heap_.erase(it, heap_.end());
        std::make_heap(heap_.begin(), heap_.end(), NodeCompareBestBound{});
    }
}

void ThreadSafeNodeQueue::prune(double cutoff) {
    std::lock_guard<std::mutex> lock(mutex_);
    prune_locked(cutoff);
    cv_.notify_all();
}

std::shared_ptr<BranchNode> ThreadSafeNodeQueue::pop_node(
    bool was_active, double prune_cutoff, bool& became_active) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (was_active) {
        if (active_workers_ > 0) {
            --active_workers_;
        }
    }
    became_active = false;

    while (!stopped_) {
        prune_locked(prune_cutoff);

        if (!heap_.empty()) {
            std::pop_heap(heap_.begin(), heap_.end(), NodeCompareBestBound{});
            auto node = std::move(heap_.back());
            heap_.pop_back();

            ++active_workers_;
            became_active = true;
            return node;
        }

        // When queue is empty and no worker is active, search is complete.
        if (active_workers_ == 0) {
            stopped_ = true;
            cv_.notify_all();
            return nullptr;
        }

        cv_.wait_for(lock, std::chrono::milliseconds(20));
    }

    return nullptr;
}

void ThreadSafeNodeQueue::deactivate_worker() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_workers_ > 0) {
        --active_workers_;
    }
    if (active_workers_ == 0 && heap_.empty()) {
        stopped_ = true;
        cv_.notify_all();
    }
}

void ThreadSafeNodeQueue::request_stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    cv_.notify_all();
}

bool ThreadSafeNodeQueue::is_stopped() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stopped_;
}

bool ThreadSafeNodeQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return heap_.empty();
}

std::size_t ThreadSafeNodeQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return heap_.size();
}

std::size_t ThreadSafeNodeQueue::active_workers() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_workers_;
}

double ThreadSafeNodeQueue::min_lower_bound() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (heap_.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    return heap_.front()->lower_bound;
}

void ThreadSafeNodeQueue::notify_all() {
    cv_.notify_all();
}

// -----------------------------------------------------------------------------
// IncumbentManager Implementation
// -----------------------------------------------------------------------------

IncumbentManager::IncumbentManager(double initial_obj)
    : best_incumbent_objective(initial_obj),
      recorded_primal_obj_(initial_obj) {}

bool IncumbentManager::update_if_better(
    double candidate_obj, const std::vector<double>& candidate_primal, double eps) {
    double current = best_incumbent_objective.load(std::memory_order_relaxed);
    while (candidate_obj < current - eps) {
        if (best_incumbent_objective.compare_exchange_weak(
                current, candidate_obj, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            {
                std::lock_guard<std::mutex> lock(incumbent_mutex);
                if (candidate_obj < recorded_primal_obj_) {
                    recorded_primal_obj_ = candidate_obj;
                    best_incumbent_primal = candidate_primal;
                }
            }
            return true;
        }
    }
    return false;
}

bool IncumbentManager::has_incumbent() const {
    return best_incumbent_objective.load(std::memory_order_relaxed) < (std::numeric_limits<double>::infinity() / 2.0);
}

double IncumbentManager::get_objective() const {
    return best_incumbent_objective.load(std::memory_order_relaxed);
}

std::vector<double> IncumbentManager::get_primal() const {
    std::lock_guard<std::mutex> lock(incumbent_mutex);
    return best_incumbent_primal;
}

// -----------------------------------------------------------------------------
// Internal Helpers & LP Solver
// -----------------------------------------------------------------------------

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
    const ParallelOptions& options,
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

struct SharedPseudoCosts {
    std::mutex mutex;
    std::vector<VariablePseudoCost> costs;
};

double compute_tree_lower_bound(
    const ThreadSafeNodeQueue& queue,
    const std::atomic<double>* worker_bounds,
    std::size_t num_threads,
    double best_incumbent) {
    double min_lb = queue.min_lower_bound();
    for (std::size_t i = 0; i < num_threads; ++i) {
        const double w_lb = worker_bounds[i].load(std::memory_order_relaxed);
        if (w_lb < min_lb) {
            min_lb = w_lb;
        }
    }
    if (min_lb > best_incumbent) {
        min_lb = best_incumbent;
    }
    return min_lb;
}

void worker_loop(
    std::size_t thread_id,
    const model::Model& root_model,
    const ParallelOptions& options,
    ThreadSafeNodeQueue& queue,
    IncumbentManager& incumbent,
    std::atomic<std::size_t>& next_node_id,
    std::atomic<std::size_t>& total_nodes_explored,
    std::atomic<std::size_t>& total_lp_iterations,
    std::atomic<std::size_t>& total_heuristics_found,
    std::atomic<double>* worker_bounds,
    std::size_t num_threads,
    SharedPseudoCosts& shared_pseudo_costs,
    const std::chrono::steady_clock::time_point start_time,
    std::stop_token stop_token) {

    bool was_active = false;
    model::Model node_model = root_model;

    while (!stop_token.stop_requested() && !queue.is_stopped()) {
        const auto now = std::chrono::steady_clock::now();
        const auto time_spent = std::chrono::duration<double>(now - start_time).count();
        if (time_spent > options.time_limit_seconds) {
            queue.request_stop();
            break;
        }
        if (total_nodes_explored.load(std::memory_order_relaxed) >= options.max_nodes) {
            queue.request_stop();
            break;
        }

        const double current_incumbent = incumbent.best_incumbent_objective.load(std::memory_order_relaxed);
        const double prune_cutoff = current_incumbent - options.absolute_gap_tolerance;

        // Check relative gap tolerance
        if (incumbent.has_incumbent()) {
            const double current_inc = incumbent.get_objective();
            const double tree_lb = compute_tree_lower_bound(queue, worker_bounds, num_threads, current_inc);
            if (tree_lb > -1e15) {
                const double gap = std::abs(current_inc - tree_lb) / std::max(1.0, std::abs(current_inc));
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

        // Pruning check prior to node LP
        if (node->lower_bound >= incumbent.best_incumbent_objective.load(std::memory_order_relaxed) - options.absolute_gap_tolerance) {
            worker_bounds[thread_id].store(std::numeric_limits<double>::infinity(), std::memory_order_relaxed);
            continue;
        }

        // Configure node bounds and solve LP relaxation
        node_model.variable_lower = node->variable_lower;
        node_model.variable_upper = node->variable_upper;

        const auto node_lp_res = solve_node_lp(node_model, options, node->warm_basis);
        total_lp_iterations.fetch_add(node_lp_res.iterations, std::memory_order_relaxed);
        total_nodes_explored.fetch_add(1, std::memory_order_relaxed);

        if (node_lp_res.status != lp::reference::SolveStatus::optimal) {
            worker_bounds[thread_id].store(std::numeric_limits<double>::infinity(), std::memory_order_relaxed);
            continue;
        }

        node->lower_bound = node_lp_res.objective;
        worker_bounds[thread_id].store(node->lower_bound, std::memory_order_relaxed);

        // Pruning check after node LP
        if (node_lp_res.objective >= incumbent.best_incumbent_objective.load(std::memory_order_relaxed) - options.absolute_gap_tolerance) {
            worker_bounds[thread_id].store(std::numeric_limits<double>::infinity(), std::memory_order_relaxed);
            continue;
        }

        // Record pseudo-cost from parent branch
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

        // Check integrality
        const auto fractional_vars = find_fractional_variables(
            node_lp_res.primal, root_model.variable_type, options.integrality_tolerance);

        if (fractional_vars.empty()) {
            if (incumbent.update_if_better(node_lp_res.objective, node_lp_res.primal, options.absolute_gap_tolerance)) {
                queue.prune(incumbent.best_incumbent_objective.load(std::memory_order_relaxed) - options.absolute_gap_tolerance);
            }
            worker_bounds[thread_id].store(std::numeric_limits<double>::infinity(), std::memory_order_relaxed);
            continue;
        }

        // Primal heuristics periodically
        if (options.enable_heuristics && total_nodes_explored.load(std::memory_order_relaxed) % 10 == 0) {
            const auto hr = simple_rounding(node_model, node_lp_res.primal, options.feasibility_tolerance, options.integrality_tolerance);
            if (hr.found) {
                if (incumbent.update_if_better(hr.objective, hr.primal, options.absolute_gap_tolerance)) {
                    total_heuristics_found.fetch_add(1, std::memory_order_relaxed);
                    queue.prune(incumbent.best_incumbent_objective.load(std::memory_order_relaxed) - options.absolute_gap_tolerance);
                }
            }
        }

        // Variable selection for branching
        std::vector<VariablePseudoCost> pc_snapshot;
        {
            std::lock_guard<std::mutex> pc_lock(shared_pseudo_costs.mutex);
            pc_snapshot = shared_pseudo_costs.costs;
        }
        const std::size_t branch_var = select_branching_variable(
            node_lp_res.primal, root_model.variable_type, pc_snapshot,
            options.branching_strategy, options.integrality_tolerance);

        if (branch_var >= root_model.matrix.column_count) {
            worker_bounds[thread_id].store(std::numeric_limits<double>::infinity(), std::memory_order_relaxed);
            continue;
        }

        const double branch_val = node_lp_res.primal[branch_var];
        const double floor_val = std::floor(branch_val);
        const double ceil_val = std::ceil(branch_val);

        std::shared_ptr<BranchNode> down_child;
        std::shared_ptr<BranchNode> up_child;

        // Down Branch: x_k <= floor(x_k)
        bool down_valid = true;
        if (node->variable_lower[branch_var].is_finite() &&
            floor_val < node->variable_lower[branch_var].value - 1e-9) {
            down_valid = false;
        }
        if (down_valid) {
            down_child = std::make_shared<BranchNode>();
            down_child->id = next_node_id.fetch_add(1, std::memory_order_relaxed);
            down_child->parent_id = node->id;
            down_child->depth = node->depth + 1;
            down_child->lower_bound = node_lp_res.objective;
            down_child->branch_variable = branch_var;
            down_child->branch_value = branch_val;
            down_child->is_down_branch = true;
            down_child->variable_lower = node->variable_lower;
            down_child->variable_upper = node->variable_upper;
            down_child->variable_upper[branch_var] = model::Bound::finite(floor_val);
            down_child->warm_basis = node_lp_res.basis;
        }

        // Up Branch: x_k >= ceil(x_k)
        bool up_valid = true;
        if (node->variable_upper[branch_var].is_finite() &&
            ceil_val > node->variable_upper[branch_var].value + 1e-9) {
            up_valid = false;
        }
        if (up_valid) {
            up_child = std::make_shared<BranchNode>();
            up_child->id = next_node_id.fetch_add(1, std::memory_order_relaxed);
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
        }

        queue.push_children(std::move(down_child), std::move(up_child));
        worker_bounds[thread_id].store(std::numeric_limits<double>::infinity(), std::memory_order_relaxed);
    }

    if (was_active) {
        queue.deactivate_worker();
    }
    worker_bounds[thread_id].store(std::numeric_limits<double>::infinity(), std::memory_order_relaxed);
}

} // namespace

// -----------------------------------------------------------------------------
// Main solve_parallel Entry Point
// -----------------------------------------------------------------------------

Result solve_parallel(const model::Model& model, const ParallelOptions& options) {
    const auto start_time = std::chrono::steady_clock::now();
    Result result;

    try {
        model.validate();
    } catch (const std::exception& e) {
        result.status = lp::reference::SolveStatus::invalid_model;
        result.message = e.what();
        return result;
    }

    // Check if model contains discrete variables
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

    // Initialize root search state
    IncumbentManager incumbent;
    std::atomic<std::size_t> next_node_id{1};
    std::atomic<std::size_t> total_nodes_explored{1};
    std::atomic<std::size_t> total_lp_iterations{0};
    std::atomic<std::size_t> total_heuristics_found{0};
    std::size_t root_cuts_generated = 0;

    SharedPseudoCosts shared_pseudo_costs;
    shared_pseudo_costs.costs.resize(model.matrix.column_count);

    model::Model root_model = model;

    // 1. Solve Root Continuous LP Relaxation
    const auto root_lp = solve_node_lp(root_model, options, std::nullopt);
    total_lp_iterations.fetch_add(root_lp.iterations, std::memory_order_relaxed);

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

    double best_lower_bound = root_lp.objective;

    // Check if root continuous solution is integer feasible
    if (check_integer_feasibility(root_model, root_lp.primal, options.feasibility_tolerance, options.integrality_tolerance)) {
        result.status = lp::reference::SolveStatus::optimal;
        result.primal = root_lp.primal;
        result.objective = root_lp.objective;
        result.best_bound = root_lp.objective;
        result.relative_gap = 0.0;
        result.nodes_explored = 1;
        result.lp_iterations = total_lp_iterations.load(std::memory_order_relaxed);
        result.message = "root relaxation integer feasible (integer optimal)";
        const auto elapsed = std::chrono::steady_clock::now() - start_time;
        result.runtime_ms = std::chrono::duration<double, std::milli>(elapsed).count();
        return result;
    }

    // 2. Primal heuristics at root
    if (options.enable_heuristics) {
        const auto hr = simple_rounding(root_model, root_lp.primal, options.feasibility_tolerance, options.integrality_tolerance);
        if (hr.found) {
            if (incumbent.update_if_better(hr.objective, hr.primal, options.absolute_gap_tolerance)) {
                total_heuristics_found.fetch_add(1, std::memory_order_relaxed);
            }
        }

        const auto fp = feasibility_pump(root_model, root_lp.primal, options.max_pump_iterations, options.feasibility_tolerance, options.integrality_tolerance);
        if (fp.found) {
            if (incumbent.update_if_better(fp.objective, fp.primal, options.absolute_gap_tolerance)) {
                total_heuristics_found.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // 3. Gomory cuts at root
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
                root_cuts_generated = cuts.size();

                const auto cut_lp = solve_node_lp(root_model, options, root_lp.basis);
                total_lp_iterations.fetch_add(cut_lp.iterations, std::memory_order_relaxed);
                if (cut_lp.status == lp::reference::SolveStatus::optimal) {
                    current_primal = cut_lp.primal;
                    current_obj = cut_lp.objective;
                    current_basis = cut_lp.basis;
                    best_lower_bound = std::max(best_lower_bound, current_obj);

                    if (check_integer_feasibility(root_model, current_primal, options.feasibility_tolerance, options.integrality_tolerance)) {
                        incumbent.update_if_better(current_obj, current_primal, options.absolute_gap_tolerance);
                    }
                }
            }
        } catch (...) {
        }
    }

    // Check if root cuts closed the optimality gap
    if (incumbent.has_incumbent() && best_lower_bound > -1e15) {
        const double gap = std::abs(incumbent.get_objective() - best_lower_bound) / std::max(1.0, std::abs(incumbent.get_objective()));
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
            const auto elapsed = std::chrono::steady_clock::now() - start_time;
            result.runtime_ms = std::chrono::duration<double, std::milli>(elapsed).count();
            return result;
        }
    }

    // 4. Initialize Pseudo-Costs and Evaluate Strong Branching at Root
    if (options.enable_strong_branching && current_basis.has_value()) {
        try {
            StrongBranchingOptions sb_opts;
            sb_opts.integrality_tolerance = options.integrality_tolerance;
            sb_opts.feasibility_tolerance = options.feasibility_tolerance;
            sb_opts.update_pseudo_costs = true;
            std::vector<VariablePseudoCost> initial_pc = shared_pseudo_costs.costs;
            const auto sb_res = evaluate_strong_branching(
                root_model, current_primal, current_obj, current_basis, sb_opts, &initial_pc);

            if (sb_res.subproblem_infeasible) {
                result.status = lp::reference::SolveStatus::infeasible;
                result.message = "proven infeasible by strong branching at root";
                const auto elapsed = std::chrono::steady_clock::now() - start_time;
                result.runtime_ms = std::chrono::duration<double, std::milli>(elapsed).count();
                return result;
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

    // 5. Branch root node to populate initial queue
    const std::size_t root_branch_var = select_branching_variable(
        current_primal, root_model.variable_type, shared_pseudo_costs.costs,
        options.branching_strategy, options.integrality_tolerance);

    if (root_branch_var >= root_model.matrix.column_count) {
        // No fractional variable to branch on
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
        const auto elapsed = std::chrono::steady_clock::now() - start_time;
        result.runtime_ms = std::chrono::duration<double, std::milli>(elapsed).count();
        return result;
    }

    ThreadSafeNodeQueue queue;

    const double root_branch_val = current_primal[root_branch_var];
    const double floor_val = std::floor(root_branch_val);
    const double ceil_val = std::ceil(root_branch_val);

    std::shared_ptr<BranchNode> down_child;
    std::shared_ptr<BranchNode> up_child;

    if (!root_model.variable_lower[root_branch_var].is_finite() ||
        floor_val >= root_model.variable_lower[root_branch_var].value - 1e-9) {
        down_child = std::make_shared<BranchNode>();
        down_child->id = next_node_id.fetch_add(1, std::memory_order_relaxed);
        down_child->parent_id = 0;
        down_child->depth = 1;
        down_child->lower_bound = current_obj;
        down_child->branch_variable = root_branch_var;
        down_child->branch_value = root_branch_val;
        down_child->is_down_branch = true;
        down_child->variable_lower = root_model.variable_lower;
        down_child->variable_upper = root_model.variable_upper;
        down_child->variable_upper[root_branch_var] = model::Bound::finite(floor_val);
        down_child->warm_basis = current_basis;
    }

    if (!root_model.variable_upper[root_branch_var].is_finite() ||
        ceil_val <= root_model.variable_upper[root_branch_var].value + 1e-9) {
        up_child = std::make_shared<BranchNode>();
        up_child->id = next_node_id.fetch_add(1, std::memory_order_relaxed);
        up_child->parent_id = 0;
        up_child->depth = 1;
        up_child->lower_bound = current_obj;
        up_child->branch_variable = root_branch_var;
        up_child->branch_value = root_branch_val;
        up_child->is_down_branch = false;
        up_child->variable_lower = root_model.variable_lower;
        up_child->variable_upper = root_model.variable_upper;
        up_child->variable_lower[root_branch_var] = model::Bound::finite(ceil_val);
        up_child->warm_basis = current_basis;
    }

    queue.push_children(std::move(down_child), std::move(up_child));

    // 5. Worker Thread Pool
    const std::size_t num_threads = std::max<std::size_t>(1, options.num_threads);
    auto worker_bounds = std::make_unique<std::atomic<double>[]>(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
        worker_bounds[i].store(std::numeric_limits<double>::infinity(), std::memory_order_relaxed);
    }

    {
        std::vector<std::jthread> workers;
        workers.reserve(num_threads);
        for (std::size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([&, i](std::stop_token st) {
                worker_loop(i, root_model, options, queue, incumbent,
                            next_node_id, total_nodes_explored, total_lp_iterations,
                            total_heuristics_found, worker_bounds.get(), num_threads,
                            shared_pseudo_costs, start_time, st);
            });
        }
        // jthreads automatically join on destruction of the vector
    }

    // 6. Assemble Final Result
    const auto end_time = std::chrono::steady_clock::now();
    result.runtime_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    result.nodes_explored = total_nodes_explored.load(std::memory_order_relaxed);
    result.lp_iterations = total_lp_iterations.load(std::memory_order_relaxed);
    result.cuts_generated = root_cuts_generated;
    result.heuristics_found = total_heuristics_found.load(std::memory_order_relaxed);

    if (incumbent.has_incumbent()) {
        result.status = lp::reference::SolveStatus::optimal;
        result.primal = incumbent.get_primal();
        result.objective = incumbent.get_objective();

        const double final_lb = compute_tree_lower_bound(queue, worker_bounds.get(), num_threads, result.objective);
        if (queue.empty() && queue.active_workers() == 0) {
            result.best_bound = result.objective;
        } else {
            result.best_bound = (std::abs(final_lb) > 1e15 ? result.objective : final_lb);
        }
        result.relative_gap = std::max(0.0, std::abs(result.objective - result.best_bound) / std::max(1.0, std::abs(result.objective)));
        result.message = "parallel tree search MILP optimum";
    } else {
        result.status = lp::reference::SolveStatus::infeasible;
        result.message = "no integer feasible solution found";
    }

    return result;
}

} // namespace markov_cero::milp
