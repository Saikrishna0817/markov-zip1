#pragma once

#include "markov_cero/lp/dual/dual_simplex.hpp"
#include "markov_cero/lp/reference/revised_simplex.hpp"
#include "markov_cero/milp/branch_node.hpp"
#include "markov_cero/milp/branch_selector.hpp"
#include "markov_cero/milp/milp_solver.hpp"
#include "markov_cero/model/model.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace markov_cero::milp {

struct ParallelOptions {
    std::size_t num_threads{4};
    double time_limit_seconds{60.0};
    std::size_t max_nodes{50000};
    double relative_gap_tolerance{1e-4};
    double absolute_gap_tolerance{1e-6};
    double integrality_tolerance{1e-6};
    double feasibility_tolerance{1e-7};
    std::size_t max_iterations{500000};
    bool enable_warm_start{true};
    bool enable_cuts{true};
    bool enable_mir_cuts{true};
    bool enable_heuristics{true};
    bool enable_strong_branching{true};
    std::size_t max_cut_rounds{5};
    std::size_t max_pump_iterations{10};
    BranchingStrategy branching_strategy{BranchingStrategy::pseudo_cost};
};

using ParallelResult = Result;

/// Thread-safe min-heap priority queue of BranchNodes prioritizing lowest lower bound.
class ThreadSafeNodeQueue {
public:
    ThreadSafeNodeQueue() = default;
    ~ThreadSafeNodeQueue() = default;

    ThreadSafeNodeQueue(const ThreadSafeNodeQueue&) = delete;
    ThreadSafeNodeQueue& operator=(const ThreadSafeNodeQueue&) = delete;
    ThreadSafeNodeQueue(ThreadSafeNodeQueue&&) = delete;
    ThreadSafeNodeQueue& operator=(ThreadSafeNodeQueue&&) = delete;

    void push(std::shared_ptr<BranchNode> node);
    void push_children(std::shared_ptr<BranchNode> left, std::shared_ptr<BranchNode> right);

    [[nodiscard]] std::shared_ptr<BranchNode> pop_node(bool was_active, double prune_cutoff, bool& became_active);

    void deactivate_worker();
    void prune(double cutoff);
    void request_stop();
    [[nodiscard]] bool is_stopped() const;
    [[nodiscard]] bool empty() const;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t active_workers() const;
    [[nodiscard]] double min_lower_bound() const;
    void notify_all();

private:
    void prune_locked(double cutoff);

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::shared_ptr<BranchNode>> heap_;
    std::size_t active_workers_{0};
    bool stopped_{false};
};

/// Thread-safe atomic incumbent management.
class IncumbentManager {
public:
    explicit IncumbentManager(double initial_obj = std::numeric_limits<double>::infinity());

    bool update_if_better(double candidate_obj, const std::vector<double>& candidate_primal, double eps = 1e-9);

    [[nodiscard]] bool has_incumbent() const;
    [[nodiscard]] double get_objective() const;
    [[nodiscard]] std::vector<double> get_primal() const;

    std::atomic<double> best_incumbent_objective;
    mutable std::mutex incumbent_mutex;
    std::vector<double> best_incumbent_primal;

private:
    double recorded_primal_obj_{std::numeric_limits<double>::infinity()};
};

[[nodiscard]] Result solve_parallel(const model::Model& model, const ParallelOptions& options = {});

} // namespace markov_cero::milp
