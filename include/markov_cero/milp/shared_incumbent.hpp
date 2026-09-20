#pragma once

#include <atomic>
#include <limits>
#include <mutex>
#include <vector>

namespace markov_cero::milp {

class ThreadSafeNodeQueue;

/// Thread-safe atomic incumbent management.
class IncumbentManager {
  public:
    explicit IncumbentManager(double initial_obj = std::numeric_limits<double>::infinity());

    bool update_if_better(double candidate_obj, const std::vector<double>& candidate_primal,
                          double eps = 1e-9);

    [[nodiscard]] bool has_incumbent() const;
    [[nodiscard]] double get_objective() const;
    [[nodiscard]] std::vector<double> get_primal() const;

    std::atomic<double> best_incumbent_objective;
    mutable std::mutex incumbent_mutex;
    std::vector<double> best_incumbent_primal;

  private:
    double recorded_primal_obj_{std::numeric_limits<double>::infinity()};
};

using SharedIncumbent = IncumbentManager;

[[nodiscard]] double compute_tree_lower_bound(const ThreadSafeNodeQueue& queue,
                                              const std::atomic<double>* worker_bounds,
                                              std::size_t num_threads, double best_incumbent);

} // namespace markov_cero::milp
