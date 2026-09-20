#include "markov_cero/milp/shared_incumbent.hpp"
#include "markov_cero/milp/work_queue.hpp"

#include <algorithm>
#include <limits>

namespace markov_cero::milp {

IncumbentManager::IncumbentManager(double initial_obj)
    : best_incumbent_objective(initial_obj), recorded_primal_obj_(initial_obj) {}

bool IncumbentManager::update_if_better(double candidate_obj,
                                        const std::vector<double>& candidate_primal, double eps) {
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
    return best_incumbent_objective.load(std::memory_order_relaxed) <
           (std::numeric_limits<double>::infinity() / 2.0);
}

double IncumbentManager::get_objective() const {
    return best_incumbent_objective.load(std::memory_order_relaxed);
}

std::vector<double> IncumbentManager::get_primal() const {
    std::lock_guard<std::mutex> lock(incumbent_mutex);
    return best_incumbent_primal;
}

double compute_tree_lower_bound(const ThreadSafeNodeQueue& queue,
                                const std::atomic<double>* worker_bounds, std::size_t num_threads,
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

} // namespace markov_cero::milp
