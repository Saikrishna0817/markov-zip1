#pragma once

#include "markov_cero/lp/dual/dual_simplex.hpp"
#include "markov_cero/milp/branch_node.hpp"
#include "markov_cero/model/model.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace markov_cero::milp {

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

    void push_branch_children(const BranchNode& parent, std::size_t branch_var, double branch_val,
                              double lower_bound,
                              const std::optional<lp::dual::BasisState>& warm_basis,
                              std::atomic<std::size_t>& next_node_id);

    [[nodiscard]] std::shared_ptr<BranchNode> pop_node(bool was_active, double prune_cutoff,
                                                       bool& became_active);

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

using WorkQueue = ThreadSafeNodeQueue;

} // namespace markov_cero::milp
