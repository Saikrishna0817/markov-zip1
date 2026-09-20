#pragma once

#include "markov_cero/lp/dual/dual_simplex.hpp"
#include "markov_cero/model/model.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace markov_cero::milp {

struct BranchNode {
    std::size_t id{0};
    std::size_t parent_id{0};
    std::size_t depth{0};
    double lower_bound{0.0};
    std::size_t branch_variable{0};
    double branch_value{0.0};
    bool is_down_branch{true};
    std::vector<model::Bound> variable_lower;
    std::vector<model::Bound> variable_upper;
    std::optional<lp::dual::BasisState> warm_basis;
};

struct NodeCompareBestBound {
    bool operator()(const std::shared_ptr<BranchNode>& a,
                    const std::shared_ptr<BranchNode>& b) const {
        if (!a || !b) {
            return a != nullptr;
        }
        if (a->lower_bound != b->lower_bound) {
            return a->lower_bound >
                   b->lower_bound; // Min-heap: smallest lower bound has highest priority
        }
        return a->depth < b->depth; // Tie-breaker: deeper node first (dive)
    }
};

} // namespace markov_cero::milp
