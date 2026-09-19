#pragma once

#include "markov_cero/lp/reference/revised_simplex.hpp"
#include "markov_cero/milp/branch_selector.hpp"
#include "markov_cero/model/model.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace markov_cero::milp {

enum class NodeSelectionStrategy {
    best_bound,
    depth_first,
    best_bound_plunge
};

struct Options {
    std::size_t max_nodes{50000};
    std::size_t max_iterations{500000};
    double time_limit_seconds{60.0};
    double relative_gap_tolerance{1e-4};
    double absolute_gap_tolerance{1e-6};
    double integrality_tolerance{1e-6};
    double feasibility_tolerance{1e-7};
    bool enable_warm_start{true};
    bool enable_cuts{true};
    bool enable_heuristics{true};
    std::size_t max_cut_rounds{5};
    std::size_t max_pump_iterations{10};
    NodeSelectionStrategy node_strategy{NodeSelectionStrategy::best_bound_plunge};
    BranchingStrategy branching_strategy{BranchingStrategy::pseudo_cost};
};

struct Result {
    lp::reference::SolveStatus status{lp::reference::SolveStatus::infeasible};
    std::vector<double> primal;
    double objective{0.0};
    double best_bound{0.0};
    double relative_gap{0.0};
    std::size_t nodes_explored{0};
    std::size_t lp_iterations{0};
    std::size_t cuts_generated{0};
    std::size_t heuristics_found{0};
    double runtime_ms{0.0};
    std::string message;
};

[[nodiscard]] Result solve(const model::Model& model, const Options& options = {});

} // namespace markov_cero::milp
