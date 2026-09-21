#pragma once

#include "markov_cero/lp/dual/dual_simplex.hpp"
#include "markov_cero/lp/reference/revised_simplex.hpp"
#include "markov_cero/milp/milp_solver.hpp"
#include "markov_cero/model/model.hpp"

#include <optional>
#include <vector>

namespace markov_cero::milp {

struct NodeLpResult {
    lp::reference::SolveStatus status{lp::reference::SolveStatus::infeasible};
    std::vector<double> primal;
    double objective{0.0};
    std::size_t iterations{0};
    std::optional<lp::dual::BasisState> basis;
};

[[nodiscard]] NodeLpResult solve_node_relaxation(
    const model::Model& node_model,
    const Options& options,
    const std::optional<lp::dual::BasisState>& warm_start);

} // namespace markov_cero::milp
