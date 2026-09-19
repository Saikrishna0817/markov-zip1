#pragma once

#include "markov_cero/model/model.hpp"

#include <cstddef>
#include <vector>

namespace markov_cero::milp {

enum class BranchingStrategy {
    most_fractional,
    pseudo_cost,
    strong_branching,
    reliability
};

struct VariablePseudoCost {
    double down_sum{0.0};
    double up_sum{0.0};
    std::size_t down_count{0};
    std::size_t up_count{0};

    [[nodiscard]] double down_cost() const noexcept;
    [[nodiscard]] double up_cost() const noexcept;
    void record_down(double delta_obj, double fraction) noexcept;
    void record_up(double delta_obj, double fraction) noexcept;
};

[[nodiscard]] std::vector<std::size_t> find_fractional_variables(
    const std::vector<double>& primal,
    const std::vector<model::VariableType>& types,
    double integrality_tol = 1e-6);

[[nodiscard]] std::size_t select_most_fractional(
    const std::vector<double>& primal,
    const std::vector<std::size_t>& candidates);

[[nodiscard]] std::size_t select_pseudo_cost(
    const std::vector<double>& primal,
    const std::vector<std::size_t>& candidates,
    const std::vector<VariablePseudoCost>& pseudo_costs);

[[nodiscard]] std::size_t select_branching_variable(
    const std::vector<double>& primal,
    const std::vector<model::VariableType>& types,
    const std::vector<VariablePseudoCost>& pseudo_costs,
    BranchingStrategy strategy,
    double integrality_tol = 1e-6);

} // namespace markov_cero::milp
