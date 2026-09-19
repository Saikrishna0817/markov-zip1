#pragma once

#include "markov_cero/model/model.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace markov_cero::verify {

struct Tolerance final {
    double absolute{1e-7};
    double relative{1e-7};
    void validate() const;
};

struct Candidate final {
    std::vector<double> primal;
    double claimed_objective{0.0};
};

struct Violation final {
    std::string category;
    std::size_t index{};
    double actual{};
    double bound{};
    double magnitude{};
    double allowance{};
};

struct PrimalVerificationReport final {
    bool passed{false};
    double recomputed_objective{0.0};
    double objective_difference{0.0};
    double maximum_row_violation{0.0};
    double maximum_variable_violation{0.0};
    double maximum_integrality_violation{0.0};
    std::vector<Violation> violations;
};

[[nodiscard]] PrimalVerificationReport verify_primal(
    const model::Model& model,
    const Candidate& candidate,
    const Tolerance& feasibility_tolerance = {},
    const Tolerance& objective_tolerance = {},
    double integrality_tolerance = 1e-6);

} // namespace markov_cero::verify
