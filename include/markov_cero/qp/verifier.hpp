#pragma once

#include "markov_cero/qp/admm_solver.hpp"
#include "markov_cero/qp/model.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace markov_cero::qp {

struct QpVerificationReport {
    bool passed{false};
    double maximum_primal_violation{0.0};
    double maximum_dual_violation{0.0};
    double maximum_complementarity_violation{0.0};
    double maximum_integrality_violation{0.0};
    double objective_discrepancy{0.0};
    std::string failure_reason;
};

/// Independently validates a candidate QP solution against KKT conditions.
[[nodiscard]] QpVerificationReport verify_qp_solution(
    const QuadraticModel& model,
    const QpSolution& solution,
    double tolerance = 1e-4);

} // namespace markov_cero::qp
