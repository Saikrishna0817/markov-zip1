#include "markov_cero/qp/verifier.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace markov_cero::qp {

namespace {

std::vector<double> multiply_A(const linalg::SparseCsc& A,
                               const std::vector<double>& x) {
    std::vector<double> y(A.rows, 0.0);
    for (std::size_t j = 0; j < A.columns; ++j) {
        const double xj = x[j];
        if (std::abs(xj) < 1e-20) {
            continue;
        }
        const std::size_t start = A.column_offsets[j];
        const std::size_t end = A.column_offsets[j + 1];
        for (std::size_t k = start; k < end; ++k) {
            y[A.row_indices[k]] += A.values[k] * xj;
        }
    }
    return y;
}

std::vector<double> multiply_AT(const linalg::SparseCsc& A,
                                const std::vector<double>& y) {
    std::vector<double> x(A.columns, 0.0);
    for (std::size_t j = 0; j < A.columns; ++j) {
        double sum = 0.0;
        const std::size_t start = A.column_offsets[j];
        const std::size_t end = A.column_offsets[j + 1];
        for (std::size_t k = start; k < end; ++k) {
            sum += A.values[k] * y[A.row_indices[k]];
        }
        x[j] = sum;
    }
    return x;
}

} // namespace

QpVerificationReport verify_qp_solution(const QuadraticModel& model,
                                        const QpSolution& solution,
                                        double tolerance) {
    QpVerificationReport report;
    const std::size_t n = model.num_variables();
    const std::size_t m = model.num_constraints();

    if (solution.x.size() != n) {
        report.passed = false;
        report.failure_reason = "Solution vector dimension does not match variables";
        return report;
    }

    // 1. Primal feasibility
    const std::vector<double> Ax = multiply_A(model.A, solution.x);
    double max_rel_primal = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        if (!std::isinf(model.l[i]) && Ax[i] < model.l[i]) {
            const double viol = model.l[i] - Ax[i];
            const double scale = 1.0 + std::max(std::abs(Ax[i]), std::abs(model.l[i]));
            report.maximum_primal_violation =
                std::max(report.maximum_primal_violation, viol);
            max_rel_primal = std::max(max_rel_primal, viol / scale);
        }
        if (!std::isinf(model.u[i]) && Ax[i] > model.u[i]) {
            const double viol = Ax[i] - model.u[i];
            const double scale = 1.0 + std::max(std::abs(Ax[i]), std::abs(model.u[i]));
            report.maximum_primal_violation =
                std::max(report.maximum_primal_violation, viol);
            max_rel_primal = std::max(max_rel_primal, viol / scale);
        }
    }

    // 2. Integrality feasibility
    if (!model.variable_types.empty()) {
        for (std::size_t j = 0; j < n; ++j) {
            if (j < model.variable_types.size() &&
                model.variable_types[j] != model::VariableType::continuous) {
                const double val = solution.x[j];
                const double nearest = std::round(val);
                const double viol = std::abs(val - nearest);
                report.maximum_integrality_violation =
                    std::max(report.maximum_integrality_violation, viol);
            }
        }
    }

    // 3. Dual feasibility (stationarity: P x + q + A^T y = 0)
    double max_rel_dual = 0.0;
    if (solution.y.size() == m) {
        const std::vector<double> Px = model.P.multiply(solution.x);
        const std::vector<double> ATy = multiply_AT(model.A, solution.y);
        for (std::size_t j = 0; j < n; ++j) {
            const double stat = std::abs(Px[j] + model.q[j] + ATy[j]);
            const double scale = 1.0 + std::max({std::abs(Px[j]), std::abs(ATy[j]),
                                                 std::abs(model.q[j])});
            report.maximum_dual_violation =
                std::max(report.maximum_dual_violation, stat);
            max_rel_dual = std::max(max_rel_dual, stat / scale);
        }

        // 4. Complementary slackness
        for (std::size_t i = 0; i < m; ++i) {
            const double yi = solution.y[i];
            if (std::abs(yi) > tolerance) {
                double dist = 0.0;
                const bool finite_l = !std::isinf(model.l[i]);
                const bool finite_u = !std::isinf(model.u[i]);
                if (finite_l && finite_u) {
                    dist = std::min(std::abs(Ax[i] - model.l[i]),
                                    std::abs(Ax[i] - model.u[i]));
                } else if (finite_l) {
                    dist = std::abs(Ax[i] - model.l[i]);
                } else if (finite_u) {
                    dist = std::abs(Ax[i] - model.u[i]);
                }
                const double comp = dist * std::abs(yi);
                report.maximum_complementarity_violation =
                    std::max(report.maximum_complementarity_violation, comp);
            }
        }
    }

    // 5. Objective check
    const double energy = model.P.evaluate_energy(solution.x);
    double linear = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        linear += model.q[j] * solution.x[j];
    }
    const double internal_obj = 0.5 * energy + linear;
    const double expected_obj =
        (model.sense == model::ObjectiveSense::maximize)
            ? (-internal_obj + model.objective_offset)
            : (internal_obj + model.objective_offset);

    report.objective_discrepancy =
        std::abs(expected_obj - solution.objective_value);
    const double rel_obj_disc =
        report.objective_discrepancy / (1.0 + std::abs(expected_obj));

    std::ostringstream reasons;
    bool ok = true;
    if (max_rel_primal > tolerance) {
        ok = false;
        reasons << "Primal violation (" << max_rel_primal
                << " > " << tolerance << "); ";
    }
    if (report.maximum_integrality_violation > tolerance) {
        ok = false;
        reasons << "Integrality violation ("
                << report.maximum_integrality_violation << " > " << tolerance
                << "); ";
    }
    if (max_rel_dual > tolerance * 10.0) {
        ok = false;
        reasons << "Dual violation (" << max_rel_dual << " > "
                << tolerance * 10.0 << "); ";
    }
    if (rel_obj_disc > tolerance) {
        ok = false;
        reasons << "Objective discrepancy (" << rel_obj_disc
                << " > " << tolerance << "); ";
    }

    report.passed = ok;
    report.failure_reason = reasons.str();
    return report;
}

} // namespace markov_cero::qp
