#include "markov_cero/verify/primal_verifier.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace markov_cero::verify {
namespace {
double allowance(double actual, double expected, const Tolerance& tolerance) {
    const double value = tolerance.absolute +
                         tolerance.relative * std::max({1.0, std::abs(actual), std::abs(expected)});
    if (!std::isfinite(value))
        throw std::invalid_argument("tolerance allowance overflow");
    return value;
}
void check_lower(double actual, const model::Bound& bound, const Tolerance& tolerance,
                 const char* category, std::size_t index, PrimalVerificationReport& report,
                 double& maximum) {
    if (!bound.is_finite())
        return;
    const double magnitude = std::max(0.0, bound.value - actual);
    const double permitted = allowance(actual, bound.value, tolerance);
    maximum = std::max(maximum, magnitude);
    if (magnitude > permitted)
        report.violations.push_back({category, index, actual, bound.value, magnitude, permitted});
}
void check_upper(double actual, const model::Bound& bound, const Tolerance& tolerance,
                 const char* category, std::size_t index, PrimalVerificationReport& report,
                 double& maximum) {
    if (!bound.is_finite())
        return;
    const double magnitude = std::max(0.0, actual - bound.value);
    const double permitted = allowance(actual, bound.value, tolerance);
    maximum = std::max(maximum, magnitude);
    if (magnitude > permitted)
        report.violations.push_back({category, index, actual, bound.value, magnitude, permitted});
}
} // namespace
void Tolerance::validate() const {
    if (!std::isfinite(absolute) || !std::isfinite(relative) || absolute < 0.0 || relative < 0.0 ||
        absolute > 1e-4 || relative > 1e-4)
        throw std::invalid_argument("tolerances must be finite and nonnegative");
}
PrimalVerificationReport verify_primal(const model::Model& problem, const Candidate& candidate,
                                       const Tolerance& feasibility_tolerance,
                                       const Tolerance& objective_tolerance,
                                       double integrality_tolerance) {
    problem.validate();
    feasibility_tolerance.validate();
    objective_tolerance.validate();
    if (!std::isfinite(integrality_tolerance) || integrality_tolerance < 0.0 ||
        integrality_tolerance >= 0.5)
        throw std::invalid_argument("integrality tolerance must be finite and nonnegative");
    if (candidate.primal.size() != problem.matrix.column_count)
        throw std::invalid_argument("candidate dimension mismatch");
    if (!std::isfinite(candidate.claimed_objective))
        throw std::invalid_argument("claimed objective must be finite");
    PrimalVerificationReport report;
    for (std::size_t j = 0; j < candidate.primal.size(); ++j) {
        const double x = candidate.primal[j];
        if (!std::isfinite(x))
            throw std::invalid_argument("candidate contains non-finite value");
        check_lower(x, problem.variable_lower[j], feasibility_tolerance, "variable_lower", j,
                    report, report.maximum_variable_violation);
        check_upper(x, problem.variable_upper[j], feasibility_tolerance, "variable_upper", j,
                    report, report.maximum_variable_violation);
        if (problem.variable_type[j] != model::VariableType::continuous) {
            const double lower_integer = std::floor(x);
            const double upper_integer = std::ceil(x);
            const double nearest_integer =
                (x - lower_integer <= upper_integer - x) ? lower_integer : upper_integer;
            const double residual = std::abs(x - nearest_integer);
            report.maximum_integrality_violation =
                std::max(report.maximum_integrality_violation, residual);
            if (residual > integrality_tolerance)
                report.violations.push_back(
                    {"integrality", j, x, nearest_integer, residual, integrality_tolerance});
        }
    }
    const auto activity = problem.matrix.multiply(candidate.primal);
    for (std::size_t i = 0; i < activity.size(); ++i) {
        check_lower(activity[i], problem.row_lower[i], feasibility_tolerance, "row_lower", i,
                    report, report.maximum_row_violation);
        check_upper(activity[i], problem.row_upper[i], feasibility_tolerance, "row_upper", i,
                    report, report.maximum_row_violation);
    }
    long double objective = problem.objective_offset;
    for (std::size_t j = 0; j < candidate.primal.size(); ++j)
        objective += static_cast<long double>(problem.objective[j]) * candidate.primal[j];
    report.recomputed_objective = static_cast<double>(objective);
    if (!std::isfinite(report.recomputed_objective))
        throw std::overflow_error("recomputed objective is non-finite");
    report.objective_difference =
        std::abs(report.recomputed_objective - candidate.claimed_objective);
    const double objective_allowance =
        allowance(report.recomputed_objective, candidate.claimed_objective, objective_tolerance);
    if (report.objective_difference > objective_allowance)
        report.violations.push_back({"objective", 0U, candidate.claimed_objective,
                                     report.recomputed_objective, report.objective_difference,
                                     objective_allowance});
    report.passed = report.violations.empty();
    return report;
}
} // namespace markov_cero::verify
