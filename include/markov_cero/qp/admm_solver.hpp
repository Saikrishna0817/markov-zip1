#pragma once

#include "markov_cero/qp/kkt.hpp"
#include "markov_cero/qp/model.hpp"

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

namespace markov_cero::qp {

enum class QpStatus {
    optimal,
    primal_infeasible,
    dual_infeasible,
    iteration_limit,
    time_limit,
    non_convex,
    numerical_error
};

[[nodiscard]] const char* to_string(QpStatus status) noexcept;

struct QpOptions {
    double absolute_tolerance{1e-4};
    double relative_tolerance{1e-4};
    double primal_infeasible_tolerance{1e-5};
    double dual_infeasible_tolerance{1e-5};
    double sigma{1e-6};
    double rho_init{0.1};
    double alpha{1.6};
    std::size_t max_iterations{4000};
    double time_limit_seconds{60.0};
    bool adaptive_rho{true};
    std::size_t adaptive_rho_interval{25};
    bool verbose{false};
};

struct QpSolution {
    QpStatus status{QpStatus::numerical_error};
    double objective_value{0.0};
    std::vector<double> x;
    std::vector<double> z;
    std::vector<double> y;
    std::size_t iterations{0};
    double solve_time_seconds{0.0};
    double primal_residual{0.0};
    double dual_residual{0.0};
};

class AdmmQpSolver {
public:
    explicit AdmmQpSolver(QpOptions options = {}) : options_(options) {}

    [[nodiscard]] QpSolution solve(const QuadraticModel& model);

private:
    QpOptions options_;
};

/// High-level function to solve a QuadraticModel using ADMM.
[[nodiscard]] QpSolution solve_qp(const QuadraticModel& model, const QpOptions& options = {});

} // namespace markov_cero::qp
