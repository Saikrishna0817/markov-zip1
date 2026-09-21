#include "markov_cero/qp/admm_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace markov_cero::qp {

const char* to_string(QpStatus status) noexcept {
    switch (status) {
    case QpStatus::optimal:
        return "optimal";
    case QpStatus::primal_infeasible:
        return "primal_infeasible";
    case QpStatus::dual_infeasible:
        return "dual_infeasible";
    case QpStatus::iteration_limit:
        return "iteration_limit";
    case QpStatus::time_limit:
        return "time_limit";
    case QpStatus::non_convex:
        return "non_convex";
    case QpStatus::numerical_error:
        return "numerical_error";
    }
    return "unknown";
}

namespace {

double inf_norm(const std::vector<double>& v) noexcept {
    double max_val = 0.0;
    for (double x : v) {
        max_val = std::max(max_val, std::abs(x));
    }
    return max_val;
}

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

double project_bound(double v, double l, double u) noexcept {
    if (v < l) {
        return l;
    }
    if (v > u) {
        return u;
    }
    return v;
}

} // namespace

QpSolution AdmmQpSolver::solve(const QuadraticModel& model) {
    const auto start_time = std::chrono::steady_clock::now();
    QpSolution sol;

    if (!check_convexity(model.P)) {
        sol.status = QpStatus::non_convex;
        return sol;
    }

    const std::size_t n = model.num_variables();
    const std::size_t m = model.num_constraints();

    if (n == 0) {
        sol.status = QpStatus::optimal;
        sol.objective_value = model.objective_offset;
        return sol;
    }

    std::vector<double> x(n, 0.0);
    std::vector<double> z(m, 0.0);
    std::vector<double> y(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        z[i] = project_bound(0.0, model.l[i], model.u[i]);
    }

    std::vector<double> rho(m, options_.rho_init);

    KktSolver kkt;
    if (!kkt.factorize(model.P, model.A, options_.sigma, rho)) {
        sol.status = QpStatus::numerical_error;
        return sol;
    }

    std::vector<double> rhs_x(n, 0.0);
    std::vector<double> rhs_z(m, 0.0);
    std::vector<double> x_tilde(n, 0.0);
    std::vector<double> nu(m, 0.0);
    std::vector<double> z_tilde(m, 0.0);
    std::vector<double> x_hat(n, 0.0);
    std::vector<double> z_hat(m, 0.0);
    std::vector<double> x_prev(n, 0.0);
    std::vector<double> y_prev(m, 0.0);

    const double alpha = options_.alpha;
    const double sigma = options_.sigma;

    for (std::size_t iter = 0; iter < options_.max_iterations; ++iter) {
        sol.iterations = iter + 1;

        const auto now = std::chrono::steady_clock::now();
        const double elapsed =
            std::chrono::duration<double>(now - start_time).count();
        if (elapsed > options_.time_limit_seconds) {
            sol.status = QpStatus::time_limit;
            break;
        }

        x_prev = x;
        y_prev = y;

        // 1. Construct RHS for KKT system
        for (std::size_t j = 0; j < n; ++j) {
            rhs_x[j] = sigma * x[j] - model.q[j];
        }
        for (std::size_t i = 0; i < m; ++i) {
            rhs_z[i] = z[i] - y[i] / rho[i];
        }

        // 2. Linear system solve
        kkt.solve(rhs_x, rhs_z, x_tilde, nu);

        // 3. Slack before relaxation
        for (std::size_t i = 0; i < m; ++i) {
            z_tilde[i] = z[i] + (nu[i] - y[i]) / rho[i];
        }

        // 4. Over-relaxation
        for (std::size_t j = 0; j < n; ++j) {
            x_hat[j] = alpha * x_tilde[j] + (1.0 - alpha) * x[j];
        }
        for (std::size_t i = 0; i < m; ++i) {
            z_hat[i] = alpha * z_tilde[i] + (1.0 - alpha) * z[i];
        }

        // 5. Projection onto constraint set [l, u]
        for (std::size_t i = 0; i < m; ++i) {
            const double arg = z_hat[i] + y[i] / rho[i];
            z[i] = project_bound(arg, model.l[i], model.u[i]);
        }

        // 6. Dual and Primal variable updates
        for (std::size_t i = 0; i < m; ++i) {
            y[i] += rho[i] * (z_hat[i] - z[i]);
        }
        x = x_hat;

        // 7. Residual computation and convergence checks
        const std::vector<double> Ax = multiply_A(model.A, x);
        std::vector<double> r_prim(m, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            r_prim[i] = Ax[i] - z[i];
        }

        const std::vector<double> Px = model.P.multiply(x);
        const std::vector<double> ATy = multiply_AT(model.A, y);
        std::vector<double> r_dual(n, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            r_dual[j] = Px[j] + model.q[j] + ATy[j];
        }

        const double norm_prim = inf_norm(r_prim);
        const double norm_dual = inf_norm(r_dual);
        sol.primal_residual = norm_prim;
        sol.dual_residual = norm_dual;

        const double eps_prim = options_.absolute_tolerance +
                                options_.relative_tolerance *
                                    std::max(inf_norm(Ax), inf_norm(z));
        const double eps_dual = options_.absolute_tolerance +
                                options_.relative_tolerance *
                                    std::max({inf_norm(Px), inf_norm(ATy),
                                              inf_norm(model.q)});

        if (norm_prim <= eps_prim && norm_dual <= eps_dual) {
            sol.status = QpStatus::optimal;
            break;
        }

        // 8. Infeasibility certificates (Banjac et al. 2019)
        if (iter % 10 == 0) {
            // Primal infeasibility check
            std::vector<double> delta_y(m, 0.0);
            for (std::size_t i = 0; i < m; ++i) {
                delta_y[i] = y[i] - y_prev[i];
            }
            const double norm_dy = inf_norm(delta_y);
            if (norm_dy > 1e-10) {
                const std::vector<double> ATdy = multiply_AT(model.A, delta_y);
                if (inf_norm(ATdy) <=
                    options_.primal_infeasible_tolerance * norm_dy) {
                    double support = 0.0;
                    bool has_infinite_ray = false;
                    for (std::size_t i = 0; i < m; ++i) {
                        if (delta_y[i] > 1e-12) {
                            if (std::isinf(model.u[i])) {
                                has_infinite_ray = true;
                                break;
                            }
                            support += model.u[i] * delta_y[i];
                        } else if (delta_y[i] < -1e-12) {
                            if (std::isinf(model.l[i])) {
                                has_infinite_ray = true;
                                break;
                            }
                            support += model.l[i] * delta_y[i];
                        }
                    }
                    if (!has_infinite_ray &&
                        support <
                            -options_.primal_infeasible_tolerance * norm_dy) {
                        sol.status = QpStatus::primal_infeasible;
                        break;
                    }
                }
            }

            // Dual infeasibility (unboundedness) check
            std::vector<double> delta_x(n, 0.0);
            for (std::size_t j = 0; j < n; ++j) {
                delta_x[j] = x[j] - x_prev[j];
            }
            const double norm_dx = inf_norm(delta_x);
            if (norm_dx > 1e-10) {
                const std::vector<double> Pdx = model.P.multiply(delta_x);
                double q_dot_dx = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    q_dot_dx += model.q[j] * delta_x[j];
                }
                if (inf_norm(Pdx) <=
                        options_.dual_infeasible_tolerance * norm_dx &&
                    q_dot_dx < -options_.dual_infeasible_tolerance * norm_dx) {
                    const std::vector<double> Adx = multiply_A(model.A, delta_x);
                    bool ray_compatible = true;
                    for (std::size_t i = 0; i < m; ++i) {
                        if (Adx[i] > options_.dual_infeasible_tolerance * norm_dx &&
                            !std::isinf(model.u[i])) {
                            ray_compatible = false;
                            break;
                        }
                        if (Adx[i] < -options_.dual_infeasible_tolerance * norm_dx &&
                            !std::isinf(model.l[i])) {
                            ray_compatible = false;
                            break;
                        }
                    }
                    if (ray_compatible) {
                        sol.status = QpStatus::dual_infeasible;
                        break;
                    }
                }
            }
        }

        // 9. Adaptive penalty parameter update
        if (options_.adaptive_rho && iter > 0 &&
            (iter % options_.adaptive_rho_interval == 0)) {
            const double s_prim =
                norm_prim / (std::max(inf_norm(Ax), inf_norm(z)) + 1e-10);
            const double s_dual =
                norm_dual / (std::max({inf_norm(Px), inf_norm(ATy),
                                       inf_norm(model.q)}) +
                             1e-10);
            if (s_dual > 1e-15) {
                double scale = std::sqrt(s_prim / s_dual);
                if (scale > 5.0 || scale < 0.2) {
                    scale = std::clamp(scale, 0.2, 5.0);
                    for (std::size_t i = 0; i < m; ++i) {
                        rho[i] = std::clamp(rho[i] * scale, 1e-6, 1e6);
                    }
                    kkt.update_numeric(model.P, model.A, options_.sigma, rho);
                }
            }
        }
    }

    if (sol.status == QpStatus::numerical_error) {
        sol.status = QpStatus::iteration_limit;
    }

    const auto end_time = std::chrono::steady_clock::now();
    sol.solve_time_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    // Compute objective value: (1/2) x^T P x + q^T x
    sol.x = std::move(x);
    sol.z = std::move(z);
    sol.y = std::move(y);

    const double energy = model.P.evaluate_energy(sol.x);
    double linear = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        linear += model.q[j] * sol.x[j];
    }
    const double internal_obj = 0.5 * energy + linear;

    if (model.sense == model::ObjectiveSense::maximize) {
        sol.objective_value = -internal_obj + model.objective_offset;
    } else {
        sol.objective_value = internal_obj + model.objective_offset;
    }

    return sol;
}

QpSolution solve_qp(const QuadraticModel& model, const QpOptions& options) {
    AdmmQpSolver solver(options);
    return solver.solve(model);
}

} // namespace markov_cero::qp
