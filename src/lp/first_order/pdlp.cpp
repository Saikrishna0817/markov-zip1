// markov-cero: sovereign first-order LP engine
// Primal-Dual Hybrid Gradient (PDHG / PDLP)
// Grounding: Chambolle & Pock (2011); Applegate et al. (2021)
//            "Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient"

#include "markov_cero/lp/first_order/pdlp.hpp"
#include "markov_cero/gpu/pdhg_step.hpp"
#include "markov_cero/scale/ruiz_scaling.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace markov_cero::lp::first_order {

namespace {

// Sparse matrix-vector product: y = A * x  (CSC format, row-wise accumulation)
std::vector<double> spmv(const model::SparseMatrixCSC& A, const std::vector<double>& x) {
    std::vector<double> y(A.row_count, 0.0);
    for (std::size_t col = 0; col < A.column_count; ++col) {
        const double xc = x[col];
        if (xc == 0.0) {
            continue;
        }
        for (std::size_t ptr = A.column_start[col]; ptr < A.column_start[col + 1]; ++ptr) {
            y[A.row_index[ptr]] += A.value[ptr] * xc;
        }
    }
    return y;
}

// Sparse transpose matrix-vector product: z = A^T * y
std::vector<double> spmv_t(const model::SparseMatrixCSC& A, const std::vector<double>& y) {
    std::vector<double> z(A.column_count, 0.0);
    for (std::size_t col = 0; col < A.column_count; ++col) {
        double s = 0.0;
        for (std::size_t ptr = A.column_start[col]; ptr < A.column_start[col + 1]; ++ptr) {
            s += A.value[ptr] * y[A.row_index[ptr]];
        }
        z[col] = s;
    }
    return z;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

// Project x_j onto [l_j, u_j] respecting Bound types
inline double project_bound(double x, const model::Bound& lo, const model::Bound& hi) {
    double lo_val = (lo.kind == model::BoundKind::negative_infinity) ? -1e300 : lo.value;
    double hi_val = (hi.kind == model::BoundKind::positive_infinity) ? +1e300 : hi.value;
    return std::clamp(x, lo_val, hi_val);
}

// Build effective RHS midpoint b from row_lower / row_upper
// For equality rows (l == u), b = l.
// For one-sided, project dual onto feasible range.
// PDLP treats all rows as: A x in [row_lower, row_upper].
// Dual variable y_i >= 0 for upper-bounded rows, <= 0 for lower-bounded.
// Primal update projects x onto [l_j, u_j].
// Dual update projects y_i onto appropriate sign.

} // namespace

PdlpResult solve_pdlp(const model::Model& model, const PdlpOptions& options) {
    const std::size_t m = model.matrix.row_count;
    const std::size_t n = model.matrix.column_count;

    if (n == 0 || m == 0) {
        return PdlpResult{PdlpStatus::optimal, {}, {}, 0.0, 0.0, 0.0, 0.0, 0, "trivial"};
    }

    if (options.backend == Backend::gpu) {
        return gpu::solve_pdlp_gpu(model, options);
    }

    model::Model mdl = model;
    scale::RuizScalers scalers;
    if (options.ruiz_scaling) {
        scalers = scale::equilibrate_model(mdl, {options.ruiz_iterations});
    }

    const double obj_sign = (mdl.objective_sense == model::ObjectiveSense::maximize) ? -1.0 : 1.0;

    std::vector<double> c(n);
    double c_norm_inf = 1.0;
    for (std::size_t j = 0; j < n; ++j) {
        c[j] = obj_sign * mdl.objective[j];
        c_norm_inf = std::max(c_norm_inf, std::abs(c[j]));
    }

    std::vector<double> b_lo(m), b_hi(m);
    double b_norm_inf = 1.0;
    for (std::size_t i = 0; i < m; ++i) {
        b_lo[i] = (mdl.row_lower[i].kind == model::BoundKind::negative_infinity)
                      ? -1e300
                      : mdl.row_lower[i].value;
        b_hi[i] = (mdl.row_upper[i].kind == model::BoundKind::positive_infinity)
                      ? +1e300
                      : mdl.row_upper[i].value;
        if (std::abs(b_lo[i]) < 1e299) b_norm_inf = std::max(b_norm_inf, std::abs(b_lo[i]));
        if (std::abs(b_hi[i]) < 1e299) b_norm_inf = std::max(b_norm_inf, std::abs(b_hi[i]));
    }

    std::vector<double> row_norms(m, 0.0);
    std::vector<double> col_norms(n, 0.0);
    for (std::size_t col = 0; col < n; ++col) {
        for (std::size_t ptr = mdl.matrix.column_start[col];
             ptr < mdl.matrix.column_start[col + 1]; ++ptr) {
            const double val = std::abs(mdl.matrix.value[ptr]);
            col_norms[col] += val;
            row_norms[mdl.matrix.row_index[ptr]] += val;
        }
    }
    for (std::size_t j = 0; j < n; ++j) {
        if (col_norms[j] < 1e-12) col_norms[j] = 1.0;
    }
    for (std::size_t i = 0; i < m; ++i) {
        if (row_norms[i] < 1e-12) row_norms[i] = 1.0;
    }

    double omega = (options.initial_primal_weight > 0.0)
                       ? options.initial_primal_weight
                       : std::clamp(std::sqrt(c_norm_inf / b_norm_inf), 0.01, 100.0);
    double eta = std::clamp(options.step_size_reduction, 0.1, 0.99);

    std::vector<double> tau(n);
    std::vector<double> sigma(m);
    auto update_step_sizes = [&]() {
        for (std::size_t j = 0; j < n; ++j) {
            tau[j] = (eta / omega) / col_norms[j];
        }
        for (std::size_t i = 0; i < m; ++i) {
            sigma[i] = (eta * omega) / row_norms[i];
        }
    };
    update_step_sizes();

    std::vector<double> x(n, 0.0);
    std::vector<double> y(m, 0.0);
    std::vector<double> x_bar(n, 0.0);
    std::vector<double> x_avg(n, 0.0);
    std::vector<double> y_avg(m, 0.0);
    std::vector<double> delta_x(n, 0.0);

    for (std::size_t j = 0; j < n; ++j) {
        x[j] = project_bound(0.0, mdl.variable_lower[j], mdl.variable_upper[j]);
        x_bar[j] = x[j];
        x_avg[j] = x[j];
    }

    std::size_t iter = 0;
    double primal_infeas = std::numeric_limits<double>::max();
    double dual_infeas = std::numeric_limits<double>::max();
    double gap_val = std::numeric_limits<double>::max();
    std::size_t avg_count = 0;
    std::size_t iters_since_restart = 0;
    double last_restart_score = 1e300;

    while (iter < options.max_iterations) {
        auto At_y = spmv_t(mdl.matrix, y);
        std::vector<double> x_prev = x;
        double dx_norm_sq = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            double g = c[j] + At_y[j];
            double xnew = x[j] - tau[j] * g;
            double proj_x = project_bound(xnew, mdl.variable_lower[j], mdl.variable_upper[j]);
            const double dx = proj_x - x[j];
            delta_x[j] = dx;
            dx_norm_sq += col_norms[j] * dx * dx;
            x[j] = proj_x;
        }

        for (std::size_t j = 0; j < n; ++j) {
            x_bar[j] = 2.0 * x[j] - x_prev[j];
        }

        if (options.adaptive_step_size && iter % 10 == 0 && dx_norm_sq > 1e-14) {
            auto Adx = spmv(mdl.matrix, delta_x);
            double Adx_norm_sq = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                Adx_norm_sq += (Adx[i] * Adx[i]) / row_norms[i];
            }
            if (Adx_norm_sq > 1e-14) {
                double L_local = std::sqrt(Adx_norm_sq / dx_norm_sq);
                if (L_local > 1e-6) {
                    double target_eta = 0.95 / L_local;
                    if (target_eta < eta) {
                        eta = std::max(0.1, std::max(target_eta, eta * 0.8));
                        update_step_sizes();
                    } else if (target_eta > 1.05 * eta && eta < 0.99) {
                        eta = std::min(0.99, eta * 1.05);
                        update_step_sizes();
                    }
                }
            }
        }

        auto Ax_bar = spmv(mdl.matrix, x_bar);
        for (std::size_t i = 0; i < m; ++i) {
            double v = y[i] + sigma[i] * Ax_bar[i];
            double clamped = std::clamp(v / sigma[i], b_lo[i], b_hi[i]);
            y[i] = v - sigma[i] * clamped;
        }

        ++avg_count;
        for (std::size_t j = 0; j < n; ++j) {
            x_avg[j] += (x[j] - x_avg[j]) / static_cast<double>(avg_count);
        }
        for (std::size_t i = 0; i < m; ++i) {
            y_avg[i] += (y[i] - y_avg[i]) / static_cast<double>(avg_count);
        }

        ++iter;
        ++iters_since_restart;

        if (iter % options.restart_every == 0) {
            auto Ax_avg = spmv(mdl.matrix, x_avg);
            double max_prim_viol = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                double proj = std::clamp(Ax_avg[i], b_lo[i], b_hi[i]);
                double r = std::abs(Ax_avg[i] - proj);
                double scale = 1.0 + std::abs(proj);
                max_prim_viol = std::max(max_prim_viol, r / scale);
            }
            primal_infeas = max_prim_viol;

            auto At_y_avg = spmv_t(mdl.matrix, y_avg);
            double dual_res_sq = 0.0;
            double c_scale = 1.0;
            for (std::size_t j = 0; j < n; ++j) {
                double g = c[j] + At_y_avg[j];
                double x_proj = project_bound(
                    x_avg[j] - g, mdl.variable_lower[j], mdl.variable_upper[j]);
                double diff = x_avg[j] - x_proj;
                dual_res_sq += diff * diff;
                c_scale = std::max(c_scale, std::abs(c[j]));
            }
            dual_infeas = std::sqrt(dual_res_sq) / c_scale;

            double primal_obj = dot(c, x_avg) + mdl.objective_offset;
            double dual_obj = mdl.objective_offset;
            for (std::size_t i = 0; i < m; ++i) {
                if (y_avg[i] > 0.0 && b_hi[i] < 1e299) {
                    dual_obj -= y_avg[i] * b_hi[i];
                } else if (y_avg[i] < 0.0 && b_lo[i] > -1e299) {
                    dual_obj -= y_avg[i] * b_lo[i];
                }
            }
            for (std::size_t j = 0; j < n; ++j) {
                double g = c[j] + At_y_avg[j];
                double xl = (mdl.variable_lower[j].kind == model::BoundKind::negative_infinity)
                                ? -1e300
                                : mdl.variable_lower[j].value;
                double xu = (mdl.variable_upper[j].kind == model::BoundKind::positive_infinity)
                                ? +1e300
                                : mdl.variable_upper[j].value;
                if (g > 0.0 && xl > -1e299) {
                    dual_obj += g * xl;
                } else if (g < 0.0 && xu < 1e299) {
                    dual_obj += g * xu;
                }
            }
            gap_val = std::abs(primal_obj - dual_obj) / std::max(1.0, std::abs(primal_obj));

            if (primal_infeas <= options.primal_tolerance &&
                dual_infeas <= options.dual_tolerance && gap_val <= options.gap_tolerance) {
                PdlpResult res;
                res.status = PdlpStatus::optimal;
                res.primal = x_avg;
                res.dual = y_avg;
                if (options.ruiz_scaling) {
                    scale::unscale_model_solution(scalers, res.primal, res.dual);
                }
                res.objective = dot(model.objective, res.primal) + model.objective_offset;
                res.primal_infeasibility = primal_infeas;
                res.dual_infeasibility = dual_infeas;
                res.duality_gap = gap_val;
                res.iterations = iter;
                res.message = "PDLP converged";
                return res;
            }

            const double current_score = std::max({primal_infeas, dual_infeas, gap_val});
            bool do_restart = false;
            if (options.restart_strategy == RestartStrategy::fixed) {
                do_restart = true;
            } else if (options.restart_strategy == RestartStrategy::adaptive) {
                if (current_score <= options.restart_reduction_factor * last_restart_score ||
                    (iters_since_restart >= 5 * options.restart_every &&
                     current_score < last_restart_score)) {
                    do_restart = true;
                }
            }

            if (do_restart) {
                if (options.adaptive_primal_weight && primal_infeas > 1e-12 &&
                    dual_infeas > 1e-12) {
                    double ratio = std::sqrt(primal_infeas / dual_infeas);
                    ratio = std::clamp(ratio, 0.05, 20.0);
                    omega = std::clamp(
                        omega * std::pow(ratio, options.primal_weight_smoothing), 1e-6, 1e6);
                    update_step_sizes();
                }
                x = x_avg;
                y = y_avg;
                for (std::size_t j = 0; j < n; ++j) {
                    x_bar[j] = x[j];
                }
                avg_count = 0;
                iters_since_restart = 0;
                last_restart_score = current_score;
            }
        }
    }

    PdlpResult res;
    res.status = PdlpStatus::iteration_limit;
    res.primal = x_avg;
    res.dual = y_avg;
    if (options.ruiz_scaling) {
        scale::unscale_model_solution(scalers, res.primal, res.dual);
    }
    res.objective = dot(model.objective, res.primal) + model.objective_offset;
    res.primal_infeasibility = primal_infeas;
    res.dual_infeasibility = dual_infeas;
    res.duality_gap = gap_val;
    res.iterations = iter;
    res.message = "iteration limit reached";
    return res;
}

} // namespace markov_cero::lp::first_order
