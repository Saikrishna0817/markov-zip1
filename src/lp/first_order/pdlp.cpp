// markov-cero: sovereign first-order LP engine
// Primal-Dual Hybrid Gradient (PDHG / PDLP)
// Grounding: Chambolle & Pock (2011); Applegate et al. (2021)
//            "Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient"

#include "markov_cero/lp/first_order/pdlp.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace markov_cero::lp::first_order {

namespace {

// Sparse matrix-vector product: y = A * x  (CSC format, row-wise accumulation)
std::vector<double> spmv(const model::SparseMatrixCSC& A,
                         const std::vector<double>& x) {
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
std::vector<double> spmv_t(const model::SparseMatrixCSC& A,
                            const std::vector<double>& y) {
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
        return PdlpResult{
            PdlpStatus::optimal, {}, {}, 0.0, 0.0, 0.0, 0.0, 0, "trivial"};
    }

    // Objective sign for minimize
    const double obj_sign =
        (model.objective_sense == model::ObjectiveSense::maximize) ? -1.0 : 1.0;

    // Effective cost vector c
    std::vector<double> c(n);
    for (std::size_t j = 0; j < n; ++j) {
        c[j] = obj_sign * model.objective[j];
    }

    // Row bounds as effective b_lower, b_upper for dual feasibility projection
    std::vector<double> b_lo(m), b_hi(m);
    for (std::size_t i = 0; i < m; ++i) {
        b_lo[i] = (model.row_lower[i].kind == model::BoundKind::negative_infinity)
                      ? -1e300 : model.row_lower[i].value;
        b_hi[i] = (model.row_upper[i].kind == model::BoundKind::positive_infinity)
                      ? +1e300 : model.row_upper[i].value;
    }

    // Diagonal preconditioning (Chambolle & Pock 2011; Applegate et al. 2021)
    std::vector<double> row_norms(m, 0.0);
    std::vector<double> col_norms(n, 0.0);
    for (std::size_t col = 0; col < n; ++col) {
        for (std::size_t ptr = model.matrix.column_start[col]; ptr < model.matrix.column_start[col + 1]; ++ptr) {
            const double val = std::abs(model.matrix.value[ptr]);
            col_norms[col] += val;
            row_norms[model.matrix.row_index[ptr]] += val;
        }
    }

    std::vector<double> tau(n, 1.0);
    std::vector<double> sigma(m, 1.0);
    for (std::size_t j = 0; j < n; ++j) {
        tau[j] = (col_norms[j] > 1e-12) ? (options.step_size_reduction / col_norms[j]) : 1.0;
    }
    for (std::size_t i = 0; i < m; ++i) {
        sigma[i] = (row_norms[i] > 1e-12) ? (options.step_size_reduction / row_norms[i]) : 1.0;
    }

    // Initialize primal x and dual y to zero
    std::vector<double> x(n, 0.0);
    std::vector<double> y(m, 0.0);
    std::vector<double> x_bar(n, 0.0);
    std::vector<double> x_avg(n, 0.0);
    std::vector<double> y_avg(m, 0.0);

    // Project initial x onto variable bounds
    for (std::size_t j = 0; j < n; ++j) {
        x[j] = project_bound(0.0, model.variable_lower[j], model.variable_upper[j]);
        x_bar[j] = x[j];
    }

    std::size_t iter = 0;
    double primal_infeas = std::numeric_limits<double>::max();
    double dual_infeas   = std::numeric_limits<double>::max();
    double gap_val       = std::numeric_limits<double>::max();
    std::size_t avg_count = 0;

    // Restart averages
    std::vector<double> x_restart(x);
    std::vector<double> y_restart(y);

    while (iter < options.max_iterations) {
        // --- Primal update: x^{k+1} = proj_[l,u](x^k - tau_j * (c + A^T y^k)) ---
        auto At_y = spmv_t(model.matrix, y);
        std::vector<double> x_prev = x;
        for (std::size_t j = 0; j < n; ++j) {
            double g = c[j] + At_y[j];
            double xnew = x[j] - tau[j] * g;
            x[j] = project_bound(xnew, model.variable_lower[j], model.variable_upper[j]);
        }

        // --- Extrapolation: x_bar^{k+1} = 2 x^{k+1} - x^k ---
        for (std::size_t j = 0; j < n; ++j) {
            x_bar[j] = 2.0 * x[j] - x_prev[j];
        }

        // --- Dual Moreau Proximal update: v = y^k + sigma_i * A * x_bar; y^{k+1} = v - sigma_i * clamp(v/sigma_i, b_lo, b_hi) ---
        auto Ax_bar = spmv(model.matrix, x_bar);
        for (std::size_t i = 0; i < m; ++i) {
            double v = y[i] + sigma[i] * Ax_bar[i];
            double clamped = std::clamp(v / sigma[i], b_lo[i], b_hi[i]);
            y[i] = v - sigma[i] * clamped;
        }

        // --- Ergodic averaging ---
        ++avg_count;
        for (std::size_t j = 0; j < n; ++j) {
            x_avg[j] += (x[j] - x_avg[j]) / static_cast<double>(avg_count);
        }
        for (std::size_t i = 0; i < m; ++i) {
            y_avg[i] += (y[i] - y_avg[i]) / static_cast<double>(avg_count);
        }

        ++iter;

        // --- Convergence check every restart_every iterations ---
        if (iter % options.restart_every == 0) {
            // Primal infeasibility: max_i |A x_avg[i] - proj_i| / (1.0 + |proj_i|)
            auto Ax_avg = spmv(model.matrix, x_avg);
            double max_prim_viol = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                double proj = std::clamp(Ax_avg[i], b_lo[i], b_hi[i]);
                double r = std::abs(Ax_avg[i] - proj);
                double scale = 1.0 + std::abs(proj);
                max_prim_viol = std::max(max_prim_viol, r / scale);
            }
            primal_infeas = max_prim_viol;

            // Dual infeasibility: ||c + A^T y_avg||_projected
            auto At_y_avg = spmv_t(model.matrix, y_avg);
            double dual_res_sq = 0.0;
            double c_scale = 1.0;
            for (std::size_t j = 0; j < n; ++j) {
                double g = c[j] + At_y_avg[j];
                double xl = (model.variable_lower[j].kind == model::BoundKind::negative_infinity)
                                ? -1e300 : model.variable_lower[j].value;
                double xu = (model.variable_upper[j].kind == model::BoundKind::positive_infinity)
                                ? +1e300 : model.variable_upper[j].value;
                double projected_g = g;
                if (x_avg[j] <= xl + 1e-6) {
                    projected_g = std::min(0.0, g);
                } else if (x_avg[j] >= xu - 1e-6) {
                    projected_g = std::max(0.0, g);
                }
                dual_res_sq += projected_g * projected_g;
                c_scale = std::max(c_scale, std::abs(c[j]));
            }
            dual_infeas = std::sqrt(dual_res_sq) / c_scale;

            // Duality gap
            double primal_obj = dot(c, x_avg) + model.objective_offset;
            double dual_obj   = model.objective_offset;
            for (std::size_t i = 0; i < m; ++i) {
                if (y_avg[i] > 0.0 && b_hi[i] < 1e299) {
                    dual_obj -= y_avg[i] * b_hi[i];
                } else if (y_avg[i] < 0.0 && b_lo[i] > -1e299) {
                    dual_obj -= y_avg[i] * b_lo[i];
                }
            }
            for (std::size_t j = 0; j < n; ++j) {
                double g = c[j] + At_y_avg[j];
                double xl = (model.variable_lower[j].kind == model::BoundKind::negative_infinity)
                                ? -1e300 : model.variable_lower[j].value;
                double xu = (model.variable_upper[j].kind == model::BoundKind::positive_infinity)
                                ? +1e300 : model.variable_upper[j].value;
                if (g > 0.0 && xl > -1e299) {
                    dual_obj += g * xl;
                } else if (g < 0.0 && xu < 1e299) {
                    dual_obj += g * xu;
                }
            }
            gap_val = std::abs(primal_obj - dual_obj) /
                      std::max(1.0, std::abs(primal_obj));

            if (primal_infeas <= options.primal_tolerance
                && dual_infeas <= options.dual_tolerance
                && gap_val <= options.gap_tolerance) {
                // Convergence!
                double final_obj = dot(model.objective, x_avg) + model.objective_offset;
                PdlpResult res;
                res.status              = PdlpStatus::optimal;
                res.primal              = x_avg;
                res.dual                = y_avg;
                res.objective           = final_obj;
                res.primal_infeasibility = primal_infeas;
                res.dual_infeasibility  = dual_infeas;
                res.duality_gap         = gap_val;
                res.iterations          = iter;
                res.message             = "PDLP converged";
                return res;
            }

            // Adaptive restart: reset averages if stagnated
            x_restart = x_avg;
            y_restart  = y_avg;
            avg_count  = 0;
        }
    }

    // Return best ergodic iterate even if not converged
    double final_obj = dot(model.objective, x_avg) + model.objective_offset;
    PdlpResult res;
    res.status               = PdlpStatus::iteration_limit;
    res.primal               = x_avg;
    res.dual                 = y_avg;
    res.objective            = final_obj;
    res.primal_infeasibility = primal_infeas;
    res.dual_infeasibility   = dual_infeas;
    res.duality_gap          = gap_val;
    res.iterations           = iter;
    res.message              = "iteration limit reached";
    return res;
}

} // namespace markov_cero::lp::first_order
