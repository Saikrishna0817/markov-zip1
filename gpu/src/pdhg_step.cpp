#include "markov_cero/gpu/pdhg_step.hpp"
#include "markov_cero/gpu/kernels.hpp"
#include "markov_cero/scale/ruiz_scaling.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace markov_cero::gpu {

namespace {

constexpr double kInfinitySentinel = 1e300;

double bound_to_double(const model::Bound& b, bool is_upper) {
    if (is_upper) {
        return (b.kind == model::BoundKind::positive_infinity) ? kInfinitySentinel : b.value;
    }
    return (b.kind == model::BoundKind::negative_infinity) ? -kInfinitySentinel : b.value;
}

inline double bound_val(const model::Bound& b, double def_inf) {
    return (b.kind == model::BoundKind::finite) ? b.value : def_inf;
}

} // namespace

PdhgState create_pdhg_state(const model::Model& model,
                            const DeviceCsr& A,
                            const DeviceCsr& At,
                            double step_size_reduction,
                            double primal_weight) {
    const std::size_t m = model.matrix.row_count;
    const std::size_t n = model.matrix.column_count;

    if (A.rows() != m || A.cols() != n || At.rows() != n || At.cols() != m) {
        throw std::invalid_argument("create_pdhg_state: matrix dimension mismatch");
    }

    const double obj_sign = (model.objective_sense == model::ObjectiveSense::maximize) ? -1.0 : 1.0;
    std::vector<double> h_c(n), h_var_lo(n), h_var_hi(n), h_x0(n);
    for (std::size_t j = 0; j < n; ++j) {
        h_c[j] = obj_sign * model.objective[j];
        h_var_lo[j] = bound_to_double(model.variable_lower[j], false);
        h_var_hi[j] = bound_to_double(model.variable_upper[j], true);
        h_x0[j] = std::clamp(0.0, h_var_lo[j], h_var_hi[j]);
    }

    std::vector<double> h_row_lo(m), h_row_hi(m);
    for (std::size_t i = 0; i < m; ++i) {
        h_row_lo[i] = bound_to_double(model.row_lower[i], false);
        h_row_hi[i] = bound_to_double(model.row_upper[i], true);
    }

    std::vector<double> col_norms(n, 0.0), row_norms(m, 0.0);
    for (std::size_t col = 0; col < n; ++col) {
        const std::size_t start = model.matrix.column_start[col];
        const std::size_t end = model.matrix.column_start[col + 1];
        for (std::size_t ptr = start; ptr < end; ++ptr) {
            const double val = std::abs(model.matrix.value[ptr]);
            col_norms[col] += val;
            row_norms[model.matrix.row_index[ptr]] += val;
        }
    }
    for (std::size_t j = 0; j < n; ++j) if (col_norms[j] < 1e-12) col_norms[j] = 1.0;
    for (std::size_t i = 0; i < m; ++i) if (row_norms[i] < 1e-12) row_norms[i] = 1.0;

    const double eta = std::clamp(step_size_reduction, 0.1, 0.99);
    const double omega = std::clamp(primal_weight, 1e-6, 1e6);

    std::vector<double> h_tau(n, 1.0), h_sigma(m, 1.0);
    for (std::size_t j = 0; j < n; ++j) h_tau[j] = (eta / omega) / col_norms[j];
    for (std::size_t i = 0; i < m; ++i) h_sigma[i] = (eta * omega) / row_norms[i];

    PdhgState state;
    state.A = &A;
    state.At = &At;
    state.num_variables = n;
    state.num_constraints = m;
    state.col_norms = std::move(col_norms);
    state.row_norms = std::move(row_norms);
    state.eta = eta;
    state.omega = omega;

    state.c = DeviceBuffer<double>(h_c);
    state.var_lower = DeviceBuffer<double>(h_var_lo);
    state.var_upper = DeviceBuffer<double>(h_var_hi);
    state.row_lower = DeviceBuffer<double>(h_row_lo);
    state.row_upper = DeviceBuffer<double>(h_row_hi);
    state.tau = DeviceBuffer<double>(h_tau);
    state.sigma = DeviceBuffer<double>(h_sigma);

    state.x = DeviceBuffer<double>(h_x0);
    state.x_bar = DeviceBuffer<double>(h_x0);
    state.x_avg = DeviceBuffer<double>(h_x0);
    state.y = DeviceBuffer<double>(std::vector<double>(m, 0.0));
    state.y_avg = DeviceBuffer<double>(std::vector<double>(m, 0.0));

    state.At_y = DeviceBuffer<double>(n);
    state.Ax_bar = DeviceBuffer<double>(m);
    return state;
}

void pdhg_update_step_sizes(PdhgState& state, double eta, double omega) {
    const std::size_t n = state.num_variables;
    const std::size_t m = state.num_constraints;
    if (n == 0 || m == 0) return;
    state.eta = std::clamp(eta, 0.1, 0.99);
    state.omega = std::clamp(omega, 1e-6, 1e6);

    std::vector<double> h_tau(n), h_sigma(m);
    for (std::size_t j = 0; j < n; ++j) {
        h_tau[j] = (state.eta / state.omega) / state.col_norms[j];
    }
    for (std::size_t i = 0; i < m; ++i) {
        h_sigma[i] = (state.eta * state.omega) / state.row_norms[i];
    }
    state.tau.upload(h_tau.data(), n);
    state.sigma.upload(h_sigma.data(), m);
}

namespace detail {

void pdhg_primal_step_cpu(std::size_t n,
                          const double* tau,
                          const double* c,
                          const double* At_y,
                          const double* var_lower,
                          const double* var_upper,
                          double* x,
                          double* x_bar,
                          double* x_avg,
                          std::size_t avg_count) {
    const double inv_avg = (avg_count > 0) ? (1.0 / static_cast<double>(avg_count)) : 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        const double g = c[j] + At_y[j];
        const double x_prev = x[j];
        double x_new = std::clamp(x_prev - tau[j] * g, var_lower[j], var_upper[j]);
        x[j] = x_new;
        x_bar[j] = 2.0 * x_new - x_prev;
        if (inv_avg > 0.0) x_avg[j] += (x_new - x_avg[j]) * inv_avg;
    }
}

void pdhg_dual_step_cpu(std::size_t m,
                        const double* sigma,
                        const double* Ax_bar,
                        const double* row_lower,
                        const double* row_upper,
                        double* y,
                        double* y_avg,
                        std::size_t avg_count) {
    const double inv_avg = (avg_count > 0) ? (1.0 / static_cast<double>(avg_count)) : 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        const double sig = sigma[i];
        const double v = y[i] + sig * Ax_bar[i];
        const double clamped = std::clamp(v / sig, row_lower[i], row_upper[i]);
        const double y_new = v - sig * clamped;
        y[i] = y_new;
        if (inv_avg > 0.0) y_avg[i] += (y_new - y_avg[i]) * inv_avg;
    }
}

} // namespace detail

void pdhg_step_cpu(PdhgState& state, std::size_t avg_count) {
    if (!state.A || !state.At) throw std::invalid_argument("null matrix in state");
    const std::size_t n = state.num_variables, m = state.num_constraints;
    if (n == 0 || m == 0) return;

    spmv_transpose_cpu(*state.At, state.y, state.At_y);

    std::vector<double> tau(n), c(n), Aty(n), vlo(n), vhi(n), x(n), xbar(n), xavg(n);
    state.tau.download(tau.data(), n);
    state.c.download(c.data(), n);
    state.At_y.download(Aty.data(), n);
    state.var_lower.download(vlo.data(), n);
    state.var_upper.download(vhi.data(), n);
    state.x.download(x.data(), n);
    state.x_bar.download(xbar.data(), n);
    state.x_avg.download(xavg.data(), n);

    detail::pdhg_primal_step_cpu(n, tau.data(), c.data(), Aty.data(), vlo.data(),
                                 vhi.data(), x.data(), xbar.data(), xavg.data(), avg_count);

    state.x.upload(x.data(), n);
    state.x_bar.upload(xbar.data(), n);
    state.x_avg.upload(xavg.data(), n);

    spmv_cpu(*state.A, state.x_bar, state.Ax_bar);

    std::vector<double> sig(m), Axbar(m), rlo(m), rhi(m), y(m), yavg(m);
    state.sigma.download(sig.data(), m);
    state.Ax_bar.download(Axbar.data(), m);
    state.row_lower.download(rlo.data(), m);
    state.row_upper.download(rhi.data(), m);
    state.y.download(y.data(), m);
    state.y_avg.download(yavg.data(), m);

    detail::pdhg_dual_step_cpu(m, sig.data(), Axbar.data(), rlo.data(), rhi.data(),
                               y.data(), yavg.data(), avg_count);

    state.y.upload(y.data(), m);
    state.y_avg.upload(yavg.data(), m);
}

void pdhg_step(PdhgState& state, std::size_t avg_count) {
    if (!state.A || !state.At) throw std::invalid_argument("pdhg_step: null matrix");
    const std::size_t n = state.num_variables, m = state.num_constraints;
    if (n == 0 || m == 0) return;

#ifdef MARKOV_CERO_HAS_CUDA
    spmv_transpose(*state.At, state.y, state.At_y);
    detail::launch_pdhg_primal_step(
        n, state.tau.data(), state.c.data(), state.At_y.data(),
        state.var_lower.data(), state.var_upper.data(),
        state.x.data(), state.x_bar.data(), state.x_avg.data(), avg_count);
    spmv(*state.A, state.x_bar, state.Ax_bar);
    detail::launch_pdhg_dual_step(
        m, state.sigma.data(), state.Ax_bar.data(),
        state.row_lower.data(), state.row_upper.data(),
        state.y.data(), state.y_avg.data(), avg_count);
#else
    pdhg_step_cpu(state, avg_count);
#endif
}

void pdhg_run_iterations(PdhgState& state, std::size_t num_iters, std::size_t start_avg) {
    for (std::size_t k = 0; k < num_iters; ++k) pdhg_step(state, start_avg + k);
}

void pdhg_run_iterations_cpu(PdhgState& state, std::size_t num_iters, std::size_t start_avg) {
    for (std::size_t k = 0; k < num_iters; ++k) pdhg_step_cpu(state, start_avg + k);
}

PdhgResiduals evaluate_residuals(const PdhgState& state,
                                 const model::Model& original_model,
                                 const scale::RuizScalers* scalers) {
    const std::size_t m = state.num_constraints, n = state.num_variables;
    if (m == 0 || n == 0) return PdhgResiduals{0.0, 0.0, 0.0, 0.0};

    std::vector<double> h_xavg(n), h_yavg(m);
    state.x_avg.download(h_xavg.data(), n);
    state.y_avg.download(h_yavg.data(), m);

    std::vector<double> x_orig(n), y_orig(m);
    for (std::size_t j = 0; j < n; ++j) {
        x_orig[j] = scalers ? (h_xavg[j] * scalers->col_scale[j]) : h_xavg[j];
    }
    for (std::size_t i = 0; i < m; ++i) {
        y_orig[i] = scalers ? (h_yavg[i] * scalers->row_scale[i]) : h_yavg[i];
    }

    std::vector<double> Ax_orig(m, 0.0);
    for (std::size_t col = 0; col < n; ++col) {
        const double xc = x_orig[col];
        if (xc != 0.0) {
            for (std::size_t ptr = original_model.matrix.column_start[col];
                 ptr < original_model.matrix.column_start[col + 1]; ++ptr) {
                Ax_orig[original_model.matrix.row_index[ptr]] +=
                    original_model.matrix.value[ptr] * xc;
            }
        }
    }

    double prim_viol = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        const double lo = bound_val(original_model.row_lower[i], -1e300);
        const double hi = bound_val(original_model.row_upper[i], +1e300);
        const double proj = std::clamp(Ax_orig[i], lo, hi);
        const double r = std::abs(Ax_orig[i] - proj);
        const double scale = 1.0 + std::max(std::abs(Ax_orig[i]),
                                            std::abs(proj) < 1e299 ? std::abs(proj) : 0.0);
        prim_viol = std::max(prim_viol, r / scale);
    }
    for (std::size_t j = 0; j < n; ++j) {
        const double lo = bound_val(original_model.variable_lower[j], -1e300);
        const double hi = bound_val(original_model.variable_upper[j], +1e300);
        const double proj = std::clamp(x_orig[j], lo, hi);
        const double r = std::abs(x_orig[j] - proj);
        const double scale = 1.0 + std::max(std::abs(x_orig[j]),
                                            std::abs(proj) < 1e299 ? std::abs(proj) : 0.0);
        prim_viol = std::max(prim_viol, r / scale);
    }

    std::vector<double> At_yorig(n, 0.0);
    for (std::size_t col = 0; col < n; ++col) {
        double s = 0.0;
        for (std::size_t ptr = original_model.matrix.column_start[col];
             ptr < original_model.matrix.column_start[col + 1]; ++ptr) {
            s += original_model.matrix.value[ptr] * y_orig[original_model.matrix.row_index[ptr]];
        }
        At_yorig[col] = s;
    }

    const double obj_sign = (original_model.objective_sense == model::ObjectiveSense::maximize)
                                ? -1.0 : 1.0;
    double dual_res_sq = 0.0, c_scale = 1.0;
    for (std::size_t j = 0; j < n; ++j) {
        const double c_j = obj_sign * original_model.objective[j];
        c_scale = std::max(c_scale, std::abs(c_j));
        const double g = c_j + At_yorig[j];
        const double lo = bound_val(original_model.variable_lower[j], -1e300);
        const double hi = bound_val(original_model.variable_upper[j], +1e300);
        const double x_proj = std::clamp(x_orig[j] - g, lo, hi);
        const double diff = x_orig[j] - x_proj;
        dual_res_sq += diff * diff;
    }
    const double dual_viol = std::sqrt(dual_res_sq) / c_scale;

    double prim_obj = original_model.objective_offset;
    for (std::size_t j = 0; j < n; ++j) prim_obj += original_model.objective[j] * x_orig[j];

    double dual_obj = original_model.objective_offset;
    for (std::size_t i = 0; i < m; ++i) {
        const double lo = bound_val(original_model.row_lower[i], -1e300);
        const double hi = bound_val(original_model.row_upper[i], +1e300);
        if (y_orig[i] > 0.0 && hi < 1e299) dual_obj -= y_orig[i] * hi;
        else if (y_orig[i] < 0.0 && lo > -1e299) dual_obj -= y_orig[i] * lo;
    }
    for (std::size_t j = 0; j < n; ++j) {
        const double c_j = obj_sign * original_model.objective[j];
        const double g = c_j + At_yorig[j];
        const double lo = bound_val(original_model.variable_lower[j], -1e300);
        const double hi = bound_val(original_model.variable_upper[j], +1e300);
        if (g > 0.0 && lo > -1e299) dual_obj += g * lo;
        else if (g < 0.0 && hi < 1e299) dual_obj += g * hi;
    }

    const double gap_viol = std::abs(prim_obj - dual_obj) /
                            (1.0 + std::abs(prim_obj) + std::abs(dual_obj));
    const double score = std::max({prim_viol, dual_viol, gap_viol});
    return PdhgResiduals{prim_viol, dual_viol, gap_viol, score};
}

void pdhg_restart(PdhgState& state) {
    const std::size_t n = state.num_variables, m = state.num_constraints;
    if (n == 0 || m == 0) return;
    std::vector<double> h_x(n), h_y(m);
    state.x_avg.download(h_x.data(), n);
    state.y_avg.download(h_y.data(), m);
    state.x.upload(h_x.data(), n);
    state.x_bar.upload(h_x.data(), n);
    state.y.upload(h_y.data(), m);
}

markov_cero::lp::first_order::PdlpResult solve_pdlp_gpu(
    const model::Model& model,
    const markov_cero::lp::first_order::PdlpOptions& options) {
    using namespace markov_cero::lp::first_order;
    const std::size_t m = model.matrix.row_count, n = model.matrix.column_count;
    if (m == 0 || n == 0) {
        return PdlpResult{PdlpStatus::optimal, {}, {}, 0.0, 0.0, 0.0, 0.0,
                          options.primal_tolerance, 0, "trivial"};
    }

    model::Model scaled_model = model;
    scale::RuizScalers scalers;
    if (options.ruiz_scaling) {
        scalers = scale::equilibrate_model(scaled_model, {options.ruiz_iterations});
    }

    double c_norm_inf = 1.0, b_norm_inf = 1.0;
    for (std::size_t j = 0; j < n; ++j) {
        c_norm_inf = std::max(c_norm_inf, std::abs(scaled_model.objective[j]));
    }
    for (std::size_t i = 0; i < m; ++i) {
        if (scaled_model.row_lower[i].kind == model::BoundKind::finite) {
            b_norm_inf = std::max(b_norm_inf, std::abs(scaled_model.row_lower[i].value));
        }
        if (scaled_model.row_upper[i].kind == model::BoundKind::finite) {
            b_norm_inf = std::max(b_norm_inf, std::abs(scaled_model.row_upper[i].value));
        }
    }
    double omega = (options.initial_primal_weight > 0.0)
                       ? options.initial_primal_weight
                       : std::clamp(std::sqrt(c_norm_inf / b_norm_inf), 0.01, 100.0);

    DeviceCsr A = DeviceCsr::from_csc(scaled_model.matrix);
    DeviceCsr At = DeviceCsr::transpose_from_csc(scaled_model.matrix);
    PdhgState state = create_pdhg_state(
        scaled_model, A, At, options.step_size_reduction, omega);

    std::size_t iter = 0, avg_count = 0, iters_since_restart = 0;
    const scale::RuizScalers* p_scalers = options.ruiz_scaling ? &scalers : nullptr;
    auto initial_resids = evaluate_residuals(state, model, p_scalers);
    double last_score = initial_resids.score;
    PdhgResiduals last_resids = initial_resids;

    const std::size_t check_interval = std::max<std::size_t>(1, options.restart_every);

    auto make_result = [&](PdlpStatus st, const char* msg) {
        std::vector<double> h_x(n), h_y(m);
        state.x_avg.download(h_x.data(), n);
        state.y_avg.download(h_y.data(), m);
        if (options.ruiz_scaling) {
            scale::unscale_model_solution(scalers, h_x, h_y);
        }
        double final_obj = model.objective_offset;
        for (std::size_t j = 0; j < n; ++j) final_obj += model.objective[j] * h_x[j];
        return PdlpResult{st, std::move(h_x), std::move(h_y), final_obj,
                          last_resids.primal_infeasibility,
                          last_resids.dual_infeasibility,
                          last_resids.duality_gap,
                          options.primal_tolerance, iter, msg};
    };

    while (iter < options.max_iterations) {
        const std::size_t chunk = std::min(check_interval, options.max_iterations - iter);
        for (std::size_t k = 0; k < chunk; ++k) pdhg_step(state, ++avg_count);
        iter += chunk;
        iters_since_restart += chunk;

        last_resids = evaluate_residuals(state, model, p_scalers);

        if (last_resids.primal_infeasibility <= options.primal_tolerance &&
            last_resids.dual_infeasibility <= options.dual_tolerance &&
            last_resids.duality_gap <= options.gap_tolerance) {
            return make_result(PdlpStatus::optimal, "GPU PDLP converged");
        }

        bool do_restart = false;
        if (options.restart_strategy == RestartStrategy::fixed) {
            do_restart = true;
        } else if (options.restart_strategy == RestartStrategy::adaptive) {
            if (last_resids.score <= options.restart_reduction_factor * last_score ||
                (iters_since_restart >= 5 * check_interval && last_resids.score < last_score)) {
                do_restart = true;
            }
        }

        if (do_restart) {
            if (options.adaptive_primal_weight &&
                last_resids.primal_infeasibility > 1e-12 &&
                last_resids.dual_infeasibility > 1e-12) {
                double ratio = std::sqrt(last_resids.primal_infeasibility /
                                         last_resids.dual_infeasibility);
                ratio = std::clamp(ratio, 0.05, 20.0);
                state.omega = std::clamp(
                    state.omega * std::pow(ratio, options.primal_weight_smoothing),
                    1e-6, 1e6);
                pdhg_update_step_sizes(state, state.eta, state.omega);
            }
            pdhg_restart(state);
            avg_count = 0;
            iters_since_restart = 0;
            last_score = last_resids.score;
        }
    }

    return make_result(PdlpStatus::iteration_limit, "GPU PDLP iteration limit reached");
}

} // namespace markov_cero::gpu
