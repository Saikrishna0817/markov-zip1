#include "markov_cero/gpu/pdhg_step.hpp"
#include "markov_cero/gpu/kernels.hpp"

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

} // namespace

PdhgState create_pdhg_state(const model::Model& model,
                            const DeviceCsr& A,
                            const DeviceCsr& At,
                            double step_size_reduction) {
    const std::size_t m = model.matrix.row_count;
    const std::size_t n = model.matrix.column_count;

    if (A.rows() != m || A.cols() != n) {
        throw std::invalid_argument("create_pdhg_state: matrix A dimension mismatch");
    }
    if (At.rows() != n || At.cols() != m) {
        throw std::invalid_argument("create_pdhg_state: matrix At dimension mismatch");
    }

    const double obj_sign = (model.objective_sense == model::ObjectiveSense::maximize) ? -1.0 : 1.0;
    std::vector<double> h_c(n);
    for (std::size_t j = 0; j < n; ++j) {
        h_c[j] = obj_sign * model.objective[j];
    }

    std::vector<double> h_var_lo(n);
    std::vector<double> h_var_hi(n);
    std::vector<double> h_x0(n);
    for (std::size_t j = 0; j < n; ++j) {
        h_var_lo[j] = bound_to_double(model.variable_lower[j], false);
        h_var_hi[j] = bound_to_double(model.variable_upper[j], true);
        h_x0[j] = std::clamp(0.0, h_var_lo[j], h_var_hi[j]);
    }

    std::vector<double> h_row_lo(m);
    std::vector<double> h_row_hi(m);
    for (std::size_t i = 0; i < m; ++i) {
        h_row_lo[i] = bound_to_double(model.row_lower[i], false);
        h_row_hi[i] = bound_to_double(model.row_upper[i], true);
    }

    std::vector<double> col_norms(n, 0.0);
    std::vector<double> row_norms(m, 0.0);
    for (std::size_t col = 0; col < n; ++col) {
        const std::size_t start = model.matrix.column_start[col];
        const std::size_t end = model.matrix.column_start[col + 1];
        for (std::size_t ptr = start; ptr < end; ++ptr) {
            const double val = std::abs(model.matrix.value[ptr]);
            col_norms[col] += val;
            row_norms[model.matrix.row_index[ptr]] += val;
        }
    }

    std::vector<double> h_tau(n, 1.0);
    std::vector<double> h_sigma(m, 1.0);
    for (std::size_t j = 0; j < n; ++j) {
        h_tau[j] = (col_norms[j] > 1e-12) ? (step_size_reduction / col_norms[j]) : 1.0;
    }
    for (std::size_t i = 0; i < m; ++i) {
        h_sigma[i] = (row_norms[i] > 1e-12) ? (step_size_reduction / row_norms[i]) : 1.0;
    }

    PdhgState state;
    state.A = &A;
    state.At = &At;
    state.num_variables = n;
    state.num_constraints = m;

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
        double x_new = x_prev - tau[j] * g;
        x_new = std::clamp(x_new, var_lower[j], var_upper[j]);
        x[j] = x_new;
        x_bar[j] = 2.0 * x_new - x_prev;
        if (inv_avg > 0.0) {
            x_avg[j] += (x_new - x_avg[j]) * inv_avg;
        }
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
        if (inv_avg > 0.0) {
            y_avg[i] += (y_new - y_avg[i]) * inv_avg;
        }
    }
}

} // namespace detail

void pdhg_step_cpu(PdhgState& state, std::size_t avg_count) {
    if (state.A == nullptr || state.At == nullptr) {
        throw std::invalid_argument("pdhg_step_cpu: null matrix pointers in state");
    }
    const std::size_t n = state.num_variables;
    const std::size_t m = state.num_constraints;
    if (n == 0 || m == 0) {
        return;
    }

    spmv_transpose_cpu(*state.At, state.y, state.At_y);

    std::vector<double> h_tau(n), h_c(n), h_Aty(n), h_vlo(n), h_vhi(n);
    std::vector<double> h_x(n), h_xbar(n), h_xavg(n);
    state.tau.download(h_tau.data(), n);
    state.c.download(h_c.data(), n);
    state.At_y.download(h_Aty.data(), n);
    state.var_lower.download(h_vlo.data(), n);
    state.var_upper.download(h_vhi.data(), n);
    state.x.download(h_x.data(), n);
    state.x_bar.download(h_xbar.data(), n);
    state.x_avg.download(h_xavg.data(), n);

    detail::pdhg_primal_step_cpu(
        n, h_tau.data(), h_c.data(), h_Aty.data(), h_vlo.data(), h_vhi.data(),
        h_x.data(), h_xbar.data(), h_xavg.data(), avg_count);

    state.x.upload(h_x.data(), n);
    state.x_bar.upload(h_xbar.data(), n);
    state.x_avg.upload(h_xavg.data(), n);

    spmv_cpu(*state.A, state.x_bar, state.Ax_bar);

    std::vector<double> h_sig(m), h_Axbar(m), h_rlo(m), h_rhi(m);
    std::vector<double> h_y(m), h_yavg(m);
    state.sigma.download(h_sig.data(), m);
    state.Ax_bar.download(h_Axbar.data(), m);
    state.row_lower.download(h_rlo.data(), m);
    state.row_upper.download(h_rhi.data(), m);
    state.y.download(h_y.data(), m);
    state.y_avg.download(h_yavg.data(), m);

    detail::pdhg_dual_step_cpu(
        m, h_sig.data(), h_Axbar.data(), h_rlo.data(), h_rhi.data(),
        h_y.data(), h_yavg.data(), avg_count);

    state.y.upload(h_y.data(), m);
    state.y_avg.upload(h_yavg.data(), m);
}

void pdhg_step(PdhgState& state, std::size_t avg_count) {
    if (state.A == nullptr || state.At == nullptr) {
        throw std::invalid_argument("pdhg_step: null matrix pointers in state");
    }
    const std::size_t n = state.num_variables;
    const std::size_t m = state.num_constraints;
    if (n == 0 || m == 0) {
        return;
    }

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

void pdhg_run_iterations(PdhgState& state,
                         std::size_t num_iterations,
                         std::size_t start_avg_count) {
    for (std::size_t k = 0; k < num_iterations; ++k) {
        pdhg_step(state, start_avg_count + k);
    }
}

void pdhg_run_iterations_cpu(PdhgState& state,
                             std::size_t num_iterations,
                             std::size_t start_avg_count) {
    for (std::size_t k = 0; k < num_iterations; ++k) {
        pdhg_step_cpu(state, start_avg_count + k);
    }
}

} // namespace markov_cero::gpu
