#include "markov_cero/gpu/pdhg_step.hpp"

#ifdef MARKOV_CERO_HAS_CUDA
#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace markov_cero::gpu::detail {

namespace {

__global__ void pdhg_primal_step_kernel(std::size_t n,
                                        const double* __restrict__ tau,
                                        const double* __restrict__ c,
                                        const double* __restrict__ At_y,
                                        const double* __restrict__ var_lower,
                                        const double* __restrict__ var_upper,
                                        double* __restrict__ x,
                                        double* __restrict__ x_bar,
                                        double* __restrict__ x_avg,
                                        double inv_avg_count) {
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += stride) {
        const double g = c[j] + At_y[j];
        const double x_prev = x[j];
        double x_new = x_prev - tau[j] * g;
        const double lo = var_lower[j];
        const double hi = var_upper[j];
        if (x_new < lo) {
            x_new = lo;
        }
        if (x_new > hi) {
            x_new = hi;
        }
        x[j] = x_new;
        x_bar[j] = 2.0 * x_new - x_prev;
        if (inv_avg_count > 0.0) {
            x_avg[j] += (x_new - x_avg[j]) * inv_avg_count;
        }
    }
}

__global__ void pdhg_dual_step_kernel(std::size_t m,
                                      const double* __restrict__ sigma,
                                      const double* __restrict__ Ax_bar,
                                      const double* __restrict__ row_lower,
                                      const double* __restrict__ row_upper,
                                      double* __restrict__ y,
                                      double* __restrict__ y_avg,
                                      double inv_avg_count) {
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += stride) {
        const double sig = sigma[i];
        const double v = y[i] + sig * Ax_bar[i];
        double scaled_v = v / sig;
        const double lo = row_lower[i];
        const double hi = row_upper[i];
        if (scaled_v < lo) {
            scaled_v = lo;
        }
        if (scaled_v > hi) {
            scaled_v = hi;
        }
        const double y_new = v - sig * scaled_v;
        y[i] = y_new;
        if (inv_avg_count > 0.0) {
            y_avg[i] += (y_new - y_avg[i]) * inv_avg_count;
        }
    }
}

inline unsigned int compute_num_blocks(std::size_t count, unsigned int threads_per_block) {
    constexpr unsigned int max_blocks = 1024U;
    const unsigned int needed = static_cast<unsigned int>((count + threads_per_block - 1U) /
                                                          threads_per_block);
    return std::min(needed, max_blocks);
}

} // namespace

void launch_pdhg_primal_step(std::size_t n,
                             const double* tau,
                             const double* c,
                             const double* At_y,
                             const double* var_lower,
                             const double* var_upper,
                             double* x,
                             double* x_bar,
                             double* x_avg,
                             std::size_t avg_count) {
    if (n == 0) {
        return;
    }
    constexpr unsigned int threads = 256U;
    const unsigned int blocks = compute_num_blocks(n, threads);
    const double inv_avg = (avg_count > 0) ? (1.0 / static_cast<double>(avg_count)) : 0.0;

    pdhg_primal_step_kernel<<<blocks, threads>>>(
        n, tau, c, At_y, var_lower, var_upper, x, x_bar, x_avg, inv_avg);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("launch_pdhg_primal_step failed: ") +
                                 cudaGetErrorString(err));
    }
}

void launch_pdhg_dual_step(std::size_t m,
                           const double* sigma,
                           const double* Ax_bar,
                           const double* row_lower,
                           const double* row_upper,
                           double* y,
                           double* y_avg,
                           std::size_t avg_count) {
    if (m == 0) {
        return;
    }
    constexpr unsigned int threads = 256U;
    const unsigned int blocks = compute_num_blocks(m, threads);
    const double inv_avg = (avg_count > 0) ? (1.0 / static_cast<double>(avg_count)) : 0.0;

    pdhg_dual_step_kernel<<<blocks, threads>>>(
        m, sigma, Ax_bar, row_lower, row_upper, y, y_avg, inv_avg);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("launch_pdhg_dual_step failed: ") +
                                 cudaGetErrorString(err));
    }
}

} // namespace markov_cero::gpu::detail
#endif
