#include "markov_cero/gpu/kernels.hpp"

#ifdef MARKOV_CERO_HAS_CUDA
#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace markov_cero::gpu::detail {

namespace {

__global__ void axpy_kernel(std::size_t n, double alpha,
                            const double* __restrict__ x,
                            double* __restrict__ y) {
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        y[i] += alpha * x[i];
    }
}

__global__ void scale_kernel(std::size_t n, double alpha, double* __restrict__ x) {
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        x[i] *= alpha;
    }
}

__global__ void project_bounds_kernel(std::size_t n, double* __restrict__ x,
                                      const double* __restrict__ lower,
                                      const double* __restrict__ upper) {
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        double val = x[i];
        const double lo = lower[i];
        const double hi = upper[i];
        if (val < lo) {
            val = lo;
        }
        if (val > hi) {
            val = hi;
        }
        x[i] = val;
    }
}

inline unsigned int compute_num_blocks(std::size_t n, unsigned int threads_per_block) {
    constexpr unsigned int max_blocks = 1024U;
    const unsigned int needed = static_cast<unsigned int>((n + threads_per_block - 1U) /
                                                          threads_per_block);
    return std::min(needed, max_blocks);
}

} // namespace

void launch_axpy(std::size_t n, double alpha, const double* x, double* y) {
    if (n == 0) {
        return;
    }
    constexpr unsigned int threads = 256U;
    const unsigned int blocks = compute_num_blocks(n, threads);
    axpy_kernel<<<blocks, threads>>>(n, alpha, x, y);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("axpy_kernel launch failed: ") +
                                 cudaGetErrorString(err));
    }
}

void launch_scale(std::size_t n, double alpha, double* x) {
    if (n == 0) {
        return;
    }
    constexpr unsigned int threads = 256U;
    const unsigned int blocks = compute_num_blocks(n, threads);
    scale_kernel<<<blocks, threads>>>(n, alpha, x);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("scale_kernel launch failed: ") +
                                 cudaGetErrorString(err));
    }
}

void launch_project_bounds(std::size_t n, double* x,
                           const double* lower, const double* upper) {
    if (n == 0) {
        return;
    }
    constexpr unsigned int threads = 256U;
    const unsigned int blocks = compute_num_blocks(n, threads);
    project_bounds_kernel<<<blocks, threads>>>(n, x, lower, upper);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("project_bounds_kernel launch failed: ") +
                                 cudaGetErrorString(err));
    }
}

} // namespace markov_cero::gpu::detail

#else

namespace markov_cero::gpu::detail {

void launch_axpy(std::size_t n, double alpha, const double* x, double* y) {
    for (std::size_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

void launch_scale(std::size_t n, double alpha, double* x) {
    for (std::size_t i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
}

void launch_project_bounds(std::size_t n, double* x,
                           const double* lower, const double* upper) {
    for (std::size_t i = 0; i < n; ++i) {
        if (x[i] < lower[i]) {
            x[i] = lower[i];
        }
        if (x[i] > upper[i]) {
            x[i] = upper[i];
        }
    }
}

} // namespace markov_cero::gpu::detail

#endif
