#include "markov_cero/gpu/kernels.hpp"

#ifdef MARKOV_CERO_HAS_CUDA
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace markov_cero::gpu::detail {

namespace {

__device__ inline double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffffU, val, offset);
    }
    return val;
}

__device__ inline double warp_reduce_max(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmax(val, __shfl_down_sync(0xffffffffU, val, offset));
    }
    return val;
}

__device__ inline double block_reduce_sum(double val) {
    __shared__ double warp_sums[8];
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int wid = threadIdx.x >> 5U;

    val = warp_reduce_sum(val);
    if (lane == 0U) {
        warp_sums[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < 8U) ? warp_sums[lane] : 0.0;
    if (wid == 0U) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__device__ inline double block_reduce_max(double val) {
    __shared__ double warp_maxs[8];
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int wid = threadIdx.x >> 5U;

    val = warp_reduce_max(val);
    if (lane == 0U) {
        warp_maxs[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < 8U) ? warp_maxs[lane] : 0.0;
    if (wid == 0U) {
        val = warp_reduce_max(val);
    }
    return val;
}

__global__ void dot_stage1_kernel(std::size_t n,
                                  const double* __restrict__ x,
                                  const double* __restrict__ y,
                                  double* __restrict__ block_sums) {
    double sum = 0.0;
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        sum += x[i] * y[i];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0U) {
        block_sums[blockIdx.x] = sum;
    }
}

__global__ void norm2_sq_stage1_kernel(std::size_t n,
                                       const double* __restrict__ x,
                                       double* __restrict__ block_sums) {
    double sum = 0.0;
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const double val = x[i];
        sum += val * val;
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0U) {
        block_sums[blockIdx.x] = sum;
    }
}

__global__ void inf_norm_stage1_kernel(std::size_t n,
                                       const double* __restrict__ x,
                                       double* __restrict__ block_maxs) {
    double max_val = 0.0;
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        max_val = fmax(max_val, fabs(x[i]));
    }
    max_val = block_reduce_max(max_val);
    if (threadIdx.x == 0U) {
        block_maxs[blockIdx.x] = max_val;
    }
}

__global__ void stage2_sum_kernel(unsigned int num_blocks,
                                  const double* __restrict__ block_sums,
                                  double* __restrict__ out) {
    if (threadIdx.x == 0U) {
        double total = 0.0;
        for (unsigned int i = 0; i < num_blocks; ++i) {
            total += block_sums[i];
        }
        *out = total;
    }
}

__global__ void stage2_norm2_kernel(unsigned int num_blocks,
                                    const double* __restrict__ block_sums,
                                    double* __restrict__ out) {
    if (threadIdx.x == 0U) {
        double total = 0.0;
        for (unsigned int i = 0; i < num_blocks; ++i) {
            total += block_sums[i];
        }
        *out = sqrt(fmax(0.0, total));
    }
}

__global__ void stage2_inf_norm_kernel(unsigned int num_blocks,
                                       const double* __restrict__ block_maxs,
                                       double* __restrict__ out) {
    if (threadIdx.x == 0U) {
        double total = 0.0;
        for (unsigned int i = 0; i < num_blocks; ++i) {
            total = fmax(total, block_maxs[i]);
        }
        *out = total;
    }
}

inline unsigned int compute_num_blocks(std::size_t n, unsigned int threads_per_block) {
    constexpr unsigned int max_blocks = 256U;
    const unsigned int needed = static_cast<unsigned int>((n + threads_per_block - 1U) /
                                                          threads_per_block);
    return std::max(1U, std::min(needed, max_blocks));
}

} // namespace

double launch_dot(std::size_t n, const double* x, const double* y) {
    if (n == 0) {
        return 0.0;
    }
    constexpr unsigned int threads = 256U;
    const unsigned int blocks = compute_num_blocks(n, threads);

    DeviceBuffer<double> workspace(blocks + 1U);
    double* d_blocks = workspace.data();
    double* d_out = workspace.data() + blocks;

    dot_stage1_kernel<<<blocks, threads>>>(n, x, y, d_blocks);
    stage2_sum_kernel<<<1U, 32U>>>(blocks, d_blocks, d_out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("launch_dot kernel launch failed: ") +
                                 cudaGetErrorString(err));
    }

    double result = 0.0;
    err = cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("launch_dot cudaMemcpy failed: ") +
                                 cudaGetErrorString(err));
    }
    return result;
}

double launch_norm2(std::size_t n, const double* x) {
    if (n == 0) {
        return 0.0;
    }
    constexpr unsigned int threads = 256U;
    const unsigned int blocks = compute_num_blocks(n, threads);

    DeviceBuffer<double> workspace(blocks + 1U);
    double* d_blocks = workspace.data();
    double* d_out = workspace.data() + blocks;

    norm2_sq_stage1_kernel<<<blocks, threads>>>(n, x, d_blocks);
    stage2_norm2_kernel<<<1U, 32U>>>(blocks, d_blocks, d_out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("launch_norm2 kernel launch failed: ") +
                                 cudaGetErrorString(err));
    }

    double result = 0.0;
    err = cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("launch_norm2 cudaMemcpy failed: ") +
                                 cudaGetErrorString(err));
    }
    return result;
}

double launch_inf_norm(std::size_t n, const double* x) {
    if (n == 0) {
        return 0.0;
    }
    constexpr unsigned int threads = 256U;
    const unsigned int blocks = compute_num_blocks(n, threads);

    DeviceBuffer<double> workspace(blocks + 1U);
    double* d_blocks = workspace.data();
    double* d_out = workspace.data() + blocks;

    inf_norm_stage1_kernel<<<blocks, threads>>>(n, x, d_blocks);
    stage2_inf_norm_kernel<<<1U, 32U>>>(blocks, d_blocks, d_out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("launch_inf_norm kernel launch failed: ") +
                                 cudaGetErrorString(err));
    }

    double result = 0.0;
    err = cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("launch_inf_norm cudaMemcpy failed: ") +
                                 cudaGetErrorString(err));
    }
    return result;
}

} // namespace markov_cero::gpu::detail
#endif
