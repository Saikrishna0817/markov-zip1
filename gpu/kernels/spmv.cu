#include "markov_cero/gpu/kernels.hpp"

#ifdef MARKOV_CERO_HAS_CUDA
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace markov_cero::gpu::detail {

namespace {

__global__ void spmv_csr_vector_kernel(std::size_t rows,
                                       const std::size_t* __restrict__ row_offsets,
                                       const std::size_t* __restrict__ col_indices,
                                       const double* __restrict__ values,
                                       const double* __restrict__ x,
                                       double* __restrict__ y) {
    const std::size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32U;
    const unsigned int lane_id = threadIdx.x & 31U;

    if (warp_id >= rows) {
        return;
    }

    const std::size_t row_start = row_offsets[warp_id];
    const std::size_t row_end = row_offsets[warp_id + 1];

    double sum = 0.0;
    for (std::size_t p = row_start + lane_id; p < row_end; p += 32U) {
        sum += values[p] * x[col_indices[p]];
    }

    // Warp-level shuffle reduction down to lane 0
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }

    if (lane_id == 0U) {
        y[warp_id] = sum;
    }
}

} // namespace

void launch_spmv_csr_vector(std::size_t rows,
                            const std::size_t* row_offsets,
                            const std::size_t* col_indices,
                            const double* values,
                            const double* x,
                            double* y) {
    if (rows == 0) {
        return;
    }

    constexpr unsigned int threads_per_block = 256U;
    constexpr unsigned int warps_per_block = threads_per_block / 32U;
    const unsigned int num_blocks = static_cast<unsigned int>((rows + warps_per_block - 1U) /
                                                              warps_per_block);

    spmv_csr_vector_kernel<<<num_blocks, threads_per_block>>>(
        rows, row_offsets, col_indices, values, x, y
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("spmv_csr_vector_kernel launch failed: ") +
                                 cudaGetErrorString(err));
    }
}

} // namespace markov_cero::gpu::detail

#else

namespace markov_cero::gpu::detail {

void launch_spmv_csr_vector(std::size_t rows,
                            const std::size_t* row_offsets,
                            const std::size_t* col_indices,
                            const double* values,
                            const double* x,
                            double* y) {
    // Graceful no-CUDA stub: executes host SpMV directly
    for (std::size_t i = 0; i < rows; ++i) {
        const std::size_t start = row_offsets[i];
        const std::size_t end = row_offsets[i + 1];
        double sum = 0.0;
        for (std::size_t p = start; p < end; ++p) {
            sum += values[p] * x[col_indices[p]];
        }
        y[i] = sum;
    }
}

} // namespace markov_cero::gpu::detail

#endif
