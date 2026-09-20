#pragma once

#include "markov_cero/gpu/buffer.hpp"
#include "markov_cero/gpu/csr.hpp"

#include <cstddef>

namespace markov_cero::gpu {

// CSR Sparse Matrix-Vector multiplication: y = A * x
// Dimensions: A is m x n, x is n x 1, y is m x 1.
// Throws std::invalid_argument on dimension mismatch.
void spmv(const DeviceCsr& A, const DeviceBuffer<double>& x, DeviceBuffer<double>& y);

// Host CPU reference implementation of CSR SpMV for equivalence checking and fallback.
void spmv_cpu(const DeviceCsr& A, const double* x, double* y);
void spmv_cpu(const DeviceCsr& A, const DeviceBuffer<double>& x, DeviceBuffer<double>& y);

// Transpose SpMV: z = A^T * y
// At is the CSR representation of A^T (dimension n x m).
// Dimensions: At is n x m, y is m x 1, z is n x 1.
void spmv_transpose(const DeviceCsr& At,
                    const DeviceBuffer<double>& y,
                    DeviceBuffer<double>& z);
void spmv_transpose_cpu(const DeviceCsr& At, const double* y, double* z);
void spmv_transpose_cpu(const DeviceCsr& At,
                        const DeviceBuffer<double>& y,
                        DeviceBuffer<double>& z);

namespace detail {

// Low-level kernel launcher for warp-per-row CUDA CSR SpMV
void launch_spmv_csr_vector(std::size_t rows,
                            const std::size_t* row_offsets,
                            const std::size_t* col_indices,
                            const double* values,
                            const double* x,
                            double* y);

} // namespace detail

} // namespace markov_cero::gpu
