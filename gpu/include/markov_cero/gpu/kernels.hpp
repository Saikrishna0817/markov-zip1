#pragma once

#include "markov_cero/gpu/buffer.hpp"
#include "markov_cero/gpu/csr.hpp"

#include <cstddef>

namespace markov_cero::gpu {

// CSR Sparse Matrix-Vector multiplication: y = A * x
void spmv(const DeviceCsr& A, const DeviceBuffer<double>& x, DeviceBuffer<double>& y);
void spmv_cpu(const DeviceCsr& A, const double* x, double* y);
void spmv_cpu(const DeviceCsr& A, const DeviceBuffer<double>& x, DeviceBuffer<double>& y);

// Transpose SpMV: z = A^T * y
void spmv_transpose(const DeviceCsr& At,
                    const DeviceBuffer<double>& y,
                    DeviceBuffer<double>& z);
void spmv_transpose_cpu(const DeviceCsr& At, const double* y, double* z);
void spmv_transpose_cpu(const DeviceCsr& At,
                        const DeviceBuffer<double>& y,
                        DeviceBuffer<double>& z);

// Vector axpy: in-place y <- alpha * x + y
void axpy(double alpha, const DeviceBuffer<double>& x, DeviceBuffer<double>& y);
void axpy_cpu(double alpha, const DeviceBuffer<double>& x, DeviceBuffer<double>& y);
void axpy_cpu(std::size_t n, double alpha, const double* x, double* y);

// Vector scale: in-place x <- alpha * x
void scale(double alpha, DeviceBuffer<double>& x);
void scale_cpu(double alpha, DeviceBuffer<double>& x);
void scale_cpu(std::size_t n, double alpha, double* x);

// Vector elementwise bound projection: x_i <- clamp(x_i, lower_i, upper_i)
void project_bounds(DeviceBuffer<double>& x,
                    const DeviceBuffer<double>& lower,
                    const DeviceBuffer<double>& upper);
void project_bounds_cpu(DeviceBuffer<double>& x,
                        const DeviceBuffer<double>& lower,
                        const DeviceBuffer<double>& upper);
void project_bounds_cpu(std::size_t n, double* x,
                        const double* lower, const double* upper);

// Vector dot product: sum(x_i * y_i)
double dot(const DeviceBuffer<double>& x, const DeviceBuffer<double>& y);
double dot_cpu(const DeviceBuffer<double>& x, const DeviceBuffer<double>& y);
double dot_cpu(std::size_t n, const double* x, const double* y);

// Vector Euclidean 2-norm: sqrt(sum(x_i^2))
double norm2(const DeviceBuffer<double>& x);
double norm2_cpu(const DeviceBuffer<double>& x);
double norm2_cpu(std::size_t n, const double* x);

// Vector infinity norm: max(|x_i|)
double inf_norm(const DeviceBuffer<double>& x);
double inf_norm_cpu(const DeviceBuffer<double>& x);
double inf_norm_cpu(std::size_t n, const double* x);

namespace detail {

void launch_spmv_csr_vector(std::size_t rows,
                            const std::size_t* row_offsets,
                            const std::size_t* col_indices,
                            const double* values,
                            const double* x,
                            double* y);

void launch_axpy(std::size_t n, double alpha, const double* x, double* y);
void launch_scale(std::size_t n, double alpha, double* x);
void launch_project_bounds(std::size_t n, double* x,
                           const double* lower, const double* upper);

double launch_dot(std::size_t n, const double* x, const double* y);
double launch_norm2(std::size_t n, const double* x);
double launch_inf_norm(std::size_t n, const double* x);

} // namespace detail

} // namespace markov_cero::gpu
