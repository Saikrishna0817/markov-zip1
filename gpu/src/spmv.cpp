#include "markov_cero/gpu/kernels.hpp"

#include <stdexcept>
#include <vector>

namespace markov_cero::gpu {

void spmv_cpu(const DeviceCsr& A, const double* x, double* y) {
    if (x == nullptr) {
        throw std::invalid_argument("spmv_cpu null x pointer");
    }
    if (y == nullptr) {
        throw std::invalid_argument("spmv_cpu null y pointer");
    }
    if (A.rows() == 0 || A.cols() == 0) {
        return;
    }

    std::vector<std::size_t> row_offsets, col_indices;
    std::vector<double> values;
    A.download_to(row_offsets, col_indices, values);

    for (std::size_t i = 0; i < A.rows(); ++i) {
        const std::size_t start = row_offsets[i];
        const std::size_t end = row_offsets[i + 1];
        double sum = 0.0;
        for (std::size_t p = start; p < end; ++p) {
            sum += values[p] * x[col_indices[p]];
        }
        y[i] = sum;
    }
}

void spmv_cpu(const DeviceCsr& A, const DeviceBuffer<double>& x, DeviceBuffer<double>& y) {
    if (x.size() != A.cols()) {
        throw std::invalid_argument("spmv_cpu dimension mismatch: x.size() != A.cols()");
    }
    if (y.size() != A.rows()) {
        throw std::invalid_argument("spmv_cpu dimension mismatch: y.size() != A.rows()");
    }
    if (A.rows() == 0 || A.cols() == 0) {
        return;
    }

    std::vector<double> h_x(A.cols());
    x.download(h_x.data(), A.cols());

    std::vector<double> h_y(A.rows(), 0.0);
    spmv_cpu(A, h_x.data(), h_y.data());

    y.upload(h_y.data(), A.rows());
}

void spmv(const DeviceCsr& A, const DeviceBuffer<double>& x, DeviceBuffer<double>& y) {
    if (x.size() != A.cols()) {
        throw std::invalid_argument("spmv dimension mismatch: x.size() != A.cols()");
    }
    if (y.size() != A.rows()) {
        throw std::invalid_argument("spmv dimension mismatch: y.size() != A.rows()");
    }
    if (A.rows() == 0 || A.cols() == 0) {
        return;
    }

#ifdef MARKOV_CERO_HAS_CUDA
    detail::launch_spmv_csr_vector(
        A.rows(),
        A.row_offsets().data(),
        A.col_indices().data(),
        A.values().data(),
        x.data(),
        y.data()
    );
#else
    spmv_cpu(A, x, y);
#endif
}

} // namespace markov_cero::gpu
