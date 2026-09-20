#include "markov_cero/gpu/kernels.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace markov_cero::gpu {

void axpy_cpu(std::size_t n, double alpha, const double* x, double* y) {
    if (n == 0) {
        return;
    }
    if (x == nullptr) {
        throw std::invalid_argument("axpy_cpu null x pointer");
    }
    if (y == nullptr) {
        throw std::invalid_argument("axpy_cpu null y pointer");
    }
    for (std::size_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

void axpy_cpu(double alpha, const DeviceBuffer<double>& x, DeviceBuffer<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("axpy_cpu size mismatch: x.size() != y.size()");
    }
    if (x.size() == 0) {
        return;
    }
    std::vector<double> h_x(x.size());
    std::vector<double> h_y(y.size());
    x.download(h_x.data(), x.size());
    y.download(h_y.data(), y.size());

    axpy_cpu(x.size(), alpha, h_x.data(), h_y.data());
    y.upload(h_y.data(), y.size());
}

void axpy(double alpha, const DeviceBuffer<double>& x, DeviceBuffer<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("axpy size mismatch: x.size() != y.size()");
    }
    if (x.size() == 0) {
        return;
    }
#ifdef MARKOV_CERO_HAS_CUDA
    detail::launch_axpy(x.size(), alpha, x.data(), y.data());
#else
    axpy_cpu(alpha, x, y);
#endif
}

void scale_cpu(std::size_t n, double alpha, double* x) {
    if (n == 0) {
        return;
    }
    if (x == nullptr) {
        throw std::invalid_argument("scale_cpu null x pointer");
    }
    for (std::size_t i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
}

void scale_cpu(double alpha, DeviceBuffer<double>& x) {
    if (x.size() == 0) {
        return;
    }
    std::vector<double> h_x(x.size());
    x.download(h_x.data(), x.size());
    scale_cpu(x.size(), alpha, h_x.data());
    x.upload(h_x.data(), x.size());
}

void scale(double alpha, DeviceBuffer<double>& x) {
    if (x.size() == 0) {
        return;
    }
#ifdef MARKOV_CERO_HAS_CUDA
    detail::launch_scale(x.size(), alpha, x.data());
#else
    scale_cpu(alpha, x);
#endif
}

void project_bounds_cpu(std::size_t n, double* x,
                        const double* lower, const double* upper) {
    if (n == 0) {
        return;
    }
    if (x == nullptr) {
        throw std::invalid_argument("project_bounds_cpu null x pointer");
    }
    if (lower == nullptr) {
        throw std::invalid_argument("project_bounds_cpu null lower pointer");
    }
    if (upper == nullptr) {
        throw std::invalid_argument("project_bounds_cpu null upper pointer");
    }
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = std::clamp(x[i], lower[i], upper[i]);
    }
}

void project_bounds_cpu(DeviceBuffer<double>& x,
                        const DeviceBuffer<double>& lower,
                        const DeviceBuffer<double>& upper) {
    if (x.size() != lower.size() || x.size() != upper.size()) {
        throw std::invalid_argument("project_bounds_cpu size mismatch");
    }
    if (x.size() == 0) {
        return;
    }
    std::vector<double> h_x(x.size());
    std::vector<double> h_lo(lower.size());
    std::vector<double> h_hi(upper.size());
    x.download(h_x.data(), x.size());
    lower.download(h_lo.data(), lower.size());
    upper.download(h_hi.data(), upper.size());

    project_bounds_cpu(x.size(), h_x.data(), h_lo.data(), h_hi.data());
    x.upload(h_x.data(), x.size());
}

void project_bounds(DeviceBuffer<double>& x,
                    const DeviceBuffer<double>& lower,
                    const DeviceBuffer<double>& upper) {
    if (x.size() != lower.size() || x.size() != upper.size()) {
        throw std::invalid_argument("project_bounds size mismatch");
    }
    if (x.size() == 0) {
        return;
    }
#ifdef MARKOV_CERO_HAS_CUDA
    detail::launch_project_bounds(x.size(), x.data(), lower.data(), upper.data());
#else
    project_bounds_cpu(x, lower, upper);
#endif
}

} // namespace markov_cero::gpu
