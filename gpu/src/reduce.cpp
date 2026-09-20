#include "markov_cero/gpu/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace markov_cero::gpu {

double dot_cpu(std::size_t n, const double* x, const double* y) {
    if (n == 0) {
        return 0.0;
    }
    if (x == nullptr) {
        throw std::invalid_argument("dot_cpu null x pointer");
    }
    if (y == nullptr) {
        throw std::invalid_argument("dot_cpu null y pointer");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

double dot_cpu(const DeviceBuffer<double>& x, const DeviceBuffer<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("dot_cpu size mismatch: x.size() != y.size()");
    }
    if (x.size() == 0) {
        return 0.0;
    }
    std::vector<double> h_x(x.size());
    std::vector<double> h_y(y.size());
    x.download(h_x.data(), x.size());
    y.download(h_y.data(), y.size());
    return dot_cpu(x.size(), h_x.data(), h_y.data());
}

double dot(const DeviceBuffer<double>& x, const DeviceBuffer<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("dot size mismatch: x.size() != y.size()");
    }
    if (x.size() == 0) {
        return 0.0;
    }
#ifdef MARKOV_CERO_HAS_CUDA
    return detail::launch_dot(x.size(), x.data(), y.data());
#else
    return dot_cpu(x, y);
#endif
}

double norm2_cpu(std::size_t n, const double* x) {
    if (n == 0) {
        return 0.0;
    }
    if (x == nullptr) {
        throw std::invalid_argument("norm2_cpu null x pointer");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += x[i] * x[i];
    }
    return std::sqrt(std::max(0.0, sum));
}

double norm2_cpu(const DeviceBuffer<double>& x) {
    if (x.size() == 0) {
        return 0.0;
    }
    std::vector<double> h_x(x.size());
    x.download(h_x.data(), x.size());
    return norm2_cpu(x.size(), h_x.data());
}

double norm2(const DeviceBuffer<double>& x) {
    if (x.size() == 0) {
        return 0.0;
    }
#ifdef MARKOV_CERO_HAS_CUDA
    return detail::launch_norm2(x.size(), x.data());
#else
    return norm2_cpu(x);
#endif
}

double inf_norm_cpu(std::size_t n, const double* x) {
    if (n == 0) {
        return 0.0;
    }
    if (x == nullptr) {
        throw std::invalid_argument("inf_norm_cpu null x pointer");
    }
    double max_val = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_val = std::max(max_val, std::abs(x[i]));
    }
    return max_val;
}

double inf_norm_cpu(const DeviceBuffer<double>& x) {
    if (x.size() == 0) {
        return 0.0;
    }
    std::vector<double> h_x(x.size());
    x.download(h_x.data(), x.size());
    return inf_norm_cpu(x.size(), h_x.data());
}

double inf_norm(const DeviceBuffer<double>& x) {
    if (x.size() == 0) {
        return 0.0;
    }
#ifdef MARKOV_CERO_HAS_CUDA
    return detail::launch_inf_norm(x.size(), x.data());
#else
    return inf_norm_cpu(x);
#endif
}

} // namespace markov_cero::gpu
