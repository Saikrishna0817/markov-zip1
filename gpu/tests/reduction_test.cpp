#include "markov_cero/gpu/buffer.hpp"
#include "markov_cero/gpu/kernels.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("Assertion failed: " + message);
    }
}

bool bit_identical(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

void test_empty_and_unit() {
    using namespace markov_cero::gpu;

    // 1. Empty vectors
    DeviceBuffer<double> x_empty(0);
    DeviceBuffer<double> y_empty(0);

    require(dot(x_empty, y_empty) == 0.0, "dot empty buffer must be 0.0");
    require(dot_cpu(x_empty, y_empty) == 0.0, "dot_cpu empty buffer must be 0.0");
    require(dot_cpu(0, nullptr, nullptr) == 0.0, "dot_cpu empty ptr must be 0.0");

    require(norm2(x_empty) == 0.0, "norm2 empty buffer must be 0.0");
    require(norm2_cpu(x_empty) == 0.0, "norm2_cpu empty buffer must be 0.0");
    require(norm2_cpu(0, nullptr) == 0.0, "norm2_cpu empty ptr must be 0.0");

    require(inf_norm(x_empty) == 0.0, "inf_norm empty buffer must be 0.0");
    require(inf_norm_cpu(x_empty) == 0.0, "inf_norm_cpu empty buffer must be 0.0");
    require(inf_norm_cpu(0, nullptr) == 0.0, "inf_norm_cpu empty ptr must be 0.0");

    // 2. Unit vectors (n = 1)
    DeviceBuffer<double> x1(std::vector<double>{3.0});
    DeviceBuffer<double> y1(std::vector<double>{-4.0});

    require(std::abs(dot(x1, y1) - (-12.0)) <= 1e-15, "unit dot mismatch");
    require(std::abs(norm2(y1) - 4.0) <= 1e-15, "unit norm2 mismatch");
    require(std::abs(inf_norm(y1) - 4.0) <= 1e-15, "unit inf_norm mismatch");

    // 3. Dimension mismatch checks
    bool caught_dot_mismatch = false;
    try {
        DeviceBuffer<double> x_mismatch(5);
        DeviceBuffer<double> y_mismatch(10);
        dot(x_mismatch, y_mismatch);
    } catch (const std::invalid_argument&) {
        caught_dot_mismatch = true;
    }
    require(caught_dot_mismatch, "dot must throw on size mismatch");

    std::cout << "test_empty_and_unit: PASS\n";
}

void test_reductions_equivalence() {
    using namespace markov_cero::gpu;

    const std::vector<std::size_t> sizes = {
        1, 2, 7, 16, 32, 64, 127, 256, 513, 1024, 8192, 65536, 100000
    };

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    for (std::size_t n : sizes) {
        std::vector<double> h_x(n);
        std::vector<double> h_y(n);
        for (std::size_t i = 0; i < n; ++i) {
            h_x[i] = dist(rng);
            h_y[i] = dist(rng);
        }

        DeviceBuffer<double> d_x(h_x);
        DeviceBuffer<double> d_y(h_y);

        // Dot product test
        double act_dot = dot(d_x, d_y);
        double exp_dot = dot_cpu(d_x, d_y);
        double scale_dot = 1.0 + std::abs(exp_dot);
        double rel_dot_diff = std::abs(act_dot - exp_dot) / scale_dot;
        require(rel_dot_diff <= 1e-13,
                "dot rel diff exceeds 1e-13 for n=" + std::to_string(n));

        // Norm2 test
        double act_norm2 = norm2(d_x);
        double exp_norm2 = norm2_cpu(d_x);
        double scale_norm2 = 1.0 + std::abs(exp_norm2);
        double rel_norm2_diff = std::abs(act_norm2 - exp_norm2) / scale_norm2;
        require(rel_norm2_diff <= 1e-13,
                "norm2 rel diff exceeds 1e-13 for n=" + std::to_string(n));

        // Inf norm test
        double act_inf = inf_norm(d_x);
        double exp_inf = inf_norm_cpu(d_x);
        double abs_inf_diff = std::abs(act_inf - exp_inf);
        require(abs_inf_diff <= 1e-15,
                "inf_norm diff exceeds 1e-15 for n=" + std::to_string(n));
    }

    std::cout << "test_reductions_equivalence: PASS\n";
}

void test_determinism() {
    using namespace markov_cero::gpu;

    const std::vector<std::size_t> sizes = {1000, 10000, 100000};
    constexpr int num_runs = 10;

    std::mt19937_64 rng(98765);
    std::uniform_real_distribution<double> dist(-50.0, 50.0);

    for (std::size_t n : sizes) {
        std::vector<double> h_x(n);
        std::vector<double> h_y(n);
        for (std::size_t i = 0; i < n; ++i) {
            h_x[i] = dist(rng);
            h_y[i] = dist(rng);
        }

        DeviceBuffer<double> d_x(h_x);
        DeviceBuffer<double> d_y(h_y);

        double base_dot = dot(d_x, d_y);
        double base_norm2 = norm2(d_x);
        double base_inf = inf_norm(d_x);

        for (int run = 1; run < num_runs; ++run) {
            double cur_dot = dot(d_x, d_y);
            double cur_norm2 = norm2(d_x);
            double cur_inf = inf_norm(d_x);

            require(bit_identical(cur_dot, base_dot),
                    "dot non-deterministic on run " + std::to_string(run));
            require(bit_identical(cur_norm2, base_norm2),
                    "norm2 non-deterministic on run " + std::to_string(run));
            require(bit_identical(cur_inf, base_inf),
                    "inf_norm non-deterministic on run " + std::to_string(run));
        }
    }

    std::cout << "test_determinism: PASS (" << num_runs << " repeated runs)\n";
}

void test_mathematical_properties() {
    using namespace markov_cero::gpu;

    constexpr std::size_t n = 500;
    std::mt19937_64 rng(1337);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    std::vector<double> h_x(n);
    std::vector<double> h_y(n);
    for (std::size_t i = 0; i < n; ++i) {
        h_x[i] = dist(rng);
        h_y[i] = dist(rng);
    }

    DeviceBuffer<double> d_x(h_x);
    DeviceBuffer<double> d_y(h_y);

    // Cauchy-Schwarz: |dot(x, y)| <= norm2(x) * norm2(y) + eps
    double dot_xy = dot(d_x, d_y);
    double nx = norm2(d_x);
    double ny = norm2(d_y);
    require(std::abs(dot_xy) <= nx * ny + 1e-12, "Cauchy-Schwarz violation");

    // Self-dot equals squared norm2: dot(x, x) == (norm2(x))^2
    double dot_xx = dot(d_x, d_x);
    require(std::abs(dot_xx - (nx * nx)) / (1.0 + dot_xx) <= 1e-12,
            "dot(x, x) != norm2(x)^2");

    // Positive definiteness: norm2(x) > 0 for non-zero vector
    require(nx > 0.0, "norm2(x) must be positive");
    require(inf_norm(d_x) > 0.0, "inf_norm(x) must be positive");

    std::cout << "test_mathematical_properties: PASS\n";
}

} // namespace

int main() {
    std::cout << "=== Markov-Cero Deterministic Reduction Tests (T-5.06) ===\n";
    test_empty_and_unit();
    test_reductions_equivalence();
    test_determinism();
    test_mathematical_properties();
    std::cout << "=== All Reduction Tests Passed Successfully ===\n";
    return 0;
}
