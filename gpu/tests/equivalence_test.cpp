#include "markov_cero/gpu/buffer.hpp"
#include "markov_cero/gpu/csr.hpp"
#include "markov_cero/gpu/kernels.hpp"
#include "markov_cero/io/mps.hpp"
#include "markov_cero/model/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
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

void test_synthetic_spmv() {
    using namespace markov_cero;
    using namespace markov_cero::gpu;

    // 1. Empty matrix
    DeviceCsr empty_csr(0, 0, 0);
    DeviceBuffer<double> x_empty(0);
    DeviceBuffer<double> y_empty(0);
    spmv(empty_csr, x_empty, y_empty);

    // 2. 1x1 matrix: [2.5]
    model::SparseMatrixBuilder b1(1, 1);
    b1.add(0, 0, 2.5);
    DeviceCsr csr1 = DeviceCsr::from_csc(b1.build());
    DeviceBuffer<double> x1(std::vector<double>{4.0});
    DeviceBuffer<double> y1(1);
    spmv(csr1, x1, y1);
    std::vector<double> h_y1 = y1.to_vector();
    require(std::abs(h_y1[0] - 10.0) <= 1e-15, "1x1 spmv result mismatch");

    // 3. 3x3 diagonal matrix: diag(1.0, -2.0, 3.0)
    model::SparseMatrixBuilder b3(3, 3);
    b3.add(0, 0, 1.0);
    b3.add(1, 1, -2.0);
    b3.add(2, 2, 3.0);
    DeviceCsr csr3 = DeviceCsr::from_csc(b3.build());
    DeviceBuffer<double> x3(std::vector<double>{2.0, 3.0, 4.0});
    DeviceBuffer<double> y3(3);
    spmv(csr3, x3, y3);
    std::vector<double> h_y3 = y3.to_vector();
    require(std::abs(h_y3[0] - 2.0) <= 1e-15, "3x3 diagonal [0]");
    require(std::abs(h_y3[1] - (-6.0)) <= 1e-15, "3x3 diagonal [1]");
    require(std::abs(h_y3[2] - 12.0) <= 1e-15, "3x3 diagonal [2]");

    // 4. Dimension mismatch checks
    bool caught_col_mismatch = false;
    try {
        DeviceBuffer<double> x_wrong(4);
        spmv(csr3, x_wrong, y3);
    } catch (const std::invalid_argument&) {
        caught_col_mismatch = true;
    }
    require(caught_col_mismatch, "failed to catch column mismatch");

    bool caught_row_mismatch = false;
    try {
        DeviceBuffer<double> y_wrong(2);
        spmv(csr3, x3, y_wrong);
    } catch (const std::invalid_argument&) {
        caught_row_mismatch = true;
    }
    require(caught_row_mismatch, "failed to catch row mismatch");

    std::cout << "test_synthetic_spmv: PASS\n";
}

double compute_max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    require(a.size() == b.size(), "vector sizes must match for abs diff computation");
    double max_diff = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

double compute_max_rel_diff(const std::vector<double>& a, const std::vector<double>& b) {
    require(a.size() == b.size(), "vector sizes must match for rel diff computation");
    double max_rel = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double abs_diff = std::abs(a[i] - b[i]);
        double scale = 1.0 + std::max(std::abs(a[i]), std::abs(b[i]));
        max_rel = std::max(max_rel, abs_diff / scale);
    }
    return max_rel;
}

void test_netlib_instance(const std::string& instance_name,
                          const std::string& filepath,
                          std::size_t seed) {
    using namespace markov_cero;
    using namespace markov_cero::gpu;

    std::ifstream file(filepath);
    model::Model mdl;
    if (file.is_open()) {
        mdl = io::parse_mps(file);
    } else {
        std::ifstream alt_file("../" + filepath);
        if (!alt_file.is_open()) {
            throw std::runtime_error("Cannot open Netlib file: " + filepath);
        }
        mdl = io::parse_mps(alt_file);
    }

    const std::size_t m = mdl.matrix.row_count;
    const std::size_t n = mdl.matrix.column_count;
    const std::size_t nnz = mdl.matrix.value.size();

    DeviceCsr A = DeviceCsr::from_csc(mdl.matrix);
    require(A.rows() == m, "A.rows mismatch");
    require(A.cols() == n, "A.cols mismatch");
    require(A.nnz() == nnz, "A.nnz mismatch");

    // Test 1: All-ones vector
    std::vector<double> h_x_ones(n, 1.0);
    std::vector<double> expected_ones = mdl.matrix.multiply(h_x_ones);
    DeviceBuffer<double> d_x_ones(h_x_ones);
    DeviceBuffer<double> d_y_ones(m);
    DeviceBuffer<double> d_y_cpu_ones(m);
    spmv(A, d_x_ones, d_y_ones);
    spmv_cpu(A, d_x_ones, d_y_cpu_ones);
    std::vector<double> actual_ones = d_y_ones.to_vector();
    std::vector<double> actual_cpu_ones = d_y_cpu_ones.to_vector();

    double diff_cpu_ones = compute_max_abs_diff(actual_ones, actual_cpu_ones);
    double diff_csc_ones = compute_max_rel_diff(expected_ones, actual_ones);
    require(diff_cpu_ones <= 1e-12, "kernel vs CPU ones diff exceeds 1e-12");
    require(diff_csc_ones <= 1e-12, "CSC vs CSR ones relative diff exceeds 1e-12");

    // Test 2: Alternating +/- 1 vector
    std::vector<double> h_x_alt(n);
    for (std::size_t j = 0; j < n; ++j) {
        h_x_alt[j] = (j % 2 == 0) ? 1.0 : -1.0;
    }
    std::vector<double> expected_alt = mdl.matrix.multiply(h_x_alt);
    DeviceBuffer<double> d_x_alt(h_x_alt);
    DeviceBuffer<double> d_y_alt(m);
    DeviceBuffer<double> d_y_cpu_alt(m);
    spmv(A, d_x_alt, d_y_alt);
    spmv_cpu(A, d_x_alt, d_y_cpu_alt);
    std::vector<double> actual_alt = d_y_alt.to_vector();
    std::vector<double> actual_cpu_alt = d_y_cpu_alt.to_vector();

    double diff_cpu_alt = compute_max_abs_diff(actual_alt, actual_cpu_alt);
    double diff_csc_alt = compute_max_rel_diff(expected_alt, actual_alt);
    require(diff_cpu_alt <= 1e-12, "kernel vs CPU alt diff exceeds 1e-12");
    require(diff_csc_alt <= 1e-12, "CSC vs CSR alt relative diff exceeds 1e-12");

    // Test 3: Random continuous vector in [-1.0, 1.0]
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> h_x_rand(n);
    for (std::size_t j = 0; j < n; ++j) {
        h_x_rand[j] = dist(rng);
    }
    std::vector<double> expected_rand = mdl.matrix.multiply(h_x_rand);
    DeviceBuffer<double> d_x_rand(h_x_rand);
    DeviceBuffer<double> d_y_rand(m);
    DeviceBuffer<double> d_y_cpu_rand(m);
    spmv(A, d_x_rand, d_y_rand);
    spmv_cpu(A, d_x_rand, d_y_cpu_rand);
    std::vector<double> actual_rand = d_y_rand.to_vector();
    std::vector<double> actual_cpu_rand = d_y_cpu_rand.to_vector();

    double diff_cpu_rand = compute_max_abs_diff(actual_rand, actual_cpu_rand);
    double diff_csc_rand = compute_max_rel_diff(expected_rand, actual_rand);
    require(diff_cpu_rand <= 1e-12, "kernel vs CPU rand diff exceeds 1e-12");
    require(diff_csc_rand <= 1e-12, "CSC vs CSR rand relative diff exceeds 1e-12");

    double max_cpu_diff = std::max({diff_cpu_ones, diff_cpu_alt, diff_cpu_rand});

    std::cout << "  " << std::left << std::setw(12) << instance_name
              << " (" << std::right << std::setw(5) << m << " x "
              << std::setw(5) << n << ", nnz="
              << std::setw(6) << nnz << ") max_abs_diff_vs_cpu="
              << std::scientific << std::setprecision(3) << max_cpu_diff
              << " [PASS]\n";
}

void test_all_netlib_matrices() {
    std::cout << "--- Testing SpMV Equivalence on Netlib Matrix Corpus ---\n";
    const std::vector<std::string> netlib_instances = {
        "afiro", "adlittle", "beaconfd", "blend", "kb2", "lotfi",
        "recipe", "sc105", "sc205", "sc50a", "sc50b", "scagr7",
        "scorpion", "scsd1", "scsd6", "share1b", "share2b"
    };

    std::size_t index = 0;
    for (const auto& name : netlib_instances) {
        const std::string path = "data/netlib/" + name + ".mps";
        test_netlib_instance(name, path, 1000U + index * 37U);
        ++index;
    }
    std::cout << "All " << netlib_instances.size()
              << " Netlib matrices passed equivalence test (max diff <= 1e-12).\n";
}

} // namespace

int main() {
    std::cout << "=== Markov-Cero SpMV Equivalence Test (T-5.03) ===\n";
    test_synthetic_spmv();
    test_all_netlib_matrices();
    std::cout << "=== All SpMV Equivalence Tests Passed Successfully ===\n";
    return 0;
}
