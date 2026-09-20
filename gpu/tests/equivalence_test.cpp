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

void test_synthetic_spmv_transpose() {
    using namespace markov_cero;
    using namespace markov_cero::gpu;

    // Construct 2x3 matrix:
    // [ 1.0  0.0  3.0 ]
    // [ 0.0  2.0  4.0 ]
    model::SparseMatrixBuilder builder(2, 3);
    builder.add(0, 0, 1.0);
    builder.add(0, 2, 3.0);
    builder.add(1, 1, 2.0);
    builder.add(1, 2, 4.0);
    model::SparseMatrixCSC mat = builder.build();

    // Build CSR of A^T (dimensions 3x2)
    DeviceCsr At = DeviceCsr::transpose_from_csc(mat);
    require(At.rows() == 3, "At.rows must be 3");
    require(At.cols() == 2, "At.cols must be 2");
    require(At.nnz() == 4, "At.nnz must be 4");

    // Vector y = [2.0, -1.0]^T
    // A^T * y = [ 1(2)+0(-1), 0(2)+2(-1), 3(2)+4(-1) ]^T = [ 2.0, -2.0, 2.0 ]^T
    DeviceBuffer<double> d_y(std::vector<double>{2.0, -1.0});
    DeviceBuffer<double> d_z(3);
    spmv_transpose(At, d_y, d_z);

    std::vector<double> h_z = d_z.to_vector();
    require(std::abs(h_z[0] - 2.0) <= 1e-15, "transpose [0]");
    require(std::abs(h_z[1] - (-2.0)) <= 1e-15, "transpose [1]");
    require(std::abs(h_z[2] - 2.0) <= 1e-15, "transpose [2]");

    // Dimension mismatch checks
    bool caught_y_mismatch = false;
    try {
        DeviceBuffer<double> d_y_wrong(3);
        spmv_transpose(At, d_y_wrong, d_z);
    } catch (const std::invalid_argument&) {
        caught_y_mismatch = true;
    }
    require(caught_y_mismatch, "failed to catch transpose y mismatch");

    bool caught_z_mismatch = false;
    try {
        DeviceBuffer<double> d_z_wrong(2);
        spmv_transpose(At, d_y, d_z_wrong);
    } catch (const std::invalid_argument&) {
        caught_z_mismatch = true;
    }
    require(caught_z_mismatch, "failed to catch transpose z mismatch");

    std::cout << "test_synthetic_spmv_transpose: PASS\n";
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

std::vector<double> csc_multiply_transpose(const markov_cero::model::SparseMatrixCSC& mat,
                                           const std::vector<double>& y) {
    std::vector<double> z(mat.column_count, 0.0);
    for (std::size_t col = 0; col < mat.column_count; ++col) {
        double sum = 0.0;
        const std::size_t start = mat.column_start[col];
        const std::size_t end = mat.column_start[col + 1];
        for (std::size_t p = start; p < end; ++p) {
            sum += mat.value[p] * y[mat.row_index[p]];
        }
        z[col] = sum;
    }
    return z;
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

    // 1. Forward SpMV (A * x)
    DeviceCsr A = DeviceCsr::from_csc(mdl.matrix);
    require(A.rows() == m, "A.rows mismatch");
    require(A.cols() == n, "A.cols mismatch");
    require(A.nnz() == nnz, "A.nnz mismatch");

    // 2. Transpose SpMV (A^T * y)
    DeviceCsr At = DeviceCsr::transpose_from_csc(mdl.matrix);
    require(At.rows() == n, "At.rows mismatch");
    require(At.cols() == m, "At.cols mismatch");
    require(At.nnz() == nnz, "At.nnz mismatch");

    // Forward test with continuous random vector in [-1.0, 1.0]
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> h_x(n);
    for (std::size_t j = 0; j < n; ++j) {
        h_x[j] = dist(rng);
    }
    std::vector<double> expected_y = mdl.matrix.multiply(h_x);
    DeviceBuffer<double> d_x(h_x);
    DeviceBuffer<double> d_y(m);
    DeviceBuffer<double> d_y_cpu(m);
    spmv(A, d_x, d_y);
    spmv_cpu(A, d_x, d_y_cpu);
    std::vector<double> actual_y = d_y.to_vector();
    std::vector<double> actual_y_cpu = d_y_cpu.to_vector();

    double fwd_diff_cpu = compute_max_abs_diff(actual_y, actual_y_cpu);
    double fwd_diff_csc = compute_max_rel_diff(expected_y, actual_y);
    require(fwd_diff_cpu <= 1e-12, "forward kernel vs CPU diff exceeds 1e-12");
    require(fwd_diff_csc <= 1e-12, "forward CSC vs CSR relative diff exceeds 1e-12");

    // Transpose test with continuous random vector in [-1.0, 1.0]
    std::vector<double> h_y(m);
    for (std::size_t i = 0; i < m; ++i) {
        h_y[i] = dist(rng);
    }
    std::vector<double> expected_z = csc_multiply_transpose(mdl.matrix, h_y);
    DeviceBuffer<double> d_yt(h_y);
    DeviceBuffer<double> d_z(n);
    DeviceBuffer<double> d_z_cpu(n);
    spmv_transpose(At, d_yt, d_z);
    spmv_transpose_cpu(At, d_yt, d_z_cpu);
    std::vector<double> actual_z = d_z.to_vector();
    std::vector<double> actual_z_cpu = d_z_cpu.to_vector();

    double trans_diff_cpu = compute_max_abs_diff(actual_z, actual_z_cpu);
    double trans_diff_csc = compute_max_rel_diff(expected_z, actual_z);
    require(trans_diff_cpu <= 1e-12, "transpose kernel vs CPU diff exceeds 1e-12");
    require(trans_diff_csc <= 1e-12, "transpose CSC vs CSR relative diff exceeds 1e-12");

    std::cout << "  " << std::left << std::setw(12) << instance_name
              << " (" << std::right << std::setw(5) << m << " x "
              << std::setw(5) << n << ", nnz="
              << std::setw(6) << nnz << ") fwd_diff="
              << std::scientific << std::setprecision(2) << fwd_diff_cpu
              << " trans_diff="
              << std::scientific << std::setprecision(2) << trans_diff_cpu
              << " [PASS]\n";
}

void test_all_netlib_matrices() {
    std::cout << "--- Testing SpMV & Transpose SpMV on Netlib Matrix Corpus ---\n";
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
              << " Netlib matrices passed forward & transpose equivalence tests.\n";
}

} // namespace

int main() {
    std::cout << "=== Markov-Cero SpMV & Transpose Equivalence Test (T-5.03/T-5.04) ===\n";
    test_synthetic_spmv();
    test_synthetic_spmv_transpose();
    test_all_netlib_matrices();
    std::cout << "=== All Equivalence Tests Passed Successfully ===\n";
    return 0;
}
