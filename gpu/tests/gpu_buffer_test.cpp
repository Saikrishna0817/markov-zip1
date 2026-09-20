#include "markov_cero/gpu/buffer.hpp"
#include "markov_cero/gpu/csr.hpp"
#include "markov_cero/gpu/device.hpp"
#include "markov_cero/io/mps.hpp"
#include "markov_cero/linalg/sparse_basis.hpp"
#include "markov_cero/model/model.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

namespace {

void test_buffer_lifecycle() {
    using namespace markov_cero::gpu;

    DeviceBuffer<double> empty_buf;
    assert(empty_buf.size() == 0);
    assert(empty_buf.bytes() == 0);
    assert(empty_buf.empty());
    assert(empty_buf.data() == nullptr);

    empty_buf.allocate(0);
    assert(empty_buf.size() == 0);

    DeviceBuffer<double> sized_buf(64);
    assert(sized_buf.size() == 64);
    assert(sized_buf.bytes() == 64 * sizeof(double));
    assert(!sized_buf.empty());
    assert(sized_buf.data() != nullptr);

    DeviceBuffer<double> moved_buf(std::move(sized_buf));
    assert(moved_buf.size() == 64);
    assert(moved_buf.data() != nullptr);
    assert(sized_buf.size() == 0);
    assert(sized_buf.data() == nullptr);

    DeviceBuffer<double> assigned_buf;
    assigned_buf = std::move(moved_buf);
    assert(assigned_buf.size() == 64);
    assert(moved_buf.size() == 0);

    assigned_buf.release();
    assert(assigned_buf.size() == 0);
    assert(assigned_buf.data() == nullptr);

    std::cout << "test_buffer_lifecycle: PASS\n";
}

void test_buffer_roundtrip_double() {
    using namespace markov_cero::gpu;

    constexpr std::size_t N = 1024;
    std::vector<double> host_src(N);

    // Populate with deterministic non-trivial and edge case values
    host_src[0] = 0.0;
    host_src[1] = -0.0;
    host_src[2] = 1.0;
    host_src[3] = -1.0;
    host_src[4] = 1e-300;
    host_src[5] = 1e300;
    host_src[6] = 3.14159265358979323846;
    host_src[7] = 2.71828182845904523536;

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (std::size_t i = 8; i < N; ++i) {
        host_src[i] = dist(rng);
    }

    DeviceBuffer<double> dev_buf(host_src);
    assert(dev_buf.size() == N);

    std::vector<double> host_dst(N, 0.0);
    dev_buf.download(host_dst.data(), N);

    // Bitwise exact check
    int cmp = std::memcmp(host_src.data(), host_dst.data(), N * sizeof(double));
    assert(cmp == 0);

    std::vector<double> vec_dst = dev_buf.to_vector();
    assert(vec_dst.size() == N);
    cmp = std::memcmp(host_src.data(), vec_dst.data(), N * sizeof(double));
    assert(cmp == 0);

    std::cout << "test_buffer_roundtrip_double: PASS\n";
}

void test_buffer_roundtrip_size_t() {
    using namespace markov_cero::gpu;

    constexpr std::size_t N = 2048;
    std::vector<std::size_t> host_src(N);
    for (std::size_t i = 0; i < N; ++i) {
        host_src[i] = i * 37U + 13U;
    }

    DeviceBuffer<std::size_t> dev_buf(N);
    dev_buf.upload(host_src.data(), N);

    std::vector<std::size_t> host_dst(N, 0);
    dev_buf.download(host_dst);

    int cmp = std::memcmp(host_src.data(), host_dst.data(), N * sizeof(std::size_t));
    assert(cmp == 0);

    std::cout << "test_buffer_roundtrip_size_t: PASS\n";
}

void test_buffer_bounds_checking() {
    using namespace markov_cero::gpu;

    DeviceBuffer<double> buf(10);
    std::vector<double> data(20, 1.0);

    bool caught_upload_overflow = false;
    try {
        buf.upload(data.data(), 20);
    } catch (const std::invalid_argument&) {
        caught_upload_overflow = true;
    }
    assert(caught_upload_overflow);

    bool caught_download_overflow = false;
    try {
        buf.download(data.data(), 20);
    } catch (const std::invalid_argument&) {
        caught_download_overflow = true;
    }
    assert(caught_download_overflow);

    bool caught_null_src = false;
    try {
        buf.upload(nullptr, 5);
    } catch (const std::invalid_argument&) {
        caught_null_src = true;
    }
    assert(caught_null_src);

    std::cout << "test_buffer_bounds_checking: PASS\n";
}

void test_csr_construction_roundtrip() {
    using namespace markov_cero;
    using namespace markov_cero::gpu;

    // Construct a canonical 4x5 sparse CSC matrix:
    // [ 1.0   0.0   2.0   0.0   0.0 ]
    // [ 0.0   3.0   0.0   4.0   0.0 ]
    // [ 5.0   0.0   0.0   6.0   7.0 ]
    // [ 0.0   8.0   9.0   0.0  10.0 ]
    linalg::SparseCsc csc;
    csc.rows = 4;
    csc.columns = 5;
    csc.column_offsets = {0, 2, 4, 6, 8, 10};
    csc.row_indices = {
        0, 2,  // col 0: (0, 1.0), (2, 5.0)
        1, 3,  // col 1: (1, 3.0), (3, 8.0)
        0, 3,  // col 2: (0, 2.0), (3, 9.0)
        1, 2,  // col 3: (1, 4.0), (2, 6.0)
        2, 3   // col 4: (2, 7.0), (3, 10.0)
    };
    csc.values = {1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 6.0, 7.0, 10.0};
    csc.validate();

    DeviceCsr dev_csr = DeviceCsr::from_csc(csc);
    assert(dev_csr.rows() == 4);
    assert(dev_csr.cols() == 5);
    assert(dev_csr.nnz() == 10);

    // Download to host CSR and check CSR structure
    std::vector<std::size_t> row_offsets, col_indices;
    std::vector<double> values;
    dev_csr.download_to(row_offsets, col_indices, values);

    // Expected CSR row offsets:
    // row 0: (0, 1.0), (2, 2.0) -> count 2
    // row 1: (1, 3.0), (3, 4.0) -> count 2
    // row 2: (0, 5.0), (3, 6.0), (4, 7.0) -> count 3
    // row 3: (1, 8.0), (2, 9.0), (4, 10.0) -> count 3
    const std::vector<std::size_t> expected_row_offsets = {0, 2, 4, 7, 10};
    assert(row_offsets == expected_row_offsets);

    const std::vector<std::size_t> expected_col_indices = {0, 2, 1, 3, 0, 3, 4, 1, 2, 4};
    assert(col_indices == expected_col_indices);

    const std::vector<double> expected_values = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
    };
    assert(values == expected_values);

    // Test round-trip conversion to CSC host
    linalg::SparseCsc rt_csc = dev_csr.to_csc_host();
    assert(rt_csc.rows == csc.rows);
    assert(rt_csc.columns == csc.columns);
    assert(rt_csc.column_offsets == csc.column_offsets);
    assert(rt_csc.row_indices == csc.row_indices);
    int val_cmp = std::memcmp(rt_csc.values.data(), csc.values.data(),
                              csc.values.size() * sizeof(double));
    assert(val_cmp == 0);

    // Test matrix-vector multiplication equivalence on host
    std::vector<double> x = {1.5, -2.0, 0.5, 3.0, -1.0};
    std::vector<double> y_csc(csc.rows, 0.0);
    for (std::size_t j = 0; j < csc.columns; ++j) {
        for (std::size_t p = csc.column_offsets[j]; p < csc.column_offsets[j + 1]; ++p) {
            y_csc[csc.row_indices[p]] += csc.values[p] * x[j];
        }
    }

    std::vector<double> y_csr(dev_csr.rows(), 0.0);
    for (std::size_t i = 0; i < dev_csr.rows(); ++i) {
        for (std::size_t p = row_offsets[i]; p < row_offsets[i + 1]; ++p) {
            y_csr[i] += values[p] * x[col_indices[p]];
        }
    }

    for (std::size_t i = 0; i < dev_csr.rows(); ++i) {
        assert(y_csc[i] == y_csr[i]);
    }

    std::cout << "test_csr_construction_roundtrip: PASS\n";
}

void test_model_sparse_matrix_roundtrip() {
    using namespace markov_cero;
    using namespace markov_cero::gpu;

    model::SparseMatrixBuilder builder(3, 3);
    builder.add(0, 0, 11.0);
    builder.add(0, 2, 13.0);
    builder.add(1, 1, 22.0);
    builder.add(2, 0, 31.0);
    builder.add(2, 1, 32.0);
    builder.add(2, 2, 33.0);

    model::SparseMatrixCSC orig = builder.build();
    DeviceCsr dev_csr = DeviceCsr::from_csc(orig);

    assert(dev_csr.rows() == 3);
    assert(dev_csr.cols() == 3);
    assert(dev_csr.nnz() == 6);

    model::SparseMatrixCSC rt = dev_csr.to_model_csc_host();
    assert(rt.row_count == orig.row_count);
    assert(rt.column_count == orig.column_count);
    assert(rt.column_start == orig.column_start);
    assert(rt.row_index == orig.row_index);
    int val_cmp = std::memcmp(rt.value.data(), orig.value.data(),
                              orig.value.size() * sizeof(double));
    assert(val_cmp == 0);

    std::cout << "test_model_sparse_matrix_roundtrip: PASS\n";
}

void test_refinery_mps_roundtrip() {
    using namespace markov_cero;
    using namespace markov_cero::gpu;

    const std::string path = "examples/refinery/refinery-feasible.mps";
    std::ifstream file(path);
    model::Model mdl;
    if (file.is_open()) {
        mdl = io::parse_mps(file);
    } else {
        std::ifstream alt_file("../" + path);
        assert(alt_file.is_open());
        mdl = io::parse_mps(alt_file);
    }

    DeviceCsr dev_csr = DeviceCsr::from_csc(mdl.matrix);
    assert(dev_csr.rows() == mdl.matrix.row_count);
    assert(dev_csr.cols() == mdl.matrix.column_count);
    assert(dev_csr.nnz() == mdl.matrix.value.size());

    model::SparseMatrixCSC rt = dev_csr.to_model_csc_host();
    assert(rt.row_count == mdl.matrix.row_count);
    assert(rt.column_count == mdl.matrix.column_count);
    assert(rt.column_start == mdl.matrix.column_start);
    assert(rt.row_index == mdl.matrix.row_index);

    int val_cmp = std::memcmp(rt.value.data(), mdl.matrix.value.data(),
                              mdl.matrix.value.size() * sizeof(double));
    assert(val_cmp == 0);

    std::cout << "test_refinery_mps_roundtrip: PASS ("
              << dev_csr.rows() << "x" << dev_csr.cols() << ", "
              << dev_csr.nnz() << " nonzeros)\n";
}

void test_device_query() {
    using namespace markov_cero::gpu;

    bool available = is_gpu_available();
    int count = get_device_count();
    DeviceInfo info = get_device_info(0);

    std::cout << "test_device_query: available=" << std::boolalpha << available
              << ", count=" << count << ", device_name='" << info.name << "'\n";

    synchronize_device();
    std::cout << "test_device_query: PASS\n";
}

} // namespace

int main() {
    std::cout << "=== Markov-Cero GPU Buffer & CSR Tests (T-5.02) ===\n";
    test_buffer_lifecycle();
    test_buffer_roundtrip_double();
    test_buffer_roundtrip_size_t();
    test_buffer_bounds_checking();
    test_csr_construction_roundtrip();
    test_model_sparse_matrix_roundtrip();
    test_refinery_mps_roundtrip();
    test_device_query();
    std::cout << "All GPU buffer & CSR tests passed successfully.\n";
    return 0;
}
