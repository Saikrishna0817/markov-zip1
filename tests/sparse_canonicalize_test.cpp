#include "markov_cero/io/mps.hpp"
#include "markov_cero/transform/canonicalize.hpp"
#include "markov_cero/transform/sparse_canonical_model.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

namespace {

constexpr const char* blend_mps = R"(NAME BLEND
OBJSENSE
 MIN
ROWS
 N COST
 E TOTAL
 L SULFUR
COLUMNS
 A COST 40 TOTAL 1
 A SULFUR 0.01
 B COST 30 TOTAL 1
 B SULFUR 0.03
RHS
 RHS1 TOTAL 100 SULFUR 2
BOUNDS
 LO BND A 0
 LO BND B 0
ENDATA
)";

void test_blend_canonical_equivalence() {
    const auto model = markov_cero::io::parse_mps_string(blend_mps);
    const auto dense = markov_cero::transform::canonicalize(model);
    const auto sparse = markov_cero::transform::sparse_canonicalize(model);

    assert(dense.matrix.rows == sparse.matrix.rows);
    assert(dense.matrix.columns == sparse.matrix.columns);
    assert(dense.rhs.size() == sparse.rhs.size());
    assert(dense.objective.size() == sparse.objective.size());
    assert(std::abs(dense.objective_offset - sparse.objective_offset) < 1e-12);

    for (std::size_t i = 0; i < dense.rhs.size(); ++i) {
        assert(std::abs(dense.rhs[i] - sparse.rhs[i]) < 1e-12);
    }
    for (std::size_t j = 0; j < dense.objective.size(); ++j) {
        assert(std::abs(dense.objective[j] - sparse.objective[j]) < 1e-12);
    }

    const auto converted_dense = sparse.to_dense();
    assert(converted_dense.matrix.values.size() == dense.matrix.values.size());
    for (std::size_t k = 0; k < dense.matrix.values.size(); ++k) {
        assert(std::abs(converted_dense.matrix.values[k] - dense.matrix.values[k]) < 1e-12);
    }

    // Test matrix-vector multiplication
    std::vector<double> x(sparse.matrix.columns, 1.5);
    auto y_sparse = sparse.multiply(x);
    auto y_dense = markov_cero::linalg::multiply(dense.matrix, x);
    assert(y_sparse.size() == y_dense.size());
    for (std::size_t i = 0; i < y_sparse.size(); ++i) {
        assert(std::abs(y_sparse[i] - y_dense[i]) < 1e-12);
    }

    // Test transpose multiplication
    std::vector<double> y(sparse.matrix.rows, 2.5);
    auto x_sparse = sparse.multiply_transpose(y);
    auto x_dense = markov_cero::linalg::multiply_transpose(dense.matrix, y);
    assert(x_sparse.size() == x_dense.size());
    for (std::size_t j = 0; j < x_sparse.size(); ++j) {
        assert(std::abs(x_sparse[j] - x_dense[j]) < 1e-12);
    }

    // Test reconstruction
    auto prim_dense = markov_cero::transform::reconstruct_primal(dense, x);
    auto prim_sparse = markov_cero::transform::reconstruct_primal(sparse, x);
    assert(prim_dense.size() == prim_sparse.size());
    for (std::size_t j = 0; j < prim_dense.size(); ++j) {
        assert(std::abs(prim_dense[j] - prim_sparse[j]) < 1e-12);
    }
}

} // namespace

int main() {
    try {
        test_blend_canonical_equivalence();
        std::cout << "sparse canonicalize tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
