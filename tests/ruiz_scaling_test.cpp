#include "markov_cero/io/mps.hpp"
#include "markov_cero/lp/reference/revised_simplex.hpp"
#include "markov_cero/scale/ruiz_scaling.hpp"
#include "markov_cero/transform/sparse_canonical_model.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

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

void test_ruiz_scaling_equilibration() {
    markov_cero::transform::SparseCanonicalModel model;
    model.matrix.rows = 2;
    model.matrix.columns = 2;
    model.matrix.column_offsets = {0, 2, 4};
    model.matrix.row_indices = {0, 1, 0, 1};
    model.matrix.values = {1000.0, 0.001, 2000.0, 0.002};
    model.rhs = {3000.0, 0.003};
    model.objective = {10.0, 20.0};
    model.record.objective_sign = 1.0;

    markov_cero::scale::RuizOptions opts;
    opts.max_iterations = 10;
    opts.tolerance = 1e-3;

    auto model_copy = model;
    const auto scalers = markov_cero::scale::equilibrate(model_copy, opts);

    // After Ruiz equilibration, max row and col norms should be close to 1
    for (std::size_t j = 0; j < model_copy.matrix.columns; ++j) {
        double col_max = 0.0;
        const std::size_t start = model_copy.matrix.column_offsets[j];
        const std::size_t end = model_copy.matrix.column_offsets[j + 1];
        for (std::size_t k = start; k < end; ++k) {
            col_max = std::max(col_max, std::abs(model_copy.matrix.values[k]));
        }
        assert(col_max > 0.05 && col_max < 20.0);
    }
}

void test_blend_scaled_matches_unscaled() {
    const auto mps = markov_cero::io::parse_mps_string(blend_mps);
    const auto sparse = markov_cero::transform::sparse_canonicalize(mps);

    // Unscaled solve
    const auto dense_unscaled = sparse.to_dense();
    const auto sol_unscaled = markov_cero::lp::reference::solve(dense_unscaled);
    assert(sol_unscaled.status == markov_cero::lp::reference::SolveStatus::optimal);

    // Scaled solve
    auto sparse_scaled = sparse;
    const auto scalers = markov_cero::scale::equilibrate(sparse_scaled);
    const auto dense_scaled = sparse_scaled.to_dense();
    auto sol_scaled = markov_cero::lp::reference::solve(dense_scaled);
    assert(sol_scaled.status == markov_cero::lp::reference::SolveStatus::optimal);

    // Unscale solution
    markov_cero::scale::unscale_solution(scalers, sol_scaled);

    // Objective should be identical
    assert(std::abs(sol_scaled.objective - sol_unscaled.objective) < 1e-7);

    // Primal variables should match
    for (std::size_t j = 0; j < sol_unscaled.primal.size(); ++j) {
        assert(std::abs(sol_scaled.primal[j] - sol_unscaled.primal[j]) < 1e-6);
    }
}

} // namespace

int main() {
    try {
        test_ruiz_scaling_equilibration();
        test_blend_scaled_matches_unscaled();
        std::cout << "ruiz scaling tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
