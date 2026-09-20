#include "markov_cero/io/mps.hpp"
#include "markov_cero/lp/reference/revised_simplex.hpp"
#include "markov_cero/presolve/presolve.hpp"
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

void test_presolve_empty_row_infeasible() {
    markov_cero::transform::SparseCanonicalModel model;
    model.matrix.rows = 1;
    model.matrix.columns = 1;
    model.matrix.column_offsets = {0, 0};
    model.rhs = {10.0}; // 0*x = 10 -> infeasible
    model.objective = {1.0};
    model.record.objective_sign = 1.0;

    const auto res = markov_cero::presolve::presolve(model);
    assert(res.status == markov_cero::lp::reference::SolveStatus::infeasible);
}

void test_presolve_empty_column_unbounded() {
    markov_cero::transform::SparseCanonicalModel model;
    model.matrix.rows = 1;
    model.matrix.columns = 1;
    model.matrix.column_offsets = {0, 0};
    model.rhs = {0.0};
    model.objective = {-5.0}; // empty col with negative cost -> unbounded
    model.record.objective_sign = 1.0;

    const auto res = markov_cero::presolve::presolve(model);
    assert(res.status == markov_cero::lp::reference::SolveStatus::unbounded);
}

void test_presolve_row_singleton_reduction() {
    // 2 rows, 2 cols:
    // row 0: 2 * x0 = 6  (row singleton -> fixes x0 = 3)
    // row 1: x0 + x1 = 7 (incident row -> becomes x1 = 4)
    markov_cero::transform::SparseCanonicalModel model;
    model.matrix.rows = 2;
    model.matrix.columns = 2;
    model.matrix.column_offsets = {0, 2, 3};
    model.matrix.row_indices = {0, 1, 1};
    model.matrix.values = {2.0, 1.0, 1.0};
    model.rhs = {6.0, 7.0};
    model.objective = {10.0, 5.0};
    model.record.objective_sign = 1.0;

    const auto presolved = markov_cero::presolve::presolve(model);
    assert(presolved.status == markov_cero::lp::reference::SolveStatus::optimal);
    assert(presolved.statistics.row_singletons_removed >= 1);

    // After row singletons and fixed variable elimination, model is fully reduced
    markov_cero::lp::reference::Result reduced_sol;
    reduced_sol.status = markov_cero::lp::reference::SolveStatus::optimal;
    reduced_sol.primal.assign(presolved.model.matrix.columns, 0.0);
    reduced_sol.dual.assign(presolved.model.matrix.rows, 0.0);

    const auto restored = markov_cero::presolve::postsolve(presolved.stack, reduced_sol, model);
    assert(std::abs(restored.primal[0] - 3.0) < 1e-9);
    assert(std::abs(restored.primal[1] - 4.0) < 1e-9);
    // Objective: 10*3 + 5*4 = 50
    assert(std::abs(restored.objective - 50.0) < 1e-9);
}

void test_blend_with_presolve_matches_without_presolve() {
    const auto mps = markov_cero::io::parse_mps_string(blend_mps);
    const auto sparse = markov_cero::transform::sparse_canonicalize(mps);

    // Solve without presolve
    const auto dense = sparse.to_dense();
    const auto sol_unpresolved = markov_cero::lp::reference::solve(dense);
    assert(sol_unpresolved.status == markov_cero::lp::reference::SolveStatus::optimal);

    // Solve with presolve
    const auto presolved = markov_cero::presolve::presolve(sparse);
    assert(presolved.status == markov_cero::lp::reference::SolveStatus::optimal);

    markov_cero::lp::reference::Result sol_reduced;
    if (presolved.model.matrix.rows > 0 && presolved.model.matrix.columns > 0) {
        sol_reduced = markov_cero::lp::reference::solve(presolved.model.to_dense());
        assert(sol_reduced.status == markov_cero::lp::reference::SolveStatus::optimal);
    } else {
        sol_reduced.status = markov_cero::lp::reference::SolveStatus::optimal;
    }

    const auto sol_postsolved =
        markov_cero::presolve::postsolve(presolved.stack, sol_reduced, sparse);
    assert(std::abs(sol_postsolved.objective - sol_unpresolved.objective) < 1e-7);

    // Verify primal variables reconstructed
    auto prim_orig_presolved =
        markov_cero::transform::reconstruct_primal(sparse, sol_postsolved.primal);
    auto prim_orig_unpresolved =
        markov_cero::transform::reconstruct_primal(sparse, sol_unpresolved.primal);
    for (std::size_t j = 0; j < prim_orig_presolved.size(); ++j) {
        assert(std::abs(prim_orig_presolved[j] - prim_orig_unpresolved[j]) < 1e-6);
    }
}

} // namespace

int main() {
    try {
        test_presolve_empty_row_infeasible();
        test_presolve_empty_column_unbounded();
        test_presolve_row_singleton_reduction();
        test_blend_with_presolve_matches_without_presolve();
        std::cout << "presolve tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
