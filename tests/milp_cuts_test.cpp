#include "markov_cero/lp/dual/dual_simplex.hpp"
#include "markov_cero/milp/cuts.hpp"
#include "markov_cero/transform/sparse_canonical_model.hpp"

#include <cassert>
#include <iostream>

namespace {

void test_gomory_cut_generation() {
    markov_cero::model::Model model;
    model.name = "CUT_TEST";
    model.objective_sense = markov_cero::model::ObjectiveSense::minimize;
    model.objective = {-10.0, -14.0, -12.0};

    markov_cero::model::SparseMatrixBuilder builder(1, 3);
    builder.add(0, 0, 4.0);
    builder.add(0, 1, 6.0);
    builder.add(0, 2, 5.0);
    model.matrix = builder.build();

    model.row_lower = {markov_cero::model::Bound::negative_infinity()};
    model.row_upper = {markov_cero::model::Bound::finite(10.0)};
    model.row_name = {"CAPACITY"};

    model.variable_lower = {markov_cero::model::Bound::finite(0.0),
                            markov_cero::model::Bound::finite(0.0),
                            markov_cero::model::Bound::finite(0.0)};
    model.variable_upper = {markov_cero::model::Bound::finite(1.0),
                            markov_cero::model::Bound::finite(1.0),
                            markov_cero::model::Bound::finite(1.0)};
    model.variable_type = {markov_cero::model::VariableType::binary,
                           markov_cero::model::VariableType::binary,
                           markov_cero::model::VariableType::binary};
    model.variable_name = {"X1", "X2", "X3"};
    model.validate();

    // Solve continuous LP relaxation
    const auto canon =
        markov_cero::transform::sparse_canonicalize(model, /*relax_integrality=*/true);
    const auto dense = canon.to_dense();
    const auto lpres = markov_cero::lp::reference::solve(dense);
    assert(lpres.status == markov_cero::lp::reference::SolveStatus::optimal);

    const auto primal = markov_cero::transform::reconstruct_primal(canon, lpres.primal);
    const auto basis_state = markov_cero::lp::dual::make_basis_state(dense, lpres.basis);

    const auto cuts = markov_cero::milp::generate_gomory_cuts(model, primal, canon, basis_state, 5);
    if (!cuts.empty()) {
        for (const auto& cut : cuts) {
            assert(cut.violation > 0.0); // Strictly cuts off fractional LP point
            // Verify integer feasible points satisfy the cut:
            // e.g. x = (1, 1, 0)
            double lhs =
                cut.coefficients[0] * 1.0 + cut.coefficients[1] * 1.0 + cut.coefficients[2] * 0.0;
            assert(lhs >= cut.rhs - 1e-6); // Must not cut off valid integer optimum
        }
    }

    auto augmented_model = model;
    markov_cero::milp::add_cuts_to_model(augmented_model, cuts);
    assert(augmented_model.matrix.row_count == model.matrix.row_count + cuts.size());
    std::cout << "[+] test_gomory_cut_generation passed (" << cuts.size() << " cuts generated)\n";
}

} // namespace

int main() {
    try {
        test_gomory_cut_generation();
        std::cout << "All cuts tests PASSED successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[-] Error: " << e.what() << "\n";
        return 1;
    }
}
