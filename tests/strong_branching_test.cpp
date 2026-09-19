#include "markov_cero/milp/strong_branching.hpp"
#include "markov_cero/milp/cuts.hpp"
#include "markov_cero/lp/dual/dual_simplex.hpp"
#include "markov_cero/lp/reference/revised_simplex.hpp"
#include "markov_cero/transform/sparse_canonical_model.hpp"
#include "markov_cero/verify/primal_verifier.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

void test_mir_mathematical_function() {
    // Marchand & Wolsey (2001) MIR rounding function tests
    // f0 = 0.4
    constexpr double f0 = 0.4;

    // Test case 1: fj > f0
    // a = 0.7 -> fl = 0, fj = 0.7 -> alpha = 0 + (0.7 - 0.4) / 0.6 = 0.5
    const double a1 = 0.7;
    const double alpha1 = markov_cero::milp::mir_function(a1, f0);
    assert(std::abs(alpha1 - 0.5) < 1e-9);

    // Test case 2: fj <= f0
    // a = 0.3 -> fl = 0, fj = 0.3 -> alpha = 0.0
    const double a2 = 0.3;
    const double alpha2 = markov_cero::milp::mir_function(a2, f0);
    assert(std::abs(alpha2 - 0.0) < 1e-9);

    // Test case 3: negative a
    // a = -0.3 -> fl = -1.0, fj = 0.7 > 0.4 -> alpha = -1 + 0.3/0.6 = -0.5
    const double a3 = -0.3;
    const double alpha3 = markov_cero::milp::mir_function(a3, f0);
    assert(std::abs(alpha3 - (-0.5)) < 1e-9);

    // Test case 4: integer a
    const double a4 = 2.0;
    const double alpha4 = markov_cero::milp::mir_function(a4, f0);
    assert(std::abs(alpha4 - 2.0) < 1e-9);

    std::cout << "[+] test_mir_mathematical_function passed\n";
}

void test_cut_efficacy_and_filtering() {
    markov_cero::milp::Cut cut1;
    cut1.coefficients = {3.0, 4.0};
    cut1.rhs = 5.0;
    cut1.violation = 2.5;

    // Norm = 5.0, Efficacy = 2.5 / 5.0 = 0.5
    const double eff1 = markov_cero::milp::compute_cut_efficacy(cut1);
    assert(std::abs(eff1 - 0.5) < 1e-9);

    // Orthogonal cuts: (1, 0) and (0, 1) -> cos_sim = 0
    markov_cero::milp::Cut cut_x;
    cut_x.coefficients = {1.0, 0.0};
    cut_x.rhs = 1.0;
    cut_x.violation = 0.5;

    markov_cero::milp::Cut cut_y;
    cut_y.coefficients = {0.0, 1.0};
    cut_y.rhs = 1.0;
    cut_y.violation = 0.4;

    const double sim_xy = markov_cero::milp::compute_cosine_similarity(cut_x, cut_y);
    assert(std::abs(sim_xy - 0.0) < 1e-9);

    // Near parallel cuts: (1, 0) and (0.999, 0.001)
    markov_cero::milp::Cut cut_x_parallel;
    cut_x_parallel.coefficients = {0.9999, 0.001};
    cut_x_parallel.rhs = 1.0;
    cut_x_parallel.violation = 0.45;

    const double sim_parallel = markov_cero::milp::compute_cosine_similarity(cut_x, cut_x_parallel);
    assert(sim_parallel > 0.95);

    // Low violation cut
    markov_cero::milp::Cut cut_low_violation;
    cut_low_violation.coefficients = {1.0, 1.0};
    cut_low_violation.rhs = 1.0;
    cut_low_violation.violation = 1e-7;

    std::vector<markov_cero::milp::Cut> pool = {cut_x, cut_x_parallel, cut_y, cut_low_violation};
    auto filtered = markov_cero::milp::filter_cuts(pool, /*max_cuts=*/10, /*min_violation=*/1e-4, /*max_parallelism=*/0.95);

    // cut_low_violation (< 1e-4) should be discarded
    // cut_x_parallel should be discarded because it is parallel to cut_x (> 0.95)
    // cut_x and cut_y should be retained
    assert(filtered.size() == 2);
    assert(std::abs(filtered[0].coefficients[0] - 1.0) < 1e-6); // highest efficacy first
    assert(std::abs(filtered[1].coefficients[1] - 1.0) < 1e-6);

    std::cout << "[+] test_cut_efficacy_and_filtering passed\n";
}

void test_mir_cuts_tighten_relaxation_and_preserve_integers() {
    // Model:
    // Min -10 x1 - 14 x2 - 12 x3
    // s.t. 4 x1 + 6 x2 + 5 x3 <= 10
    // x1, x2, x3 in {0, 1}
    markov_cero::model::Model model;
    model.name = "MIR_KNAPSACK";
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

    model.variable_lower = {
        markov_cero::model::Bound::finite(0.0),
        markov_cero::model::Bound::finite(0.0),
        markov_cero::model::Bound::finite(0.0)
    };
    model.variable_upper = {
        markov_cero::model::Bound::finite(1.0),
        markov_cero::model::Bound::finite(1.0),
        markov_cero::model::Bound::finite(1.0)
    };
    model.variable_type = {
        markov_cero::model::VariableType::binary,
        markov_cero::model::VariableType::binary,
        markov_cero::model::VariableType::binary
    };
    model.variable_name = {"X1", "X2", "X3"};
    model.validate();

    // 1. Solve root LP relaxation
    const auto canon = markov_cero::transform::sparse_canonicalize(model, /*relax_integrality=*/true);
    const auto dense = canon.to_dense();
    const auto lpres = markov_cero::lp::reference::solve(dense);
    assert(lpres.status == markov_cero::lp::reference::SolveStatus::optimal);

    const auto primal = markov_cero::transform::reconstruct_primal(canon, lpres.primal);
    const double initial_relaxation_obj = markov_cero::transform::reconstruct_objective(canon, lpres.objective);
    const auto basis_state = markov_cero::lp::dual::make_basis_state(dense, lpres.basis);

    // 2. Generate MIR cuts
    const auto mir_cuts = markov_cero::milp::generate_mir_cuts(model, primal, canon, basis_state, /*max_cuts=*/5);
    assert(!mir_cuts.empty());

    // 3. Verify cuts strictly cut off fractional relaxation point and do NOT cut off any integer feasible solutions
    const std::vector<std::vector<double>> integer_feasible_points = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 1.0, 0.0}, // Capacity: 4*1 + 6*1 = 10 <= 10 (Integer optimum, obj = -24)
        {1.0, 0.0, 1.0}  // Capacity: 4*1 + 5*1 = 9 <= 10
    };

    for (const auto& cut : mir_cuts) {
        assert(cut.violation > 1e-4);

        // Verify that all integer feasible points satisfy the cut: sum c_j x_j >= rhs
        for (const auto& pt : integer_feasible_points) {
            double lhs = 0.0;
            for (std::size_t j = 0; j < 3; ++j) {
                lhs += cut.coefficients[j] * pt[j];
            }
            assert(lhs >= cut.rhs - 1e-6); // Must never cut off integer feasible solutions!
        }
    }

    // 4. Add cuts to model and re-solve relaxation to verify bound tightening
    auto cut_model = model;
    markov_cero::milp::add_cuts_to_model(cut_model, mir_cuts);

    const auto cut_canon = markov_cero::transform::sparse_canonicalize(cut_model, /*relax_integrality=*/true);
    const auto cut_dense = cut_canon.to_dense();
    const auto cut_lpres = markov_cero::lp::reference::solve(cut_dense);
    assert(cut_lpres.status == markov_cero::lp::reference::SolveStatus::optimal);

    const double tightened_obj = markov_cero::transform::reconstruct_objective(cut_canon, cut_lpres.objective);

    // For minimization, adding valid cuts increases the lower bound (tightens relaxation towards -24.0)
    assert(tightened_obj >= initial_relaxation_obj - 1e-6);
    std::cout << "[+] test_mir_cuts_tighten_relaxation_and_preserve_integers passed: "
              << "initial relaxation obj=" << initial_relaxation_obj
              << ", with MIR cuts obj=" << tightened_obj << "\n";
}

void test_strong_branching_and_domain_reduction() {
    // Construct a model where branching down on a variable causes immediate LP infeasibility:
    // Min -2 x1 - 2 x2 - x3
    // s.t. x1 + x2 >= 1.5   (if x1 <= 0, x2 >= 1.5 > 1 => down branch infeasible!)
    //      x1 + x2 + 0.1 x3 <= 2.0
    // x1, x2, x3 in [0, 1] binary
    markov_cero::model::Model model;
    model.name = "STRONG_BRANCH_TEST";
    model.objective_sense = markov_cero::model::ObjectiveSense::minimize;
    model.objective = {-2.0, -2.0, -1.0};

    markov_cero::model::SparseMatrixBuilder builder(2, 3);
    // Row 0: x1 + x2 >= 1.5
    builder.add(0, 0, 1.0);
    builder.add(0, 1, 1.0);
    // Row 1: x1 + x2 + 0.1 x3 <= 2.0
    builder.add(1, 0, 1.0);
    builder.add(1, 1, 1.0);
    builder.add(1, 2, 0.1);
    model.matrix = builder.build();

    model.row_lower = {
        markov_cero::model::Bound::finite(1.5),
        markov_cero::model::Bound::negative_infinity()
    };
    model.row_upper = {
        markov_cero::model::Bound::positive_infinity(),
        markov_cero::model::Bound::finite(2.0)
    };
    model.row_name = {"ROW_LOWER", "ROW_UPPER"};

    model.variable_lower = {
        markov_cero::model::Bound::finite(0.0),
        markov_cero::model::Bound::finite(0.0),
        markov_cero::model::Bound::finite(0.0)
    };
    model.variable_upper = {
        markov_cero::model::Bound::finite(1.0),
        markov_cero::model::Bound::finite(1.0),
        markov_cero::model::Bound::finite(1.0)
    };
    model.variable_type = {
        markov_cero::model::VariableType::binary,
        markov_cero::model::VariableType::binary,
        markov_cero::model::VariableType::binary
    };
    model.variable_name = {"X1", "X2", "X3"};
    model.validate();

    // Solve root continuous LP
    const auto canon = markov_cero::transform::sparse_canonicalize(model, /*relax_integrality=*/true);
    const auto dense = canon.to_dense();
    const auto lpres = markov_cero::lp::reference::solve(dense);
    assert(lpres.status == markov_cero::lp::reference::SolveStatus::optimal);

    const auto primal = markov_cero::transform::reconstruct_primal(canon, lpres.primal);
    const double root_obj = markov_cero::transform::reconstruct_objective(canon, lpres.objective);
    const auto basis_state = markov_cero::lp::dual::make_basis_state(dense, lpres.basis);

    // Initial pseudo-cost tracker
    std::vector<markov_cero::milp::VariablePseudoCost> pseudo_costs(3);

    // Evaluate strong branching
    markov_cero::milp::StrongBranchingOptions sb_opts;
    sb_opts.max_lookahead_iterations = 30;
    sb_opts.score_mu = 0.16;
    sb_opts.update_pseudo_costs = true;

    const auto sb_res = markov_cero::milp::evaluate_strong_branching(
        model, primal, root_obj, basis_state, sb_opts, &pseudo_costs);

    // 1. Verify that candidates were evaluated
    assert(!sb_res.candidates.empty());

    // 2. Verify domain reduction on x1 or x2 (down branch x <= 0 is infeasible, so x >= 1)
    bool detected_infeasible_branch = false;
    for (const auto& cand : sb_res.candidates) {
        if (cand.variable_index == 0 || cand.variable_index == 1) {
            if (cand.is_down_infeasible) {
                detected_infeasible_branch = true;
            }
        }
    }
    assert(detected_infeasible_branch);

    // 3. Verify domain reductions list contains lower bound tightening
    assert(!sb_res.domain_reductions.empty());
    for (const auto& dr : sb_res.domain_reductions) {
        if (dr.variable_index == 0 || dr.variable_index == 1) {
            // Tightened lower bound should be 1.0!
            assert(dr.new_lower.is_finite() && dr.new_lower.value >= 1.0 - 1e-6);
            assert(dr.is_fixed); // Since upper was 1.0, variable is fixed to 1!
        }
    }

    // 4. Verify candidate score combination and ranking
    assert(sb_res.best_score > 0.0);

    // 5. Verify pseudo-costs tracker received updates
    bool has_pseudo_cost = false;
    for (const auto& pc : pseudo_costs) {
        if (pc.down_count > 0 || pc.up_count > 0) {
            has_pseudo_cost = true;
            break;
        }
    }
    assert(has_pseudo_cost);

    std::cout << "[+] test_strong_branching_and_domain_reduction passed (best_var="
              << sb_res.best_variable << ", score=" << sb_res.best_score
              << ", domain reductions=" << sb_res.domain_reductions.size() << ")\n";
}

void test_zero_trust_primal_verifier_integration() {
    // Test integration with markov_cero::verify::verify_primal
    markov_cero::model::Model model;
    model.name = "VERIFY_TEST";
    model.objective_sense = markov_cero::model::ObjectiveSense::minimize;
    model.objective = {1.0, 2.0};

    markov_cero::model::SparseMatrixBuilder builder(1, 2);
    builder.add(0, 0, 1.0);
    builder.add(0, 1, 1.0);
    model.matrix = builder.build();

    model.row_lower = {markov_cero::model::Bound::finite(1.0)};
    model.row_upper = {markov_cero::model::Bound::positive_infinity()};
    model.row_name = {"R1"};

    model.variable_lower = {
        markov_cero::model::Bound::finite(0.0),
        markov_cero::model::Bound::finite(0.0)
    };
    model.variable_upper = {
        markov_cero::model::Bound::finite(1.0),
        markov_cero::model::Bound::finite(1.0)
    };
    model.variable_type = {
        markov_cero::model::VariableType::binary,
        markov_cero::model::VariableType::binary
    };
    model.variable_name = {"X1", "X2"};
    model.validate();

    // Feasible candidate point: x1 = 1, x2 = 0, obj = 1.0
    markov_cero::verify::Candidate candidate;
    candidate.primal = {1.0, 0.0};
    candidate.claimed_objective = 1.0;

    const auto report = markov_cero::verify::verify_primal(model, candidate);
    assert(report.passed);
    assert(report.violations.empty());
    assert(report.maximum_integrality_violation < 1e-6);
    assert(std::abs(report.recomputed_objective - 1.0) < 1e-6);

    // Infeasible candidate point: x1 = 0, x2 = 0
    markov_cero::verify::Candidate inf_candidate;
    inf_candidate.primal = {0.0, 0.0};
    inf_candidate.claimed_objective = 0.0;

    const auto inf_report = markov_cero::verify::verify_primal(model, inf_candidate);
    assert(!inf_report.passed);
    assert(!inf_report.violations.empty());

    std::cout << "[+] test_zero_trust_primal_verifier_integration passed\n";
}

} // namespace

int main() {
    try {
        test_mir_mathematical_function();
        test_cut_efficacy_and_filtering();
        test_mir_cuts_tighten_relaxation_and_preserve_integers();
        test_strong_branching_and_domain_reduction();
        test_zero_trust_primal_verifier_integration();
        std::cout << "\n========================================\n";
        std::cout << "All Phase 4 Strong Branching & MIR tests PASSED!\n";
        std::cout << "========================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[-] Test failure: " << e.what() << "\n";
        return 1;
    }
}
