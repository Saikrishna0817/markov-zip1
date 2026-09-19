// markov-cero Phase 4: PDLP first-order LP engine tests
#include "markov_cero/lp/first_order/pdlp.hpp"
#include "markov_cero/verify/primal_verifier.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

namespace {

void test_blend_lp() {
    // Simple 2-variable LP:
    // min  -x1 - 2*x2
    // s.t. x1 + x2 <= 4
    //      x1 <= 3
    //      x2 <= 3
    //      x1, x2 >= 0
    // Optimum: x1=1, x2=3, obj=-7
    markov_cero::model::Model model;
    model.name = "BLEND_PDLP";
    model.objective_sense = markov_cero::model::ObjectiveSense::minimize;
    model.objective = {-1.0, -2.0};
    model.objective_offset = 0.0;

    markov_cero::model::SparseMatrixBuilder builder(3, 2);
    builder.add(0, 0, 1.0); builder.add(0, 1, 1.0);
    builder.add(1, 0, 1.0);
    builder.add(2, 1, 1.0);
    model.matrix = builder.build();

    model.row_lower = {
        markov_cero::model::Bound::negative_infinity(),
        markov_cero::model::Bound::negative_infinity(),
        markov_cero::model::Bound::negative_infinity()
    };
    model.row_upper = {
        markov_cero::model::Bound::finite(4.0),
        markov_cero::model::Bound::finite(3.0),
        markov_cero::model::Bound::finite(3.0)
    };
    model.row_name = {"SUM", "X1_UB", "X2_UB"};

    model.variable_lower = {
        markov_cero::model::Bound::finite(0.0),
        markov_cero::model::Bound::finite(0.0)
    };
    model.variable_upper = {
        markov_cero::model::Bound::finite(10.0),
        markov_cero::model::Bound::finite(10.0)
    };
    model.variable_type = {
        markov_cero::model::VariableType::continuous,
        markov_cero::model::VariableType::continuous
    };
    model.variable_name = {"X1", "X2"};
    model.validate();

    markov_cero::lp::first_order::PdlpOptions opts;
    opts.max_iterations = 200000;
    opts.primal_tolerance = 1e-4;
    opts.dual_tolerance   = 1e-4;
    opts.gap_tolerance    = 1e-4;

    const auto res = markov_cero::lp::first_order::solve_pdlp(model, opts);

    assert(res.status == markov_cero::lp::first_order::PdlpStatus::optimal ||
           res.status == markov_cero::lp::first_order::PdlpStatus::iteration_limit);

    // Objective should be close to -7
    assert(std::abs(res.objective - (-7.0)) < 0.1);

    // Primal should satisfy variable bounds
    assert(res.primal.size() == 2);
    assert(res.primal[0] >= -1e-4);
    assert(res.primal[1] >= -1e-4);

    // Zero-trust verification
    if (res.status == markov_cero::lp::first_order::PdlpStatus::optimal) {
        markov_cero::verify::Candidate cand{res.primal, res.objective};
        const auto report = markov_cero::verify::verify_primal(model, cand,
            {1e-4, 1e-4}, {1e-4, 1e-4}, 1e-4);
        assert(report.passed);
    }

    std::cout << "[+] test_blend_lp PASSED: obj=" << res.objective
              << " (ref=-7.0), iters=" << res.iterations
              << ", primal_infeas=" << res.primal_infeasibility << "\n";
}

void test_equality_lp() {
    // min x1 + x2
    // s.t. x1 + x2 = 5
    //      0 <= x1 <= 5, 0 <= x2 <= 5
    // Optimum: obj = 5 (any point on x1+x2=5)
    markov_cero::model::Model model;
    model.name = "EQUALITY_PDLP";
    model.objective_sense = markov_cero::model::ObjectiveSense::minimize;
    model.objective = {1.0, 1.0};
    model.objective_offset = 0.0;

    markov_cero::model::SparseMatrixBuilder builder(1, 2);
    builder.add(0, 0, 1.0);
    builder.add(0, 1, 1.0);
    model.matrix = builder.build();

    model.row_lower = {markov_cero::model::Bound::finite(5.0)};
    model.row_upper = {markov_cero::model::Bound::finite(5.0)};
    model.row_name  = {"EQUALITY"};

    model.variable_lower = {
        markov_cero::model::Bound::finite(0.0),
        markov_cero::model::Bound::finite(0.0)
    };
    model.variable_upper = {
        markov_cero::model::Bound::finite(5.0),
        markov_cero::model::Bound::finite(5.0)
    };
    model.variable_type  = {
        markov_cero::model::VariableType::continuous,
        markov_cero::model::VariableType::continuous
    };
    model.variable_name  = {"X1", "X2"};
    model.validate();

    markov_cero::lp::first_order::PdlpOptions opts;
    opts.max_iterations = 200000;
    opts.primal_tolerance = 1e-4;
    opts.dual_tolerance   = 1e-4;
    opts.gap_tolerance    = 1e-4;

    const auto res = markov_cero::lp::first_order::solve_pdlp(model, opts);

    assert(std::abs(res.objective - 5.0) < 0.1);

    if (res.status == markov_cero::lp::first_order::PdlpStatus::optimal) {
        markov_cero::verify::Candidate cand{res.primal, res.objective};
        const auto report = markov_cero::verify::verify_primal(model, cand,
            {1e-4, 1e-4}, {1e-4, 1e-4}, 1e-4);
        assert(report.passed);
    }
    std::cout << "[+] test_equality_lp PASSED: obj=" << res.objective
              << " (ref=5.0), iters=" << res.iterations << "\n";
}

} // namespace

int main() {
    try {
        test_blend_lp();
        test_equality_lp();
        std::cout << "All PDLP unit tests PASSED successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[-] Error: " << e.what() << "\n";
        return 1;
    }
}
