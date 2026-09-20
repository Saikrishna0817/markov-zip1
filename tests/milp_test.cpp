#include "markov_cero/milp/milp_solver.hpp"
#include "markov_cero/verify/primal_verifier.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

namespace {

void test_knapsack_01() {
    // max 10 x1 + 14 x2 + 12 x3
    // min -10 x1 - 14 x2 - 12 x3
    // s.t. 4 x1 + 6 x2 + 5 x3 <= 10
    // x1, x2, x3 in {0, 1}
    //
    // Continuous LP relaxation: x1=1, x3=1, x2=1/6, obj = -24.3333
    // Integer optimum: x1=1, x2=1, x3=0, obj = -24.0

    markov_cero::model::Model model;
    model.name = "KNAPSACK";
    model.objective_sense = markov_cero::model::ObjectiveSense::minimize;
    model.objective = {-10.0, -14.0, -12.0};
    model.objective_offset = 0.0;

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

    markov_cero::milp::Options options;
    const auto result = markov_cero::milp::solve(model, options);

    assert(result.status == markov_cero::lp::reference::SolveStatus::optimal);
    assert(std::abs(result.objective - (-24.0)) < 1e-5);
    assert(result.primal.size() == 3);
    assert(std::abs(result.primal[0] - 1.0) < 1e-5);
    assert(std::abs(result.primal[1] - 1.0) < 1e-5);
    assert(std::abs(result.primal[2] - 0.0) < 1e-5);

    // Verify independently with zero-trust primal verifier
    markov_cero::verify::Candidate candidate{result.primal, result.objective};
    const auto report = markov_cero::verify::verify_primal(model, candidate);
    assert(report.passed);
    assert(report.maximum_integrality_violation < 1e-6);
    std::cout << "[+] test_knapsack_01 passed\n";
}

void test_refinery_discrete_dispatch() {
    // MRPL Scenario: crude tanker discrete batches
    // min 50 x1 + 40 x2
    // s.t. x1 + x2 >= 3 (total batches needed >= 3)
    //      3 x1 + 2 x2 <= 8 (berth pipeline hours <= 8)
    // x1, x2 in {0, 1, 2, 3, ...}
    //
    // Valid integer points:
    // (0, 3): 3*0 + 2*3 = 6 <= 8, x1+x2 = 3 >= 3, cost = 50*0 + 40*3 = 120
    // (1, 2): 3*1 + 2*2 = 7 <= 8, x1+x2 = 3 >= 3, cost = 50*1 + 40*2 = 130
    // (2, 1): 3*2 + 2*1 = 8 <= 8, x1+x2 = 3 >= 3, cost = 50*2 + 40*1 = 140
    // Optimum is (0, 3) with cost 120

    markov_cero::model::Model model;
    model.name = "REFINERY_DISPATCH";
    model.objective_sense = markov_cero::model::ObjectiveSense::minimize;
    model.objective = {50.0, 40.0};
    model.objective_offset = 0.0;

    markov_cero::model::SparseMatrixBuilder builder(2, 2);
    // Row 0: x1 + x2 >= 3
    builder.add(0, 0, 1.0);
    builder.add(0, 1, 1.0);
    // Row 1: 3 x1 + 2 x2 <= 8
    builder.add(1, 0, 3.0);
    builder.add(1, 1, 2.0);
    model.matrix = builder.build();

    model.row_lower = {markov_cero::model::Bound::finite(3.0),
                       markov_cero::model::Bound::negative_infinity()};
    model.row_upper = {markov_cero::model::Bound::positive_infinity(),
                       markov_cero::model::Bound::finite(8.0)};
    model.row_name = {"MIN_BATCHES", "MAX_PIPELINE_HOURS"};

    model.variable_lower = {markov_cero::model::Bound::finite(0.0),
                            markov_cero::model::Bound::finite(0.0)};
    model.variable_upper = {markov_cero::model::Bound::finite(5.0),
                            markov_cero::model::Bound::finite(5.0)};
    model.variable_type = {markov_cero::model::VariableType::integer,
                           markov_cero::model::VariableType::integer};
    model.variable_name = {"TANKER_A", "TANKER_B"};
    model.validate();

    markov_cero::milp::Options options;
    const auto result = markov_cero::milp::solve(model, options);

    assert(result.status == markov_cero::lp::reference::SolveStatus::optimal);
    assert(std::abs(result.objective - 120.0) < 1e-5);
    assert(std::abs(result.primal[0] - 0.0) < 1e-5);
    assert(std::abs(result.primal[1] - 3.0) < 1e-5);

    markov_cero::verify::Candidate candidate{result.primal, result.objective};
    const auto report = markov_cero::verify::verify_primal(model, candidate);
    assert(report.passed);
    std::cout << "[+] test_refinery_discrete_dispatch passed\n";
}

void test_infeasible_milp() {
    // x1 + x2 <= 1
    // x1 + x2 >= 2
    // x1, x2 in {0, 1}
    markov_cero::model::Model model;
    model.name = "INFEASIBLE_MIP";
    model.objective = {1.0, 1.0};

    markov_cero::model::SparseMatrixBuilder builder(2, 2);
    builder.add(0, 0, 1.0);
    builder.add(0, 1, 1.0);
    builder.add(1, 0, 1.0);
    builder.add(1, 1, 1.0);
    model.matrix = builder.build();

    model.row_lower = {markov_cero::model::Bound::negative_infinity(),
                       markov_cero::model::Bound::finite(2.0)};
    model.row_upper = {markov_cero::model::Bound::finite(1.0),
                       markov_cero::model::Bound::positive_infinity()};
    model.row_name = {"R1", "R2"};

    model.variable_lower = {markov_cero::model::Bound::finite(0.0),
                            markov_cero::model::Bound::finite(0.0)};
    model.variable_upper = {markov_cero::model::Bound::finite(1.0),
                            markov_cero::model::Bound::finite(1.0)};
    model.variable_type = {markov_cero::model::VariableType::binary,
                           markov_cero::model::VariableType::binary};
    model.variable_name = {"X1", "X2"};
    model.validate();

    markov_cero::milp::Options options;
    const auto result = markov_cero::milp::solve(model, options);
    assert(result.status == markov_cero::lp::reference::SolveStatus::infeasible);
    std::cout << "[+] test_infeasible_milp passed\n";
}

} // namespace

int main() {
    try {
        test_knapsack_01();
        test_refinery_discrete_dispatch();
        test_infeasible_milp();
        std::cout << "All MILP unit tests PASSED successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[-] Error: " << e.what() << "\n";
        return 1;
    }
}
