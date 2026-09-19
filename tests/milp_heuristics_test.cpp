#include "markov_cero/milp/heuristics.hpp"

#include <cassert>
#include <iostream>

namespace {

void test_simple_rounding() {
    markov_cero::model::Model model;
    model.name = "ROUNDING_TEST";
    model.objective_sense = markov_cero::model::ObjectiveSense::minimize;
    model.objective = {1.0, 2.0};

    markov_cero::model::SparseMatrixBuilder builder(1, 2);
    builder.add(0, 0, 1.0);
    builder.add(0, 1, 1.0);
    model.matrix = builder.build();

    model.row_lower = {markov_cero::model::Bound::finite(1.5)};
    model.row_upper = {markov_cero::model::Bound::positive_infinity()};
    model.row_name = {"R1"};

    model.variable_lower = {markov_cero::model::Bound::finite(0.0), markov_cero::model::Bound::finite(0.0)};
    model.variable_upper = {markov_cero::model::Bound::finite(5.0), markov_cero::model::Bound::finite(5.0)};
    model.variable_type = {markov_cero::model::VariableType::integer, markov_cero::model::VariableType::integer};
    model.variable_name = {"X1", "X2"};
    model.validate();

    // Fractional LP point: x1 = 1.5, x2 = 0 -> sum = 1.5
    std::vector<double> frac_primal = {1.5, 0.0};
    const auto res = markov_cero::milp::simple_rounding(model, frac_primal);
    assert(res.found);
    assert(markov_cero::milp::check_integer_feasibility(model, res.primal));
    std::cout << "[+] test_simple_rounding passed\n";
}

void test_feasibility_pump() {
    markov_cero::model::Model model;
    model.name = "PUMP_TEST";
    model.objective_sense = markov_cero::model::ObjectiveSense::minimize;
    model.objective = {-1.0, -1.0, -1.0};

    markov_cero::model::SparseMatrixBuilder builder(2, 3);
    // x1 + x2 + x3 <= 2
    builder.add(0, 0, 1.0);
    builder.add(0, 1, 1.0);
    builder.add(0, 2, 1.0);
    // 2 x1 + x2 <= 2
    builder.add(1, 0, 2.0);
    builder.add(1, 1, 1.0);
    model.matrix = builder.build();

    model.row_lower = {markov_cero::model::Bound::negative_infinity(), markov_cero::model::Bound::negative_infinity()};
    model.row_upper = {markov_cero::model::Bound::finite(2.0), markov_cero::model::Bound::finite(2.0)};
    model.row_name = {"R1", "R2"};

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

    std::vector<double> continuous = {0.6, 0.8, 0.6};
    const auto res = markov_cero::milp::feasibility_pump(model, continuous, 10);
    assert(res.found);
    assert(markov_cero::milp::check_integer_feasibility(model, res.primal));
    std::cout << "[+] test_feasibility_pump passed\n";
}

} // namespace

int main() {
    try {
        test_simple_rounding();
        test_feasibility_pump();
        std::cout << "All heuristics tests PASSED successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[-] Error: " << e.what() << "\n";
        return 1;
    }
}
