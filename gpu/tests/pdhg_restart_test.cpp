#include "markov_cero/gpu/pdhg_step.hpp"
#include "markov_cero/io/mps.hpp"
#include "markov_cero/lp/first_order/pdlp.hpp"
#include "markov_cero/model/model.hpp"
#include "markov_cero/verify/primal_verifier.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("Assertion failed: " + message);
    }
}

markov_cero::model::Model load_mps(const std::string& filepath) {
    std::ifstream file(filepath);
    if (file.is_open()) {
        return markov_cero::io::parse_mps(file);
    }
    std::ifstream alt_file("../" + filepath);
    if (alt_file.is_open()) {
        return markov_cero::io::parse_mps(alt_file);
    }
    throw std::runtime_error("Cannot open MPS file: " + filepath);
}

markov_cero::model::Model make_blend_model() {
    using namespace markov_cero::model;
    Model model;
    model.name = "BLEND_LP";
    model.objective_sense = ObjectiveSense::minimize;
    model.objective = {-1.0, -2.0};
    model.objective_offset = 0.0;

    SparseMatrixBuilder builder(3, 2);
    builder.add(0, 0, 1.0);
    builder.add(0, 1, 1.0);
    builder.add(1, 0, 1.0);
    builder.add(2, 1, 1.0);
    model.matrix = builder.build();

    model.row_lower = {Bound::negative_infinity(),
                       Bound::negative_infinity(),
                       Bound::negative_infinity()};
    model.row_upper = {Bound::finite(4.0),
                       Bound::finite(3.0),
                       Bound::finite(3.0)};
    model.row_name = {"SUM", "X1_UB", "X2_UB"};
    model.variable_lower = {Bound::finite(0.0), Bound::finite(0.0)};
    model.variable_upper = {Bound::finite(10.0), Bound::finite(10.0)};
    model.variable_type = {VariableType::continuous, VariableType::continuous};
    model.variable_name = {"X1", "X2"};
    model.validate();
    return model;
}

markov_cero::model::Model make_equality_model() {
    using namespace markov_cero::model;
    Model model;
    model.name = "EQUALITY_LP";
    model.objective_sense = ObjectiveSense::minimize;
    model.objective = {1.0, 1.0};
    model.objective_offset = 0.0;

    SparseMatrixBuilder builder(1, 2);
    builder.add(0, 0, 1.0);
    builder.add(0, 1, 1.0);
    model.matrix = builder.build();

    model.row_lower = {Bound::finite(5.0)};
    model.row_upper = {Bound::finite(5.0)};
    model.row_name = {"EQ"};
    model.variable_lower = {Bound::finite(0.0), Bound::finite(0.0)};
    model.variable_upper = {Bound::finite(5.0), Bound::finite(5.0)};
    model.variable_type = {VariableType::continuous, VariableType::continuous};
    model.variable_name = {"X1", "X2"};
    model.validate();
    return model;
}

void test_instance_restart_reduction(const std::string& name,
                                     const markov_cero::model::Model& model,
                                     std::size_t max_test_iters,
                                     markov_cero::lp::first_order::Backend backend) {
    using namespace markov_cero::lp::first_order;

    PdlpOptions opt_none;
    opt_none.max_iterations = max_test_iters;
    opt_none.backend = backend;
    opt_none.restart_strategy = RestartStrategy::none;
    opt_none.primal_tolerance = 1e-4;
    opt_none.dual_tolerance = 1e-4;
    opt_none.gap_tolerance = 1e-4;

    PdlpOptions opt_adaptive;
    opt_adaptive.max_iterations = max_test_iters;
    opt_adaptive.backend = backend;
    opt_adaptive.restart_strategy = RestartStrategy::adaptive;
    opt_adaptive.primal_tolerance = 1e-4;
    opt_adaptive.dual_tolerance = 1e-4;
    opt_adaptive.gap_tolerance = 1e-4;

    const auto res_none = solve_pdlp(model, opt_none);
    const auto res_adaptive = solve_pdlp(model, opt_adaptive);

    std::cout << "  " << std::left << std::setw(12) << name
              << " [" << (backend == Backend::gpu ? "GPU" : "CPU") << "] "
              << "adaptive: " << std::setw(5) << res_adaptive.iterations << " iters ("
              << (res_adaptive.status == PdlpStatus::optimal ? "optimal" : "limit") << ") "
              << "vs no-restart: " << std::setw(5) << res_none.iterations << " iters ("
              << (res_none.status == PdlpStatus::optimal ? "optimal" : "limit") << ")\n";

    require(res_adaptive.status == PdlpStatus::optimal,
            name + " with adaptive restart failed to converge");
    require(res_adaptive.iterations < res_none.iterations,
            name + " adaptive restart iteration count not measurably lower: adaptive=" +
            std::to_string(res_adaptive.iterations) + " vs none=" +
            std::to_string(res_none.iterations));

    // Solution validity
    markov_cero::verify::Candidate cand{res_adaptive.primal, res_adaptive.objective};
    const auto report = markov_cero::verify::verify_primal(
        model, cand, {1e-4, 1e-4}, {1e-4, 1e-4}, 1e-4);
    require(report.passed, name + " solution failed primal verification");
}

} // namespace

int main() {
    std::cout << "=== Markov-Cero Adaptive Restart Tests (T-5.08) ===\n";
    using namespace markov_cero::lp::first_order;

    auto blend_mdl = make_blend_model();
    auto eq_mdl = make_equality_model();
    auto refinery_mdl = load_mps("examples/refinery/refinery-feasible.mps");
    auto afiro_mdl = load_mps("data/netlib/afiro.mps");

    std::cout << "\n--- 1. Testing CPU Backend ---\n";
    test_instance_restart_reduction("BLEND", blend_mdl, 10000, Backend::cpu);
    test_instance_restart_reduction("EQUALITY", eq_mdl, 10000, Backend::cpu);
    test_instance_restart_reduction("REFINERY", refinery_mdl, 10000, Backend::cpu);
    test_instance_restart_reduction("AFIRO", afiro_mdl, 15000, Backend::cpu);

    std::cout << "\n--- 2. Testing GPU Pipeline Backend ---\n";
    test_instance_restart_reduction("BLEND", blend_mdl, 10000, Backend::gpu);
    test_instance_restart_reduction("EQUALITY", eq_mdl, 10000, Backend::gpu);
    test_instance_restart_reduction("REFINERY", refinery_mdl, 10000, Backend::gpu);
    test_instance_restart_reduction("AFIRO", afiro_mdl, 15000, Backend::gpu);

    std::cout << "\n=== All Adaptive Restart Tests Passed (Measurable Reductions Verified) ===\n";
    return 0;
}
