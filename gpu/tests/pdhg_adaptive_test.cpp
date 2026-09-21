#include "markov_cero/gpu/pdhg_step.hpp"
#include "markov_cero/io/mps.hpp"
#include "markov_cero/lp/first_order/pdlp.hpp"
#include "markov_cero/model/model.hpp"
#include "markov_cero/verify/primal_verifier.hpp"

#include <cmath>
#include <cstdlib>
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
    const char* src_dir = std::getenv("MARKOV_CERO_SOURCE_DIR");
    if (src_dir) {
        std::ifstream env_file(std::string(src_dir) + "/" + filepath);
        if (env_file.is_open()) {
            return markov_cero::io::parse_mps(env_file);
        }
    }
    throw std::runtime_error("Cannot open MPS file: " + filepath);
}

struct NetlibInstance {
    std::string name;
    std::string path;
    double expected_obj;
    double obj_tol;
    std::size_t max_iters;
};

void test_adaptive_convergence(const NetlibInstance& inst,
                               markov_cero::lp::first_order::Backend backend) {
    using namespace markov_cero::lp::first_order;

    auto model = load_mps(inst.path);

    // Default PDLP options enable Ruiz scaling, adaptive step sizes,
    // adaptive primal weight, and normalized duality gap restart
    PdlpOptions options;
    options.backend = backend;
    options.max_iterations = inst.max_iters;
    options.ruiz_scaling = true;
    options.adaptive_step_size = true;
    options.adaptive_primal_weight = true;
    options.restart_strategy = RestartStrategy::adaptive;
    options.primal_tolerance = 1e-4;
    options.dual_tolerance = 1e-4;
    options.gap_tolerance = 1e-4;

    const auto res = solve_pdlp(model, options);

    std::cout << "  [" << (backend == Backend::gpu ? "GPU" : "CPU") << "] "
              << std::left << std::setw(10) << inst.name
              << " status: " << (res.status == PdlpStatus::optimal ? "optimal" : "limit")
              << " iters: " << std::setw(6) << res.iterations
              << " obj: " << std::fixed << std::setprecision(4) << res.objective
              << " (expected: " << inst.expected_obj << ")\n";

    require(res.status == PdlpStatus::optimal,
            inst.name + " failed to converge to optimal on " +
            (backend == Backend::gpu ? "GPU" : "CPU"));

    const double rel_err = std::abs(res.objective - inst.expected_obj) /
                           std::max(1.0, std::abs(inst.expected_obj));
    require(rel_err <= inst.obj_tol,
            inst.name + " relative objective error " + std::to_string(rel_err) +
            " exceeds tolerance " + std::to_string(inst.obj_tol));

    // Verify primal feasibility
    markov_cero::verify::Candidate cand{res.primal, res.objective};
    const auto report = markov_cero::verify::verify_primal(
        model, cand, {1e-4, 1e-4}, {1e-4, 1e-4}, 1e-4);
    require(report.passed, inst.name + " failed primal verification on " +
            (backend == Backend::gpu ? "GPU" : "CPU"));
}

void test_adaptive_benefit(const std::string& path,
                           markov_cero::lp::first_order::Backend backend) {
    using namespace markov_cero::lp::first_order;

    auto model = load_mps(path);

    // Without adaptive primal weight (fixed omega = 1.0) and no restart
    PdlpOptions opt_fixed;
    opt_fixed.backend = backend;
    opt_fixed.max_iterations = 60000;
    opt_fixed.ruiz_scaling = true;
    opt_fixed.adaptive_step_size = false;
    opt_fixed.adaptive_primal_weight = false;
    opt_fixed.initial_primal_weight = 1.0;
    opt_fixed.restart_strategy = RestartStrategy::none;
    opt_fixed.primal_tolerance = 1e-4;
    opt_fixed.dual_tolerance = 1e-4;
    opt_fixed.gap_tolerance = 1e-4;

    // With adaptive step size and primal weight + restarts
    PdlpOptions opt_adaptive;
    opt_adaptive.backend = backend;
    opt_adaptive.max_iterations = 15000;
    opt_adaptive.ruiz_scaling = true;
    opt_adaptive.adaptive_step_size = true;
    opt_adaptive.adaptive_primal_weight = true;
    opt_adaptive.restart_strategy = RestartStrategy::adaptive;
    opt_adaptive.primal_tolerance = 1e-4;
    opt_adaptive.dual_tolerance = 1e-4;
    opt_adaptive.gap_tolerance = 1e-4;

    const auto res_fixed = solve_pdlp(model, opt_fixed);
    const auto res_adaptive = solve_pdlp(model, opt_adaptive);

    std::cout << "  [" << (backend == Backend::gpu ? "GPU" : "CPU")
              << " Adaptation Speedup] Adaptive: " << res_adaptive.iterations
              << " iters vs Fixed: " << res_fixed.iterations << " iters\n";

    require(res_adaptive.status == PdlpStatus::optimal,
            "Adaptive configuration failed to reach optimality");
    require(res_adaptive.iterations < res_fixed.iterations,
            "Adaptive features did not reduce iteration count compared to fixed");
}

} // namespace

int main() {
    std::cout << "=== Markov-Cero Adaptive PDHG Step & Primal Weight Tests (T-5.09) ===\n";
    using namespace markov_cero::lp::first_order;

    const std::vector<NetlibInstance> instances = {
        {"AFIRO", "data/netlib/afiro.mps", -464.7531428, 1e-3, 10000},
        {"BLEND", "data/netlib/blend.mps", -30.8121498, 1e-3, 15000},
        {"SC50A", "data/netlib/sc50a.mps", -64.5750771, 1e-3, 10000},
        {"SC50B", "data/netlib/sc50b.mps", -70.0000000, 1e-3, 10000},
        {"ADLITTLE", "data/netlib/adlittle.mps", 225494.96316, 1e-3, 25000},
        {"SCSD1", "data/netlib/scsd1.mps", 8.6666667, 1e-3, 10000},
    };

    std::cout << "\n--- 1. Testing CPU Backend Convergence (Without Manual Tuning) ---\n";
    for (const auto& inst : instances) {
        test_adaptive_convergence(inst, Backend::cpu);
    }

    std::cout << "\n--- 2. Testing GPU Pipeline Backend Convergence ---\n";
    for (const auto& inst : instances) {
        test_adaptive_convergence(inst, Backend::gpu);
    }

    std::cout << "\n--- 3. Testing Adaptation Benefit (Adaptive vs Fixed) ---\n";
    test_adaptive_benefit("data/netlib/sc50a.mps", Backend::cpu);
    test_adaptive_benefit("data/netlib/sc50a.mps", Backend::gpu);

    std::cout << "\n=== All Adaptive Step Size & Primal Weight Tests Passed ===\n";
    return 0;
}
