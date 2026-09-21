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

struct ToleranceRun {
    double tolerance;
    std::size_t max_iterations;
};

void test_instance_kkt_tolerances(const std::string& name,
                                  const std::string& path,
                                  double expected_obj,
                                  markov_cero::lp::first_order::Backend backend) {
    using namespace markov_cero::lp::first_order;

    auto model = load_mps(path);
    const std::vector<ToleranceRun> runs = {
        {1e-4, 15000},
        {1e-6, 30000},
        {1e-8, 50000}
    };

    std::size_t prev_iters = 0;

    for (const auto& run : runs) {
        PdlpOptions opts;
        opts.backend = backend;
        opts.max_iterations = run.max_iterations;
        opts.set_tolerance(run.tolerance);

        const auto res = solve_pdlp(model, opts);

        std::cout << "  [" << (backend == Backend::gpu ? "GPU" : "CPU") << "] "
                  << std::left << std::setw(8) << name
                  << " tol=" << std::setw(8) << run.tolerance
                  << " iters=" << std::setw(6) << res.iterations
                  << " prim_inf=" << std::setw(11) << std::scientific << std::setprecision(2)
                  << res.primal_infeasibility
                  << " dual_inf=" << std::setw(11) << res.dual_infeasibility
                  << " gap=" << std::setw(11) << res.duality_gap
                  << " obj=" << std::fixed << std::setprecision(6) << res.objective << "\n";

        require(res.status == PdlpStatus::optimal,
                name + " failed to reach optimal at tol=" + std::to_string(run.tolerance));
        require(res.primal_infeasibility <= run.tolerance * 1.01,
                name + " primal infeasibility exceeds tolerance");
        require(res.dual_infeasibility <= run.tolerance * 1.01,
                name + " dual infeasibility exceeds tolerance");
        require(res.duality_gap <= run.tolerance * 1.01,
                name + " duality gap exceeds tolerance");
        require(res.tolerance == run.tolerance,
                name + " result tolerance does not match requested tolerance");

        // Independent zero-trust verifier
        markov_cero::verify::Candidate cand{res.primal, res.objective};
        const markov_cero::verify::Tolerance vtol{run.tolerance, run.tolerance};
        const auto report = markov_cero::verify::verify_primal(
            model, cand, vtol, vtol, run.tolerance);
        require(report.passed,
                name + " independent primal verification failed at tol=" +
                std::to_string(run.tolerance));

        // Objective error
        const double obj_err = std::abs(res.objective - expected_obj) /
                               std::max(1.0, std::abs(expected_obj));
        require(obj_err <= 0.01,
                name + " objective error too high: " + std::to_string(obj_err));

        // Monotonicity / reasonable progression of iterations
        if (prev_iters > 0) {
            require(res.iterations >= prev_iters / 2,
                    "unexpected iteration drop across tighter tolerances");
        }
        prev_iters = res.iterations;
    }
}

} // namespace

int main() {
    std::cout << "=== Markov-Cero Relative KKT Termination Tests (T-5.10 / D-GPU-09) ===\n";
    std::cout << "Tolerances tested: 1e-4 (fast), 1e-6 (medium), 1e-8 (high precision)\n\n";

    using namespace markov_cero::lp::first_order;

    std::cout << "--- 1. CPU Backend Multi-Tolerance Verification ---\n";
    test_instance_kkt_tolerances("AFIRO", "data/netlib/afiro.mps", -464.7531428, Backend::cpu);
    test_instance_kkt_tolerances("BLEND", "data/netlib/blend.mps", -30.8121498, Backend::cpu);

    std::cout << "\n--- 2. GPU Pipeline Backend Multi-Tolerance Verification ---\n";
    test_instance_kkt_tolerances("AFIRO", "data/netlib/afiro.mps", -464.7531428, Backend::gpu);
    test_instance_kkt_tolerances("BLEND", "data/netlib/blend.mps", -30.8121498, Backend::gpu);

    std::cout << "\n=== All Relative KKT Termination Tests Passed at 1e-4 / 1e-6 / 1e-8 ===\n";
    return 0;
}
