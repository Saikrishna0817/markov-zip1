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

void test_gpu_timing(const std::string& name, const std::string& path, double expected_obj) {
    using namespace markov_cero::lp::first_order;

    auto model = load_mps(path);
    PdlpOptions opts;
    opts.backend = Backend::gpu;
    opts.max_iterations = 20000;
    opts.set_tolerance(1e-4);

    const auto res = solve_pdlp(model, opts);
    require(res.status == PdlpStatus::optimal, name + " GPU solve did not converge");
    require(std::abs(res.objective - expected_obj) < 1e-1,
            name + " GPU objective mismatch: " + std::to_string(res.objective));

    // Directive D-GPU-08: Four-part timing is mandatory
    require(res.h2d_ms > 0.0, name + " h2d_ms must be positive: " + std::to_string(res.h2d_ms));
    require(res.kernel_ms > 0.0,
            name + " kernel_ms must be positive: " + std::to_string(res.kernel_ms));
    require(res.d2h_ms > 0.0, name + " d2h_ms must be positive: " + std::to_string(res.d2h_ms));
    require(res.total_ms > 0.0,
            name + " total_ms must be positive: " + std::to_string(res.total_ms));

    // Sanity check: individual transfer/kernel components should not wildly exceed total time
    const double component_sum = res.h2d_ms + res.kernel_ms + res.d2h_ms;
    require(component_sum <= res.total_ms * 1.25 + 5.0,
            name + " timing inconsistency: sum=" + std::to_string(component_sum) +
                " total=" + std::to_string(res.total_ms));

    // Zero-trust verification
    markov_cero::verify::Candidate cand{res.primal, res.objective};
    const markov_cero::verify::Tolerance tol{1e-4, 1e-4};
    const auto report = markov_cero::verify::verify_primal(model, cand, tol, tol, 1e-4);
    require(report.passed, name + " failed independent primal verification");

    std::cout << "  [GPU " << std::left << std::setw(8) << name << "]"
              << std::fixed << std::setprecision(3)
              << " H2D: " << std::setw(6) << res.h2d_ms << "ms"
              << "  Kernel: " << std::setw(6) << res.kernel_ms << "ms"
              << "  D2H: " << std::setw(6) << res.d2h_ms << "ms"
              << "  Total: " << std::setw(6) << res.total_ms << "ms"
              << "  Iters: " << res.iterations << "\n";
}

void test_cpu_timing(const std::string& name, const std::string& path, double expected_obj) {
    using namespace markov_cero::lp::first_order;

    auto model = load_mps(path);
    PdlpOptions opts;
    opts.backend = Backend::cpu;
    opts.max_iterations = 20000;
    opts.set_tolerance(1e-4);

    const auto res = solve_pdlp(model, opts);
    require(res.status == PdlpStatus::optimal, name + " CPU solve did not converge");
    require(std::abs(res.objective - expected_obj) < 1e-1,
            name + " CPU objective mismatch: " + std::to_string(res.objective));

    // CPU execution reports zero transfer times and positive compute/total
    require(res.h2d_ms == 0.0, name + " CPU h2d_ms must be 0: " + std::to_string(res.h2d_ms));
    require(res.d2h_ms == 0.0, name + " CPU d2h_ms must be 0: " + std::to_string(res.d2h_ms));
    require(res.kernel_ms > 0.0,
            name + " CPU kernel_ms must be positive: " + std::to_string(res.kernel_ms));
    require(res.total_ms > 0.0,
            name + " CPU total_ms must be positive: " + std::to_string(res.total_ms));

    std::cout << "  [CPU " << std::left << std::setw(8) << name << "]"
              << std::fixed << std::setprecision(3)
              << " H2D: " << std::setw(6) << res.h2d_ms << "ms"
              << "  Kernel: " << std::setw(6) << res.kernel_ms << "ms"
              << "  D2H: " << std::setw(6) << res.d2h_ms << "ms"
              << "  Total: " << std::setw(6) << res.total_ms << "ms"
              << "  Iters: " << res.iterations << "\n";
}

} // namespace

int main() {
    try {
        std::cout << "=== Markov-Cero Four-Part Timing Tests (T-5.11 / D-GPU-08) ===\n\n";

        std::cout << "--- Testing GPU Backend Four-Part Timing Breakdown ---\n";
        test_gpu_timing("AFIRO", "data/netlib/afiro.mps", -464.7531);
        test_gpu_timing("BLEND", "data/netlib/blend.mps", -30.8121);
        test_gpu_timing("SC50A", "data/netlib/sc50a.mps", -64.5751);

        std::cout << "\n--- Testing CPU Backend Timing Telemetry ---\n";
        test_cpu_timing("AFIRO", "data/netlib/afiro.mps", -464.7531);
        test_cpu_timing("BLEND", "data/netlib/blend.mps", -30.8121);

        std::cout << "\n=== All Four-Part Timing Tests Passed ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
