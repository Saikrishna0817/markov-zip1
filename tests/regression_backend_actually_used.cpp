// Regression: GPU-path JSON must report backend_actually_used, never pretend CUDA ran
// when only the host fallback executed (Phase 2 remediation).
#include "markov_cero/gpu/device.hpp"
#include "markov_cero/io/mps.hpp"
#include "markov_cero/lp/first_order/pdlp.hpp"

#include <iostream>
#include <sstream>
#include <string>

int main() {
    const char* mps = R"(NAME tiny
ROWS
 N  COST
 L  R1
COLUMNS
    X1        COST               1.0
    X1        R1                 1.0
RHS
    RHS1      R1                 1.0
BOUNDS
 UP BND1      X1                 1.0
ENDATA
)";
    std::istringstream in(mps);
    auto model = markov_cero::io::parse_mps(in);

    markov_cero::lp::first_order::PdlpOptions cpu_opts;
    cpu_opts.backend = markov_cero::lp::first_order::Backend::cpu;
    cpu_opts.max_iterations = 5000;
    auto cpu = markov_cero::lp::first_order::solve_pdlp(model, cpu_opts);
    if (cpu.backend_actually_used != "cpu") {
        std::cerr << "FAIL: CPU path reported backend_actually_used="
                  << cpu.backend_actually_used << "\n";
        return 1;
    }

    markov_cero::lp::first_order::PdlpOptions gpu_opts = cpu_opts;
    gpu_opts.backend = markov_cero::lp::first_order::Backend::gpu;
    auto gpu = markov_cero::lp::first_order::solve_pdlp(model, gpu_opts);
    const bool cuda = markov_cero::gpu::is_gpu_available();
    const std::string expected = cuda ? "cuda" : "cpu_fallback";
    if (gpu.backend_actually_used != expected) {
        std::cerr << "FAIL: GPU request reported backend_actually_used="
                  << gpu.backend_actually_used << " expected " << expected << "\n";
        return 1;
    }
    if (!cuda && gpu.backend_actually_used == "cuda") {
        std::cerr << "FAIL: claimed cuda without device\n";
        return 1;
    }
    std::cout << "PASS: cpu=" << cpu.backend_actually_used
              << " gpu_request=" << gpu.backend_actually_used
              << " is_gpu_available=" << (cuda ? "true" : "false") << "\n";
    return 0;
}
