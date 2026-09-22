// Regression: GPU request must not claim CUDA when GPU code is not linked.
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
    if (gpu.backend_actually_used != "cpu_fallback") {
        std::cerr << "FAIL: GPU request reported backend_actually_used="
                  << gpu.backend_actually_used << " expected cpu_fallback\n";
        return 1;
    }
    std::cout << "PASS: cpu=" << cpu.backend_actually_used
              << " gpu_request=" << gpu.backend_actually_used
              << " (GPU deferred; no CUDA in build)\n";
    return 0;
}
