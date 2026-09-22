// Regression: guards against pathological primal simplex slowdown past ~200 rows
// caused by allocating a full dense column vector for every nonbasic on every
// pricing pass (Phase 2 remediation). scale_200 is ~126 rows / 224 cols.
#include "markov_cero/io/mps.hpp"
#include "markov_cero/lp/reference/revised_simplex.hpp"
#include "markov_cero/transform/canonicalize.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static std::string find_mps() {
    const char* env = std::getenv("MARKOV_CERO_SCALE200_MPS");
    if (env && fs::exists(env)) return env;
    const char* candidates[] = {
        "data/scale_study/scale_200.mps",
        "tests/fixtures/scale_200.mps",
        "/tmp/scale_mps/scale_200.mps",
    };
    for (const char* c : candidates) {
        if (fs::exists(c)) return c;
    }
    return {};
}

int main() {
    const std::string path = find_mps();
    if (path.empty()) {
        std::cerr << "SKIP: scale_200.mps not found (set MARKOV_CERO_SCALE200_MPS)\n";
        return 0;
    }
    std::ifstream in(path);
    if (!in) {
        std::cerr << "FAIL: cannot open " << path << "\n";
        return 1;
    }
    markov_cero::model::Model model;
    try {
        model = markov_cero::io::parse_mps(in);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: parse " << path << ": " << e.what() << "\n";
        return 1;
    }
    auto canon = markov_cero::transform::canonicalize(model);
    const auto t0 = std::chrono::steady_clock::now();
    auto result = markov_cero::lp::reference::solve(canon);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    if (result.status != markov_cero::lp::reference::SolveStatus::optimal) {
        std::cerr << "FAIL: expected optimal, got "
                  << markov_cero::lp::reference::to_string(result.status)
                  << " msg=" << result.message << "\n";
        return 1;
    }
    // Guard: previously multi-second on this size due to pricing allocations.
    constexpr double wall_limit_ms = 15000.0;
    if (ms > wall_limit_ms) {
        std::cerr << "FAIL: scale_200 primal wall " << ms << " ms exceeds " << wall_limit_ms
                  << " ms (pricing regression)\n";
        return 1;
    }
    std::cout << "PASS: scale_200 optimal in " << ms << " ms iterations="
              << (result.phase_one_iterations + result.phase_two_iterations) << "\n";
    return 0;
}
