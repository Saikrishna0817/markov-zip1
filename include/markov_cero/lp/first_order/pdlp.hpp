#pragma once

// markov-cero: sovereign first-order LP engine
// Primal-Dual Hybrid Gradient (PDHG / PDLP)
// Grounding: Chambolle & Pock (2011); Applegate et al. (2021)

#include "markov_cero/model/model.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace markov_cero::lp::first_order {

enum class Backend { cpu, gpu };
enum class RestartStrategy { none, adaptive, fixed };

struct PdlpOptions {
    std::size_t max_iterations{100000};
    std::size_t restart_every{40};
    double primal_tolerance{1e-4};
    double dual_tolerance{1e-4};
    double gap_tolerance{1e-4};
    double step_size_reduction{0.9};
    std::size_t power_iterations{20};
    Backend backend{Backend::cpu};
    RestartStrategy restart_strategy{RestartStrategy::adaptive};
    double restart_reduction_factor{0.368};
};

enum class PdlpStatus { optimal, iteration_limit, infeasible_or_unbounded, numerical_failure };

struct PdlpResult {
    PdlpStatus status{PdlpStatus::iteration_limit};
    std::vector<double> primal;
    std::vector<double> dual;
    double objective{0.0};
    double primal_infeasibility{0.0};
    double dual_infeasibility{0.0};
    double duality_gap{0.0};
    std::size_t iterations{0};
    std::string message;
};

// Solve an LP using matrix-free Primal-Dual Hybrid Gradient.
// Uses only SpMV: A*x and A^T*y -- never factorizes a basis matrix.
[[nodiscard]] PdlpResult solve_pdlp(const model::Model& model, const PdlpOptions& options = {});

} // namespace markov_cero::lp::first_order
