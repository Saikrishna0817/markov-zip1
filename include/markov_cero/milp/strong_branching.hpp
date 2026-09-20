#pragma once

#include "markov_cero/lp/dual/dual_simplex.hpp"
#include "markov_cero/milp/branch_selector.hpp"
#include "markov_cero/model/model.hpp"

#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace markov_cero::milp {

struct StrongBranchingCandidate {
    std::size_t variable_index{0};
    double down_degradation{0.0};
    double up_degradation{0.0};
    double score{0.0};
    bool is_down_infeasible{false};
    bool is_up_infeasible{false};
};

struct DomainReduction {
    std::size_t variable_index{0};
    model::Bound new_lower{model::Bound::negative_infinity()};
    model::Bound new_upper{model::Bound::positive_infinity()};
    bool is_fixed{false};
};

struct StrongBranchingOptions {
    std::size_t max_lookahead_iterations{30};
    std::size_t max_candidates{0}; // 0 = all fractional variables
    double score_mu{0.16};         // Achterberg (2005) score combination weight
    double integrality_tolerance{1e-6};
    double feasibility_tolerance{1e-7};
    bool update_pseudo_costs{true};
};

struct StrongBranchingResult {
    std::size_t best_variable{0};
    double best_score{0.0};
    std::vector<StrongBranchingCandidate> candidates;
    std::vector<DomainReduction> domain_reductions;
    bool subproblem_infeasible{false};
};

[[nodiscard]] double compute_strong_branching_score(double delta_down, double delta_up,
                                                    double mu = 0.16) noexcept;

[[nodiscard]] StrongBranchingResult evaluate_strong_branching(
    const model::Model& model, const std::vector<double>& primal, double current_obj,
    const std::optional<lp::dual::BasisState>& current_basis,
    const std::vector<std::size_t>& candidate_indices, const StrongBranchingOptions& options = {},
    std::vector<VariablePseudoCost>* pseudo_costs = nullptr);

[[nodiscard]] StrongBranchingResult
evaluate_strong_branching(const model::Model& model, const std::vector<double>& primal,
                          double current_obj,
                          const std::optional<lp::dual::BasisState>& current_basis,
                          const StrongBranchingOptions& options = {},
                          std::vector<VariablePseudoCost>* pseudo_costs = nullptr);

} // namespace markov_cero::milp
