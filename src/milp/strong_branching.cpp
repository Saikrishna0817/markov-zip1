#include "markov_cero/milp/strong_branching.hpp"

#include "markov_cero/lp/dual/dual_simplex.hpp"
#include "markov_cero/transform/sparse_canonical_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace markov_cero::milp {

double compute_strong_branching_score(
    double delta_down,
    double delta_up,
    double mu) noexcept {
    const double d_down = std::max(0.0, delta_down);
    const double d_up = std::max(0.0, delta_up);
    return (1.0 - mu) * std::min(d_down, d_up) + mu * std::max(d_down, d_up);
}

StrongBranchingResult evaluate_strong_branching(
    const model::Model& model,
    const std::vector<double>& primal,
    double current_obj,
    const std::optional<lp::dual::BasisState>& current_basis,
    const std::vector<std::size_t>& candidate_indices,
    const StrongBranchingOptions& options,
    std::vector<VariablePseudoCost>* pseudo_costs) {
    StrongBranchingResult result;

    std::vector<std::size_t> eval_indices;
    if (candidate_indices.empty()) {
        eval_indices = find_fractional_variables(primal, model.variable_type, options.integrality_tolerance);
    } else {
        eval_indices = candidate_indices;
    }

    if (eval_indices.empty()) {
        return result;
    }

    // Sort candidates by fractionality if max_candidates is set
    if (options.max_candidates > 0 && eval_indices.size() > options.max_candidates) {
        std::stable_sort(eval_indices.begin(), eval_indices.end(), [&](std::size_t a, std::size_t b) {
            const double frac_a = primal[a] - std::floor(primal[a]);
            const double frac_b = primal[b] - std::floor(primal[b]);
            const double dist_a = std::min(frac_a, 1.0 - frac_a);
            const double dist_b = std::min(frac_b, 1.0 - frac_b);
            return dist_a > dist_b; // Closest to 0.5 first
        });
        eval_indices.resize(options.max_candidates);
    }

    result.best_variable = eval_indices[0];
    result.best_score = -1.0;

    constexpr double infeasible_delta = 1e8;

    for (std::size_t var_idx : eval_indices) {
        if (var_idx >= model.matrix.column_count) {
            continue;
        }

        const double x_val = primal[var_idx];
        const double x_floor = std::floor(x_val);
        const double x_ceil = std::ceil(x_val);
        const double frac = x_val - x_floor;

        StrongBranchingCandidate cand;
        cand.variable_index = var_idx;

        // 1. Tentative Down Branch: x_j <= x_floor
        if (model.variable_lower[var_idx].is_finite() &&
            x_floor < model.variable_lower[var_idx].value - 1e-9) {
            cand.is_down_infeasible = true;
            cand.down_degradation = std::numeric_limits<double>::infinity();
        } else {
            model::Model down_model = model;
            down_model.variable_upper[var_idx] = model::Bound::finite(x_floor);

            try {
                const auto canon_down = transform::sparse_canonicalize(down_model, /*relax_integrality=*/true);
                const auto dense_down = canon_down.to_dense();

                lp::dual::Options dopts;
                dopts.iteration_limit = options.max_lookahead_iterations;
                dopts.feasibility_tolerance = options.feasibility_tolerance;
                dopts.allow_cold_fallback = true;

                std::optional<lp::dual::BasisState> child_warm;
                if (current_basis.has_value() &&
                    current_basis->rows == dense_down.matrix.rows &&
                    current_basis->columns == dense_down.matrix.columns) {
                    try {
                        child_warm = lp::dual::make_basis_state(dense_down, current_basis->basic_variables);
                    } catch (...) {
                        child_warm = std::nullopt;
                    }
                }

                const auto dres = lp::dual::solve(dense_down, dopts, child_warm);
                if (dres.solution.status == lp::reference::SolveStatus::infeasible) {
                    cand.is_down_infeasible = true;
                    cand.down_degradation = std::numeric_limits<double>::infinity();
                } else if (dres.solution.status == lp::reference::SolveStatus::optimal) {
                    const double child_obj = transform::reconstruct_objective(canon_down, dres.solution.objective);
                    cand.down_degradation = std::max(0.0, child_obj - current_obj);
                    cand.is_down_infeasible = false;
                } else if (dres.solution.status == lp::reference::SolveStatus::iteration_limit) {
                    cand.is_down_infeasible = false;
                    if (!dres.telemetry.empty()) {
                        const double child_obj = transform::reconstruct_objective(canon_down, dres.telemetry.back().objective);
                        cand.down_degradation = std::max(0.0, child_obj - current_obj);
                    } else {
                        cand.down_degradation = 0.0;
                    }
                } else {
                    cand.down_degradation = 0.0;
                    cand.is_down_infeasible = false;
                }
            } catch (...) {
                cand.down_degradation = 0.0;
                cand.is_down_infeasible = false;
            }
        }

        // 2. Tentative Up Branch: x_j >= x_ceil
        if (model.variable_upper[var_idx].is_finite() &&
            x_ceil > model.variable_upper[var_idx].value + 1e-9) {
            cand.is_up_infeasible = true;
            cand.up_degradation = std::numeric_limits<double>::infinity();
        } else {
            model::Model up_model = model;
            up_model.variable_lower[var_idx] = model::Bound::finite(x_ceil);

            try {
                const auto canon_up = transform::sparse_canonicalize(up_model, /*relax_integrality=*/true);
                const auto dense_up = canon_up.to_dense();

                lp::dual::Options dopts;
                dopts.iteration_limit = options.max_lookahead_iterations;
                dopts.feasibility_tolerance = options.feasibility_tolerance;
                dopts.allow_cold_fallback = true;

                std::optional<lp::dual::BasisState> child_warm;
                if (current_basis.has_value() &&
                    current_basis->rows == dense_up.matrix.rows &&
                    current_basis->columns == dense_up.matrix.columns) {
                    try {
                        child_warm = lp::dual::make_basis_state(dense_up, current_basis->basic_variables);
                    } catch (...) {
                        child_warm = std::nullopt;
                    }
                }

                const auto dres = lp::dual::solve(dense_up, dopts, child_warm);
                if (dres.solution.status == lp::reference::SolveStatus::infeasible) {
                    cand.is_up_infeasible = true;
                    cand.up_degradation = std::numeric_limits<double>::infinity();
                } else if (dres.solution.status == lp::reference::SolveStatus::optimal) {
                    const double child_obj = transform::reconstruct_objective(canon_up, dres.solution.objective);
                    cand.up_degradation = std::max(0.0, child_obj - current_obj);
                    cand.is_up_infeasible = false;
                } else if (dres.solution.status == lp::reference::SolveStatus::iteration_limit) {
                    cand.is_up_infeasible = false;
                    if (!dres.telemetry.empty()) {
                        const double child_obj = transform::reconstruct_objective(canon_up, dres.telemetry.back().objective);
                        cand.up_degradation = std::max(0.0, child_obj - current_obj);
                    } else {
                        cand.up_degradation = 0.0;
                    }
                } else {
                    cand.up_degradation = 0.0;
                    cand.is_up_infeasible = false;
                }
            } catch (...) {
                cand.up_degradation = 0.0;
                cand.is_up_infeasible = false;
            }
        }

        // 3. Domain Reduction detection (Achterberg 2005)
        if (cand.is_down_infeasible && cand.is_up_infeasible) {
            result.subproblem_infeasible = true;
        } else if (cand.is_down_infeasible && !cand.is_up_infeasible) {
            DomainReduction dr;
            dr.variable_index = var_idx;
            dr.new_lower = model::Bound::finite(x_ceil);
            dr.new_upper = model.variable_upper[var_idx];
            if (dr.new_upper.is_finite() && dr.new_lower.value >= dr.new_upper.value - 1e-9) {
                dr.is_fixed = true;
            }
            result.domain_reductions.push_back(dr);
        } else if (cand.is_up_infeasible && !cand.is_down_infeasible) {
            DomainReduction dr;
            dr.variable_index = var_idx;
            dr.new_lower = model.variable_lower[var_idx];
            dr.new_upper = model::Bound::finite(x_floor);
            if (dr.new_lower.is_finite() && dr.new_upper.value <= dr.new_lower.value + 1e-9) {
                dr.is_fixed = true;
            }
            result.domain_reductions.push_back(dr);
        }

        // 4. Combined Score (Achterberg 2005)
        const double d_down = cand.is_down_infeasible ? infeasible_delta : cand.down_degradation;
        const double d_up = cand.is_up_infeasible ? infeasible_delta : cand.up_degradation;
        cand.score = compute_strong_branching_score(d_down, d_up, options.score_mu);

        // 5. Update VariablePseudoCost tracker to warm-start pseudo-costs
        if (pseudo_costs != nullptr && options.update_pseudo_costs && var_idx < pseudo_costs->size()) {
            if (!cand.is_down_infeasible && std::isfinite(cand.down_degradation) && cand.down_degradation >= 0.0) {
                (*pseudo_costs)[var_idx].record_down(cand.down_degradation, frac);
            }
            if (!cand.is_up_infeasible && std::isfinite(cand.up_degradation) && cand.up_degradation >= 0.0) {
                (*pseudo_costs)[var_idx].record_up(cand.up_degradation, 1.0 - frac);
            }
        }

        if (cand.score > result.best_score) {
            result.best_score = cand.score;
            result.best_variable = var_idx;
        }

        result.candidates.push_back(cand);
    }

    return result;
}

StrongBranchingResult evaluate_strong_branching(
    const model::Model& model,
    const std::vector<double>& primal,
    double current_obj,
    const std::optional<lp::dual::BasisState>& current_basis,
    const StrongBranchingOptions& options,
    std::vector<VariablePseudoCost>* pseudo_costs) {
    return evaluate_strong_branching(model, primal, current_obj, current_basis, {}, options, pseudo_costs);
}

} // namespace markov_cero::milp
