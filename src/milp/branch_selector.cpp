#include "markov_cero/milp/branch_selector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace markov_cero::milp {

double VariablePseudoCost::down_cost() const noexcept {
    if (down_count > 0) {
        return down_sum / static_cast<double>(down_count);
    }
    return 1.0;
}

double VariablePseudoCost::up_cost() const noexcept {
    if (up_count > 0) {
        return up_sum / static_cast<double>(up_count);
    }
    return 1.0;
}

void VariablePseudoCost::record_down(double delta_obj, double fraction) noexcept {
    if (fraction > 1e-6 && std::isfinite(delta_obj) && delta_obj >= 0.0) {
        down_sum += delta_obj / fraction;
        ++down_count;
    }
}

void VariablePseudoCost::record_up(double delta_obj, double fraction) noexcept {
    if (fraction > 1e-6 && std::isfinite(delta_obj) && delta_obj >= 0.0) {
        up_sum += delta_obj / fraction;
        ++up_count;
    }
}

std::vector<std::size_t> find_fractional_variables(
    const std::vector<double>& primal,
    const std::vector<model::VariableType>& types,
    double integrality_tol) {
    std::vector<std::size_t> candidates;
    const std::size_t n = std::min(primal.size(), types.size());
    for (std::size_t j = 0; j < n; ++j) {
        if (types[j] == model::VariableType::continuous) {
            continue;
        }
        const double x = primal[j];
        if (!std::isfinite(x)) {
            continue;
        }
        const double rounded = std::round(x);
        if (std::abs(x - rounded) > integrality_tol) {
            candidates.push_back(j);
        }
    }
    return candidates;
}

std::size_t select_most_fractional(
    const std::vector<double>& primal,
    const std::vector<std::size_t>& candidates) {
    if (candidates.empty()) {
        throw std::invalid_argument("candidates list is empty in select_most_fractional");
    }

    std::size_t best_var = candidates[0];
    double max_fractionality = -1.0;

    for (std::size_t j : candidates) {
        const double x = primal[j];
        const double frac = x - std::floor(x);
        const double distance = std::min(frac, 1.0 - frac);
        if (distance > max_fractionality) {
            max_fractionality = distance;
            best_var = j;
        }
    }
    return best_var;
}

std::size_t select_pseudo_cost(
    const std::vector<double>& primal,
    const std::vector<std::size_t>& candidates,
    const std::vector<VariablePseudoCost>& pseudo_costs) {
    if (candidates.empty()) {
        throw std::invalid_argument("candidates list is empty in select_pseudo_cost");
    }

    // Check if any candidate has insufficient history (< 4 observations on either side)
    // If so, fall back to most fractional for reliable initial exploration
    bool has_unreliable = false;
    for (std::size_t j : candidates) {
        if (j < pseudo_costs.size()) {
            if (pseudo_costs[j].down_count < 4 || pseudo_costs[j].up_count < 4) {
                has_unreliable = true;
                break;
            }
        } else {
            has_unreliable = true;
            break;
        }
    }

    if (has_unreliable) {
        return select_most_fractional(primal, candidates);
    }

    std::size_t best_var = candidates[0];
    double max_score = -1.0;

    for (std::size_t j : candidates) {
        const double x = primal[j];
        const double frac = x - std::floor(x);
        const double down_frac = frac;
        const double up_frac = 1.0 - frac;

        const double down_deg = pseudo_costs[j].down_cost() * down_frac;
        const double up_deg = pseudo_costs[j].up_cost() * up_frac;

        // Standard Achterberg product score: (delta_down + eps) * (delta_up + eps)
        constexpr double eps = 1e-6;
        const double score = (down_deg + eps) * (up_deg + eps);

        if (score > max_score) {
            max_score = score;
            best_var = j;
        }
    }
    return best_var;
}

std::size_t select_branching_variable(
    const std::vector<double>& primal,
    const std::vector<model::VariableType>& types,
    const std::vector<VariablePseudoCost>& pseudo_costs,
    BranchingStrategy strategy,
    double integrality_tol) {
    const auto candidates = find_fractional_variables(primal, types, integrality_tol);
    if (candidates.empty()) {
        return types.size(); // None fractional
    }
    if (strategy == BranchingStrategy::pseudo_cost) {
        return select_pseudo_cost(primal, candidates, pseudo_costs);
    }
    return select_most_fractional(primal, candidates);
}

} // namespace markov_cero::milp
