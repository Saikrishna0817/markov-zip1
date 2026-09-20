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

std::vector<std::size_t> find_fractional_variables(const std::vector<double>& primal,
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

std::size_t select_most_fractional(const std::vector<double>& primal,
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

std::size_t select_pseudo_cost(const std::vector<double>& primal,
                               const std::vector<std::size_t>& candidates,
                               const std::vector<VariablePseudoCost>& pseudo_costs) {
    if (candidates.empty()) {
        throw std::invalid_argument("candidates list is empty in select_pseudo_cost");
    }

    // Compute global average pseudo costs across observed variables
    double avg_down = 1.0;
    double avg_up = 1.0;
    double sum_down = 0.0;
    double sum_up = 0.0;
    std::size_t count_down = 0;
    std::size_t count_up = 0;

    for (const auto& pc : pseudo_costs) {
        if (pc.down_count > 0) {
            sum_down += pc.down_cost();
            ++count_down;
        }
        if (pc.up_count > 0) {
            sum_up += pc.up_cost();
            ++count_up;
        }
    }
    if (count_down > 0) {
        avg_down = sum_down / static_cast<double>(count_down);
    }
    if (count_up > 0) {
        avg_up = sum_up / static_cast<double>(count_up);
    }

    std::size_t best_var = candidates[0];
    double max_score = -1.0;

    for (std::size_t j : candidates) {
        const double x = primal[j];
        const double frac = x - std::floor(x);
        const double down_frac = frac;
        const double up_frac = 1.0 - frac;

        const double down_cost = (j < pseudo_costs.size() && pseudo_costs[j].down_count > 0)
                                     ? pseudo_costs[j].down_cost()
                                     : avg_down;
        const double up_cost = (j < pseudo_costs.size() && pseudo_costs[j].up_count > 0)
                                   ? pseudo_costs[j].up_cost()
                                   : avg_up;

        const double down_deg = down_cost * down_frac;
        const double up_deg = up_cost * up_frac;

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

std::size_t select_branching_variable(const std::vector<double>& primal,
                                      const std::vector<model::VariableType>& types,
                                      const std::vector<VariablePseudoCost>& pseudo_costs,
                                      BranchingStrategy strategy, double integrality_tol) {
    const auto candidates = find_fractional_variables(primal, types, integrality_tol);
    if (candidates.empty()) {
        return types.size(); // None fractional
    }
    if (strategy == BranchingStrategy::most_fractional) {
        return select_most_fractional(primal, candidates);
    }
    // pseudo_cost, strong_branching, and reliability fallback to pseudo-cost
    // when called without an active solver/basis state context
    return select_pseudo_cost(primal, candidates, pseudo_costs);
}

} // namespace markov_cero::milp
