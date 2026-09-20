#include "markov_cero/milp/heuristics.hpp"

#include "markov_cero/lp/reference/revised_simplex.hpp"
#include "markov_cero/transform/sparse_canonical_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace markov_cero::milp {

bool check_integer_feasibility(const model::Model& model, const std::vector<double>& primal,
                               double feasibility_tol, double integrality_tol) {
    if (primal.size() != model.matrix.column_count) {
        return false;
    }

    // 1. Variable bounds and integrality
    for (std::size_t j = 0; j < primal.size(); ++j) {
        const double x = primal[j];
        if (!std::isfinite(x)) {
            return false;
        }
        if (model.variable_lower[j].is_finite()) {
            if (x < model.variable_lower[j].value - feasibility_tol) {
                return false;
            }
        }
        if (model.variable_upper[j].is_finite()) {
            if (x > model.variable_upper[j].value + feasibility_tol) {
                return false;
            }
        }
        if (model.variable_type[j] != model::VariableType::continuous) {
            const double rounded = std::round(x);
            if (std::abs(x - rounded) > integrality_tol) {
                return false;
            }
        }
    }

    // 2. Row constraint bounds
    const auto ax = model.matrix.multiply(primal);
    for (std::size_t i = 0; i < ax.size(); ++i) {
        if (!std::isfinite(ax[i])) {
            return false;
        }
        if (model.row_lower[i].is_finite()) {
            if (ax[i] < model.row_lower[i].value - feasibility_tol) {
                return false;
            }
        }
        if (model.row_upper[i].is_finite()) {
            if (ax[i] > model.row_upper[i].value + feasibility_tol) {
                return false;
            }
        }
    }

    return true;
}

double compute_objective(const model::Model& model, const std::vector<double>& primal) {
    long double obj = model.objective_offset;
    const std::size_t n = std::min(primal.size(), model.objective.size());
    for (std::size_t j = 0; j < n; ++j) {
        obj += static_cast<long double>(model.objective[j]) * primal[j];
    }
    return static_cast<double>(obj);
}

HeuristicResult simple_rounding(const model::Model& model,
                                const std::vector<double>& continuous_primal,
                                double feasibility_tol, double integrality_tol) {
    HeuristicResult result;
    if (continuous_primal.size() != model.matrix.column_count) {
        return result;
    }

    const auto clamp_var = [&](std::size_t j, double val) {
        if (model.variable_lower[j].is_finite()) {
            val = std::max(val, model.variable_lower[j].value);
        }
        if (model.variable_upper[j].is_finite()) {
            val = std::min(val, model.variable_upper[j].value);
        }
        return val;
    };

    // Strategy 1: Nearest integer rounding
    std::vector<double> candidate1 = continuous_primal;
    for (std::size_t j = 0; j < candidate1.size(); ++j) {
        if (model.variable_type[j] != model::VariableType::continuous) {
            candidate1[j] = clamp_var(j, std::round(candidate1[j]));
        }
    }
    if (check_integer_feasibility(model, candidate1, feasibility_tol, integrality_tol)) {
        result.found = true;
        result.primal = candidate1;
        result.objective = compute_objective(model, candidate1);
        return result;
    }

    // Strategy 2: Objective-directed rounding
    std::vector<double> candidate2 = continuous_primal;
    for (std::size_t j = 0; j < candidate2.size(); ++j) {
        if (model.variable_type[j] != model::VariableType::continuous) {
            const double c = model.objective[j];
            const double val = candidate2[j];
            double rounded = std::round(val);
            if (c > 1e-9) {
                rounded = std::floor(val);
            } else if (c < -1e-9) {
                rounded = std::ceil(val);
            }
            candidate2[j] = clamp_var(j, rounded);
        }
    }
    if (check_integer_feasibility(model, candidate2, feasibility_tol, integrality_tol)) {
        result.found = true;
        result.primal = candidate2;
        result.objective = compute_objective(model, candidate2);
        return result;
    }

    // Strategy 3: Up-rounding (often feasible for covering constraints)
    std::vector<double> candidate3 = continuous_primal;
    for (std::size_t j = 0; j < candidate3.size(); ++j) {
        if (model.variable_type[j] != model::VariableType::continuous) {
            candidate3[j] = clamp_var(j, std::ceil(candidate3[j]));
        }
    }
    if (check_integer_feasibility(model, candidate3, feasibility_tol, integrality_tol)) {
        result.found = true;
        result.primal = candidate3;
        result.objective = compute_objective(model, candidate3);
        return result;
    }

    return result;
}

HeuristicResult feasibility_pump(const model::Model& model,
                                 const std::vector<double>& continuous_primal,
                                 std::size_t max_iterations, double feasibility_tol,
                                 double integrality_tol) {
    HeuristicResult result;
    if (continuous_primal.size() != model.matrix.column_count) {
        return result;
    }

    // Check if continuous point is already integer feasible
    if (check_integer_feasibility(model, continuous_primal, feasibility_tol, integrality_tol)) {
        result.found = true;
        result.primal = continuous_primal;
        result.objective = compute_objective(model, continuous_primal);
        return result;
    }

    std::vector<double> current_lp_x = continuous_primal;
    std::vector<std::vector<double>> visited_rounded;

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        // 1. Round integer variables to nearest integer
        std::vector<double> rounded_x = current_lp_x;
        for (std::size_t j = 0; j < rounded_x.size(); ++j) {
            if (model.variable_type[j] != model::VariableType::continuous) {
                double val = std::round(rounded_x[j]);
                if (model.variable_lower[j].is_finite()) {
                    val = std::max(val, model.variable_lower[j].value);
                }
                if (model.variable_upper[j].is_finite()) {
                    val = std::min(val, model.variable_upper[j].value);
                }
                rounded_x[j] = val;
            }
        }

        // Check if rounded point is linearly feasible
        if (check_integer_feasibility(model, rounded_x, feasibility_tol, integrality_tol)) {
            result.found = true;
            result.primal = rounded_x;
            result.objective = compute_objective(model, rounded_x);
            return result;
        }

        // Cycling detection & perturbation (Fischetti, Lodi, Glover 2005)
        bool cycle_detected = false;
        for (const auto& prev : visited_rounded) {
            bool identical = true;
            for (std::size_t j = 0; j < rounded_x.size(); ++j) {
                if (model.variable_type[j] != model::VariableType::continuous) {
                    if (std::abs(rounded_x[j] - prev[j]) > integrality_tol) {
                        identical = false;
                        break;
                    }
                }
            }
            if (identical) {
                cycle_detected = true;
                break;
            }
        }

        if (cycle_detected) {
            // Find integer variables with continuous relaxation closest to 0.5 (most ambiguous)
            std::vector<std::pair<double, std::size_t>> ambig;
            for (std::size_t j = 0; j < model.matrix.column_count; ++j) {
                if (model.variable_type[j] != model::VariableType::continuous) {
                    const double frac = current_lp_x[j] - std::floor(current_lp_x[j]);
                    const double dist = std::abs(frac - 0.5);
                    ambig.push_back({dist, j});
                }
            }
            std::sort(ambig.begin(), ambig.end());

            // Flip top 1 to min(3, ambig.size()) candidates
            const std::size_t flip_count = std::min<std::size_t>(3, ambig.size());
            for (std::size_t k = 0; k < flip_count; ++k) {
                const std::size_t flip_j = ambig[k].second;
                const double lo = model.variable_lower[flip_j].is_finite()
                                      ? model.variable_lower[flip_j].value
                                      : 0.0;
                const double up = model.variable_upper[flip_j].is_finite()
                                      ? model.variable_upper[flip_j].value
                                      : 1.0;
                if (std::abs(up - lo - 1.0) < 1e-4) {
                    // Binary flip
                    rounded_x[flip_j] = (rounded_x[flip_j] <= lo + 1e-4) ? up : lo;
                } else {
                    // General integer shift away from round direction
                    if (current_lp_x[flip_j] > rounded_x[flip_j] &&
                        rounded_x[flip_j] + 1.0 <= up + 1e-4) {
                        rounded_x[flip_j] += 1.0;
                    } else if (rounded_x[flip_j] - 1.0 >= lo - 1e-4) {
                        rounded_x[flip_j] -= 1.0;
                    }
                }
            }
        }
        visited_rounded.push_back(rounded_x);

        // 2. Set up distance-minimization LP
        // min sum_{j in I} |x_j - x_tilde_j|
        // For binary/bounded: if x_tilde == lower, cost = +1; if x_tilde == upper, cost = -1; else
        // +1 / -1
        model::Model pump_model = model;
        pump_model.objective.assign(model.matrix.column_count, 0.0);
        pump_model.objective_offset = 0.0;
        pump_model.objective_sense = model::ObjectiveSense::minimize;

        for (std::size_t j = 0; j < model.matrix.column_count; ++j) {
            if (model.variable_type[j] != model::VariableType::continuous) {
                const double x_tilde = rounded_x[j];
                const double lo =
                    model.variable_lower[j].is_finite() ? model.variable_lower[j].value : 0.0;
                const double up =
                    model.variable_upper[j].is_finite() ? model.variable_upper[j].value : 1.0;
                if (x_tilde <= lo + 1e-4) {
                    pump_model.objective[j] = 1.0;
                } else if (x_tilde >= up - 1e-4) {
                    pump_model.objective[j] = -1.0;
                } else {
                    pump_model.objective[j] = (current_lp_x[j] >= x_tilde) ? 1.0 : -1.0;
                }
            }
        }

        try {
            const auto canon =
                transform::sparse_canonicalize(pump_model, /*relax_integrality=*/true);
            const auto dense = canon.to_dense();
            lp::reference::Options popts;
            popts.iteration_limit = 5000;
            const auto lpres = lp::reference::solve(dense, popts);
            if (lpres.status != lp::reference::SolveStatus::optimal) {
                break;
            }
            current_lp_x = transform::reconstruct_primal(canon, lpres.primal);
        } catch (...) {
            break;
        }

        // Check if new LP solution is integer feasible
        if (check_integer_feasibility(model, current_lp_x, feasibility_tol, integrality_tol)) {
            result.found = true;
            result.primal = current_lp_x;
            result.objective = compute_objective(model, current_lp_x);
            return result;
        }
    }

    return result;
}

} // namespace markov_cero::milp
