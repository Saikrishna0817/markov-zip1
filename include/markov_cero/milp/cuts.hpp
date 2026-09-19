#pragma once

#include "markov_cero/lp/dual/dual_simplex.hpp"
#include "markov_cero/model/model.hpp"
#include "markov_cero/transform/sparse_canonical_model.hpp"

#include <cstddef>
#include <limits>
#include <vector>

namespace markov_cero::milp {

struct Cut {
    std::vector<double> coefficients; // length = model.matrix.column_count
    double rhs{0.0};                  // sum c_j x_j >= rhs
    double violation{0.0};
};

[[nodiscard]] double compute_cut_efficacy(const Cut& cut) noexcept;

[[nodiscard]] double compute_cosine_similarity(const Cut& a, const Cut& b) noexcept;

[[nodiscard]] double mir_function(double a, double f0) noexcept;

[[nodiscard]] std::vector<Cut> filter_cuts(
    std::vector<Cut> candidates,
    std::size_t max_cuts = 10,
    double min_violation = 1e-4,
    double max_parallelism = 0.95);

[[nodiscard]] std::vector<Cut> generate_gomory_cuts(
    const model::Model& model,
    const std::vector<double>& original_primal,
    const transform::SparseCanonicalModel& canonical,
    const lp::dual::BasisState& basis_state,
    std::size_t max_cuts = 10,
    double min_fractionality = 0.05);

[[nodiscard]] std::vector<Cut> generate_mir_cuts(
    const model::Model& model,
    const std::vector<double>& original_primal,
    const transform::SparseCanonicalModel& canonical,
    const lp::dual::BasisState& basis_state,
    std::size_t max_cuts = 10,
    double min_fractionality = 0.05);

void add_cuts_to_model(
    model::Model& model,
    const std::vector<Cut>& cuts);

} // namespace markov_cero::milp
