#pragma once

#include "markov_cero/lp/dual/dual_simplex.hpp"
#include "markov_cero/milp/cut_pool.hpp"
#include "markov_cero/model/model.hpp"
#include "markov_cero/transform/sparse_canonical_model.hpp"

#include <cstddef>
#include <vector>

namespace markov_cero::milp {

[[nodiscard]] std::vector<Cut>
generate_gomory_cuts(const model::Model& model, const std::vector<double>& original_primal,
                     const transform::SparseCanonicalModel& canonical,
                     const lp::dual::BasisState& basis_state, std::size_t max_cuts = 10,
                     double min_fractionality = 0.05);

} // namespace markov_cero::milp
