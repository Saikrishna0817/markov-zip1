#pragma once

#include "markov_cero/linalg/dense_lu.hpp"
#include "markov_cero/linalg/sparse_basis.hpp"
#include "markov_cero/model/model.hpp"
#include "markov_cero/transform/canonicalize.hpp"

#include <cstddef>
#include <vector>

namespace markov_cero::transform {

struct SparseCanonicalModel {
    linalg::SparseCsc matrix;
    std::vector<double> rhs;
    std::vector<double> objective;
    double objective_offset{0.0};
    CanonicalizationRecord record;
    std::vector<model::VariableType> original_variable_types;

    void validate() const;
    [[nodiscard]] std::vector<double> multiply(const std::vector<double>& x) const;
    [[nodiscard]] std::vector<double> multiply_transpose(const std::vector<double>& y) const;
    [[nodiscard]] CanonicalModel to_dense() const;
};

[[nodiscard]] SparseCanonicalModel sparse_canonicalize(const model::Model& input,
                                                       bool relax_integrality = false);
[[nodiscard]] std::vector<double> reconstruct_primal(const SparseCanonicalModel& model,
                                                     const std::vector<double>& canonical_primal);
[[nodiscard]] double reconstruct_objective(const SparseCanonicalModel& model,
                                           double canonical_objective);

} // namespace markov_cero::transform
