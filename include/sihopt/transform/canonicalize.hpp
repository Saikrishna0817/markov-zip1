#pragma once
#include "sihopt/linalg/dense_lu.hpp"
#include "sihopt/model/model.hpp"
#include <vector>
namespace sihopt::transform {
struct OriginalVariableMap { double offset{}; std::vector<std::size_t> canonical_index; std::vector<double> multiplier; };
struct CanonicalizationRecord { std::vector<OriginalVariableMap> variables; double objective_sign{1.0}; std::size_t structural_variables{}; };
struct CanonicalModel { linalg::DenseMatrix matrix; std::vector<double> rhs; std::vector<double> objective; double objective_offset{}; CanonicalizationRecord record; void validate() const; };
[[nodiscard]] CanonicalModel canonicalize(const model::Model& input);
[[nodiscard]] std::vector<double> reconstruct_primal(const CanonicalModel&,const std::vector<double>& canonical_primal);
[[nodiscard]] double reconstruct_objective(const CanonicalModel&,double canonical_objective);
}
