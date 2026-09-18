#pragma once
#include "sihopt/lp/reference/revised_simplex.hpp"
#include <string>
namespace sihopt::verify {
struct ReferenceVerification { bool accepted{}; double maximum_primal_violation{}; double maximum_dual_violation{}; double maximum_complementarity_violation{}; std::string message; };
[[nodiscard]] ReferenceVerification verify_reference_result(const transform::CanonicalModel& model,const lp::reference::Result& result,double tolerance=1e-8);
}
