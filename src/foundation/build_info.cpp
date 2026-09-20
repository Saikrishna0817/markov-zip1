#include "markov_cero/foundation/build_info.hpp"
namespace markov_cero::foundation {
static_assert(version() == "0.5.2");
static_assert(milestone() == "M5");
static_assert(contains_solver_algorithms());
} // namespace markov_cero::foundation
