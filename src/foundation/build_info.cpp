#include "sihopt/foundation/build_info.hpp"
namespace sihopt::foundation {
static_assert(version() == "0.5.1");
static_assert(milestone() == "M5");
static_assert(contains_solver_algorithms());
}
