#include "sihopt/foundation/build_info.hpp"
int main() {
    if (sihopt::foundation::version() != "0.5.1") return 1;
    if (sihopt::foundation::milestone() != "M5") return 2;
    if (!sihopt::foundation::contains_solver_algorithms()) return 3;
    return 0;
}
