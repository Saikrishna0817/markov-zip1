#include "markov_cero/foundation/build_info.hpp"
int main() {
    if (markov_cero::foundation::version() != "0.5.1") return 1;
    if (markov_cero::foundation::milestone() != "M5") return 2;
    if (!markov_cero::foundation::contains_solver_algorithms()) return 3;
    return 0;
}
