#include "sihopt/foundation/build_info.hpp"
#include <iostream>
int main() {
    std::cout << "SIHOpt " << sihopt::foundation::version() << " milestone "
              << sihopt::foundation::milestone() << " (sparse basis linear algebra and update substrate)\n";
    return 0;
}
