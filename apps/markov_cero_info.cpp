#include "markov_cero/foundation/build_info.hpp"
#include <iostream>
int main() {
    std::cout << "markov-cero " << markov_cero::foundation::version() << " milestone "
              << markov_cero::foundation::milestone()
              << " (sparse basis linear algebra and update substrate)\n";
    return 0;
}
