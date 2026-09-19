#include "markov_cero/io/mps.hpp"
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
int main() {
    std::mt19937_64 generator(0x5349484f5054ULL);
    std::uniform_int_distribution<int> length(0, 512), byte(0, 255);
    markov_cero::io::MpsLimits limits; limits.maximum_bytes = 1024; limits.maximum_lines = 256; limits.maximum_rows = 128; limits.maximum_columns = 128; limits.maximum_nonzeros = 1024; limits.maximum_name_bytes = 64;
    for (int trial = 0; trial < 5000; ++trial) {
        std::string input(static_cast<std::size_t>(length(generator)), '\0');
        for (char& value : input) value = static_cast<char>(byte(generator));
        try { (void)markov_cero::io::parse_mps_string(input, limits); } catch (const std::exception&) {}
    }
    std::cout << "deterministic MPS fuzz smoke passed\n";
}
