#include "sihopt/io/mps.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    sihopt::io::MpsLimits limits; limits.maximum_bytes = 1U << 20U; limits.maximum_lines = 10000U; limits.maximum_rows = 10000U; limits.maximum_columns = 10000U; limits.maximum_nonzeros = 100000U;
    try { (void)sihopt::io::parse_mps_string(std::string(reinterpret_cast<const char*>(data), size), limits); } catch (...) {}
    return 0;
}
