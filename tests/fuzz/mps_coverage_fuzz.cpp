#include "markov_cero/io/mps.hpp"
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
std::array<unsigned char, 65536> trial{}, total{};
}
extern "C" void __sanitizer_cov_trace_pc() {
    const auto pc = reinterpret_cast<std::uintptr_t>(__builtin_return_address(0));
    trial[(pc >> 4U) & 65535U] = 1U;
}
int main(int argc, char** argv) {
    const int seconds = argc > 1 ? std::max(1, std::atoi(argv[1])) : 60;
    std::mt19937_64 random(0x4d31434f564552ULL);
    std::vector<std::string> corpus = {
        "", "NAME X\nROWS\n N O\nENDATA\n",
        "NAME X\nROWS\n N O\n L R\nCOLUMNS\n X O 1 R 1\nRHS\n R R 1\nENDATA\n"};
    std::size_t executions = 0, discoveries = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    markov_cero::io::MpsLimits limits;
    limits.maximum_bytes = 4096;
    limits.maximum_lines = 512;
    limits.maximum_rows = 128;
    limits.maximum_columns = 128;
    limits.maximum_nonzeros = 1024;
    limits.maximum_name_bytes = 64;
    while (std::chrono::steady_clock::now() < deadline) {
        std::string input = corpus[random() % corpus.size()];
        const unsigned op = random() % 4U;
        if (op == 0U && !input.empty())
            input[random() % input.size()] ^= static_cast<char>(1U << (random() % 8U));
        else if (op == 1U && input.size() < limits.maximum_bytes)
            input.insert(input.begin() +
                             static_cast<std::ptrdiff_t>(random() % (input.size() + 1U)),
                         static_cast<char>(random() % 128U));
        else if (op == 2U && !input.empty())
            input.erase(input.begin() + static_cast<std::ptrdiff_t>(random() % input.size()));
        else if (op == 3U && corpus.size() > 1U && input.size() < 2048U) {
            const auto& other = corpus[random() % corpus.size()];
            input += other.substr(0, std::min<std::size_t>(other.size(), 2048U - input.size()));
        }
        trial.fill(0U);
        try {
            (void)markov_cero::io::parse_mps_string(input, limits);
        } catch (const std::exception&) {
        }
        bool novel = false;
        for (std::size_t i = 0; i < trial.size(); ++i)
            if (trial[i] && !total[i]) {
                total[i] = 1U;
                novel = true;
            }
        if (novel && corpus.size() < 4096U) {
            corpus.push_back(std::move(input));
            ++discoveries;
        }
        ++executions;
    }
    std::size_t edges = 0;
    for (auto value : total)
        edges += value != 0U;
    std::cout << "coverage-guided fuzz passed: executions=" << executions
              << " corpus=" << corpus.size() << " discoveries=" << discoveries
              << " edge_slots=" << edges << " seconds=" << seconds << "\n";
}
