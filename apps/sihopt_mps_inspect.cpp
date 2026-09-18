#include "sihopt/io/mps.hpp"
#include <fstream>
#include <iostream>
#include <string>

static std::string escape(const std::string& text) { std::string out; for (char c : text) { if (c == '"' || c == '\\') out += '\\'; out += c; } return out; }
int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: sihopt-mps-inspect MODEL.mps\n"; return 2; }
    std::ifstream input(argv[1]); if (!input) { std::cerr << "cannot open input\n"; return 3; }
    try {
        const auto model = sihopt::io::parse_mps(input);
        std::cout << "{\"name\":\"" << escape(model.name) << "\",\"sense\":\"" << sihopt::model::to_string(model.objective_sense)
                  << "\",\"rows\":" << model.matrix.row_count << ",\"columns\":" << model.matrix.column_count
                  << ",\"nonzeros\":" << model.matrix.value.size() << "}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 4; }
}
