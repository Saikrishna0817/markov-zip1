#include "markov_cero/io/mps.hpp"
#include "markov_cero/verify/primal_verifier.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

static void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main() {
    try {
        const std::string mps = R"(NAME BLEND
OBJSENSE
 MIN
ROWS
 N COST
 E TOTAL
 L SULFUR
COLUMNS
 A COST 40 TOTAL 1
 A SULFUR 0.01
 B COST 30 TOTAL 1
 B SULFUR 0.03
RHS
 RHS1 TOTAL 100 SULFUR 2
BOUNDS
 LO BND A 0
 LO BND B 0
ENDATA
)";
        const auto model = markov_cero::io::parse_mps_string(mps);
        require(model.matrix.row_count == 2 && model.matrix.column_count == 2 && model.matrix.value.size() == 4, "parsed dimensions");
        auto report = markov_cero::verify::verify_primal(model, {{50, 50}, 3500}); require(report.passed, "feasible blend rejected");
        report = markov_cero::verify::verify_primal(model, {{0, 100}, 3000}); require(!report.passed && report.maximum_row_violation == 1.0, "sulfur violation missed");
        report = markov_cero::verify::verify_primal(model, {{50, 50}, 3499}); require(!report.passed && report.objective_difference == 1.0, "objective corruption missed");
        const auto empty = markov_cero::io::parse_mps_string("NAME X\nROWS\n N O\nENDATA\n"); require(empty.matrix.row_count == 0 && empty.matrix.column_count == 0, "empty model rejected");
        bool failed = false; try { (void)markov_cero::io::parse_mps_string("NAME X\nROWS\n N O\n L R\nCOLUMNS\n X O nan\nENDATA\n"); } catch (const markov_cero::io::MpsError&) { failed = true; } require(failed, "NaN accepted");
        const auto duplicate = markov_cero::io::parse_mps_string("NAME D\nROWS\n N O\n L R\nCOLUMNS\n X O 1 R 2\n X R -2\nRHS\n R1 R 0\nENDATA\n"); require(duplicate.matrix.value.empty(), "duplicate cancellation failed");
        std::cout << "M1 tests passed\n"; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
