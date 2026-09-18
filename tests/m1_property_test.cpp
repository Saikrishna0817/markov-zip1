#include "sihopt/model/model.hpp"
#include "sihopt/verify/primal_verifier.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

static void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main() {
    std::mt19937_64 random(0x4d3150524f50ULL);
    std::uniform_real_distribution<double> coefficient(-5.0, 5.0), value(-3.0, 3.0);
    for (int trial = 0; trial < 250; ++trial) {
        constexpr std::size_t rows = 7, columns = 5;
        sihopt::model::SparseMatrixBuilder builder(rows, columns);
        std::vector<std::vector<long double>> dense(rows, std::vector<long double>(columns));
        for (std::size_t column = 0; column < columns; ++column) for (std::size_t row = 0; row < rows; ++row) {
            if ((random() & 3U) == 0U) continue;
            const double entry = coefficient(random); builder.add(row, column, entry); dense[row][column] += entry;
            if ((random() & 7U) == 0U) { builder.add(row, column, -entry); dense[row][column] -= entry; }
        }
        const auto matrix = builder.build(); std::vector<double> x(columns); for (double& item : x) item = value(random);
        const auto actual = matrix.multiply(x);
        for (std::size_t row = 0; row < rows; ++row) { long double expected = 0; for (std::size_t column = 0; column < columns; ++column) expected += dense[row][column] * x[column]; require(std::abs(actual[row] - static_cast<double>(expected)) <= 1e-12 * std::max(1.0, std::abs(actual[row])), "CSC/dense mismatch"); }
    }
    return 0;
}
