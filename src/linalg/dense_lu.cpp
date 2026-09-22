#include "markov_cero/linalg/dense_lu.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace markov_cero::linalg {
namespace {
constexpr std::size_t maximum_dense_elements = 16U * 1024U * 1024U;
std::size_t checked_product(std::size_t a, std::size_t b) {
    if (a != 0U && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::length_error("dense matrix size overflow");
    const auto n = a * b;
    if (n > maximum_dense_elements)
        throw std::length_error("dense oracle element limit exceeded");
    return n;
}
void finite(double v, const char* message) {
    if (!std::isfinite(v))
        throw std::overflow_error(message);
}
} // namespace
double& DenseMatrix::operator()(std::size_t r, std::size_t c) {
    if (r >= rows || c >= columns)
        throw std::out_of_range("dense matrix index");
    const auto n = checked_product(rows, columns);
    if (values.size() != n)
        throw std::invalid_argument("dense matrix dimension mismatch");
    return values.at(r * columns + c);
}
double DenseMatrix::operator()(std::size_t r, std::size_t c) const {
    if (r >= rows || c >= columns)
        throw std::out_of_range("dense matrix index");
    const auto n = checked_product(rows, columns);
    if (values.size() != n)
        throw std::invalid_argument("dense matrix dimension mismatch");
    return values.at(r * columns + c);
}
void DenseMatrix::validate() const {
    if (values.size() != checked_product(rows, columns))
        throw std::invalid_argument("dense matrix dimension mismatch");
    for (double v : values)
        finite(v, "dense matrix entry non-finite");
}
DenseLu DenseLu::factorize(const DenseMatrix& a, double pivot_tolerance) {
    a.validate();
    if (a.rows != a.columns)
        throw std::invalid_argument("LU requires square matrix");
    DenseLu lu;
    lu.n = a.rows;
    lu.lu = a.values;
    lu.pivots.resize(lu.n);
    for (std::size_t i = 0; i < lu.n; ++i)
        lu.pivots[i] = i;
    for (std::size_t k = 0; k < lu.n; ++k) {
        std::size_t piv = k;
        double best = std::abs(lu.lu[k * lu.n + k]);
        for (std::size_t i = k + 1; i < lu.n; ++i) {
            const double cand = std::abs(lu.lu[i * lu.n + k]);
            if (cand > best) {
                best = cand;
                piv = i;
            }
        }
        if (best <= pivot_tolerance)
            throw std::runtime_error("singular dense basis");
        if (piv != k) {
            for (std::size_t j = 0; j < lu.n; ++j)
                std::swap(lu.lu[k * lu.n + j], lu.lu[piv * lu.n + j]);
            std::swap(lu.pivots[k], lu.pivots[piv]);
        }
        const double akk = lu.lu[k * lu.n + k];
        for (std::size_t i = k + 1; i < lu.n; ++i) {
            lu.lu[i * lu.n + k] /= akk;
            const double lik = lu.lu[i * lu.n + k];
            for (std::size_t j = k + 1; j < lu.n; ++j)
                lu.lu[i * lu.n + j] -= lik * lu.lu[k * lu.n + j];
        }
    }
    return lu;
}
std::vector<double> DenseLu::solve(const std::vector<double>& b) const {
    if (b.size() != n)
        throw std::invalid_argument("RHS dimension mismatch");
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = b[pivots[i]];
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j)
            x[i] -= lu[i * n + j] * x[j];
    }
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t j = i + 1; j < n; ++j)
            x[i] -= lu[i * n + j] * x[j];
        x[i] /= lu[i * n + i];
        finite(x[i], "non-finite dense solve");
    }
    return x;
}
std::vector<double> DenseLu::solve_transpose(const std::vector<double>& b) const {
    if (b.size() != n)
        throw std::invalid_argument("RHS dimension mismatch");
    std::vector<double> y = b;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j)
            y[i] -= lu[j * n + i] * y[j];
        y[i] /= lu[i * n + i];
    }
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t j = i + 1; j < n; ++j)
            y[i] -= lu[j * n + i] * y[j];
        finite(y[i], "non-finite dense transpose solve");
    }
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[pivots[i]] = y[i];
    return x;
}
} // namespace markov_cero::linalg
