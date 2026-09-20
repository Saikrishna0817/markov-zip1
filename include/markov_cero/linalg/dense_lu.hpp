#pragma once
#include <cstddef>
#include <vector>
namespace markov_cero::linalg {
struct DenseMatrix {
    std::size_t rows{};
    std::size_t columns{};
    std::vector<double> values;
    double& operator()(std::size_t r, std::size_t c);
    double operator()(std::size_t r, std::size_t c) const;
    void validate() const;
};
struct LuDiagnostics {
    double maximum_original_entry{};
    double minimum_absolute_pivot{};
    double maximum_absolute_pivot{};
    double pivot_ratio{};
};
class DenseLu final {
  public:
    static DenseLu factorize(const DenseMatrix& matrix, double singular_tolerance = 1e-14);
    [[nodiscard]] std::vector<double> solve(const std::vector<double>& rhs) const;
    [[nodiscard]] std::vector<double> solve_transpose(const std::vector<double>& rhs) const;
    [[nodiscard]] const LuDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }

  private:
    std::size_t dimension_{};
    std::vector<double> lu_;
    std::vector<std::size_t> pivots_;
    LuDiagnostics diagnostics_;
};
[[nodiscard]] std::vector<double> multiply(const DenseMatrix&, const std::vector<double>&);
[[nodiscard]] std::vector<double> multiply_transpose(const DenseMatrix&,
                                                     const std::vector<double>&);
[[nodiscard]] double infinity_residual(const DenseMatrix&, const std::vector<double>& x,
                                       const std::vector<double>& rhs, bool transpose = false);
} // namespace markov_cero::linalg
