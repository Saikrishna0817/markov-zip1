#pragma once
#include <cstddef>
#include <utility>
#include <vector>
namespace markov_cero::linalg {
struct SparseCsc {
  std::size_t rows{};
  std::size_t columns{};
  std::vector<std::size_t> column_offsets;
  std::vector<std::size_t> row_indices;
  std::vector<double> values;
  void validate(std::size_t maximum_nonzeros=4U*1024U*1024U) const;
  [[nodiscard]] std::vector<double> dense_column(std::size_t column) const;
  [[nodiscard]] static SparseCsc from_columns(std::size_t rows,const std::vector<std::vector<double>>& columns);
};
struct SparseLuDiagnostics {
  std::size_t lower_nonzeros{};
  std::size_t upper_nonzeros{};
  std::size_t factor_nonzeros{};
  double minimum_absolute_pivot{};
  double maximum_absolute_pivot{};
  double growth_factor{};
};
class SparseLu final {
 public:
  static SparseLu factorize(const SparseCsc& matrix,double singular_tolerance=1e-14,std::size_t maximum_factor_nonzeros=4U*1024U*1024U);
  [[nodiscard]] std::vector<double> solve(const std::vector<double>& rhs) const;
  [[nodiscard]] std::vector<double> solve_transpose(const std::vector<double>& rhs) const;
  [[nodiscard]] const SparseLuDiagnostics& diagnostics() const noexcept { return diagnostics_; }
  [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
 private:
  using Entry=std::pair<std::size_t,double>;
  std::size_t dimension_{};
  std::vector<std::vector<Entry>> lower_rows_;
  std::vector<std::vector<Entry>> upper_rows_;
  std::vector<std::vector<Entry>> lower_columns_;
  std::vector<std::vector<Entry>> upper_columns_;
  std::vector<std::size_t> row_order_;
  SparseLuDiagnostics diagnostics_;
};
struct SparseBasisOptions {
  double singular_tolerance{1e-14};
  double update_pivot_tolerance{1e-12};
  double eta_density_trigger{0.5};
  std::size_t maximum_updates{64};
  std::size_t maximum_dimension{4096};
  std::size_t maximum_nonzeros{4U*1024U*1024U};
  std::size_t maximum_factor_nonzeros{4U*1024U*1024U};
};
struct SparseBasisStatistics {
  std::size_t refactorizations{};
  std::size_t updates{};
  std::size_t current_update_chain{};
  std::size_t last_rhs_nonzeros{};
  std::size_t last_solution_nonzeros{};
  std::size_t maximum_eta_nonzeros{};
  bool update_limit_triggered{};
  bool density_triggered{};
};
class SparseBasisFactorization final {
 public:
  static SparseBasisFactorization factorize(const SparseCsc& basis,const SparseBasisOptions& options={});
  [[nodiscard]] std::vector<double> solve(const std::vector<double>& rhs);
  [[nodiscard]] std::vector<double> solve_transpose(const std::vector<double>& rhs);
  void replace_column(std::size_t position,const std::vector<double>& column);
  [[nodiscard]] bool needs_refactorization() const noexcept;
  void refactorize();
  [[nodiscard]] const SparseBasisStatistics& statistics() const noexcept { return statistics_; }
  [[nodiscard]] const SparseLuDiagnostics& diagnostics() const noexcept { return base_.diagnostics(); }
  [[nodiscard]] const SparseCsc& current_basis() const noexcept { return current_basis_; }
 private:
  struct Eta { std::size_t pivot{}; double pivot_value{}; std::vector<std::pair<std::size_t,double>> entries; };
  SparseBasisOptions options_;
  SparseCsc current_basis_;
  SparseLu base_;
  std::vector<Eta> updates_;
  SparseBasisStatistics statistics_;
};
[[nodiscard]] double sparse_infinity_residual(const SparseCsc& matrix,const std::vector<double>& x,const std::vector<double>& rhs,bool transpose=false);
}
