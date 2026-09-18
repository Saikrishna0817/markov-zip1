#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sihopt::model {

enum class ObjectiveSense { minimize, maximize };
enum class VariableType { continuous, integer, binary };
enum class BoundKind { negative_infinity, finite, positive_infinity };

struct Bound final {
    BoundKind kind{BoundKind::finite};
    double value{0.0};

    [[nodiscard]] static Bound negative_infinity() noexcept;
    [[nodiscard]] static Bound finite(double value);
    [[nodiscard]] static Bound positive_infinity() noexcept;
    [[nodiscard]] bool is_finite() const noexcept;
};

struct SparseMatrixCSC final {
    std::size_t row_count{};
    std::size_t column_count{};
    std::vector<std::size_t> column_start;
    std::vector<std::size_t> row_index;
    std::vector<double> value;

    void validate() const;
    [[nodiscard]] std::vector<double> multiply(const std::vector<double>& x) const;
};

class SparseMatrixBuilder final {
  public:
    SparseMatrixBuilder(std::size_t row_count, std::size_t column_count);
    void add(std::size_t row, std::size_t column, double value);
    [[nodiscard]] SparseMatrixCSC build() const;

  private:
    std::size_t row_count_;
    std::size_t column_count_;
    struct Entry final { std::size_t row; std::size_t column; double value; };
    std::vector<Entry> entries_;
};

struct Model final {
    std::string name;
    ObjectiveSense objective_sense{ObjectiveSense::minimize};
    double objective_offset{0.0};
    SparseMatrixCSC matrix;
    std::vector<double> objective;
    std::vector<Bound> row_lower;
    std::vector<Bound> row_upper;
    std::vector<Bound> variable_lower;
    std::vector<Bound> variable_upper;
    std::vector<VariableType> variable_type;
    std::vector<std::string> row_name;
    std::vector<std::string> variable_name;

    void validate() const;
};

[[nodiscard]] const char* to_string(ObjectiveSense sense) noexcept;
[[nodiscard]] const char* to_string(VariableType type) noexcept;

} // namespace sihopt::model
