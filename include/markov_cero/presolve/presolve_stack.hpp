#pragma once

#include <cstddef>
#include <variant>
#include <vector>

namespace markov_cero::presolve {

struct EmptyRowRecord {
    std::size_t original_row_index;
    double rhs_value{0.0};
};

struct EmptyColumnRecord {
    std::size_t original_col_index;
    double objective_coeff{0.0};
    double fixed_value{0.0};
};

struct FixedVariableRecord {
    std::size_t original_col_index;
    double fixed_value{0.0};
    double objective_coeff{0.0};
    std::vector<std::size_t> row_indices;
    std::vector<double> coefficients;
};

struct RowSingletonRecord {
    std::size_t original_row_index;
    std::size_t variable_index;
    double coefficient{0.0};
    double rhs_value{0.0};
};

using ReductionRecord = std::variant<
    EmptyRowRecord,
    EmptyColumnRecord,
    FixedVariableRecord,
    RowSingletonRecord
>;

class PresolveStack {
public:
    void push(ReductionRecord record) {
        records_.push_back(std::move(record));
    }
    [[nodiscard]] std::size_t size() const noexcept {
        return records_.size();
    }
    [[nodiscard]] bool empty() const noexcept {
        return records_.empty();
    }
    [[nodiscard]] const std::vector<ReductionRecord>& records() const noexcept {
        return records_;
    }
    void set_col_map(std::vector<std::size_t> map) {
        presolved_to_original_cols_ = std::move(map);
    }
    void set_row_map(std::vector<std::size_t> map) {
        presolved_to_original_rows_ = std::move(map);
    }
    [[nodiscard]] const std::vector<std::size_t>& presolved_to_original_cols() const noexcept {
        return presolved_to_original_cols_;
    }
    [[nodiscard]] const std::vector<std::size_t>& presolved_to_original_rows() const noexcept {
        return presolved_to_original_rows_;
    }
    void clear() noexcept {
        records_.clear();
        presolved_to_original_cols_.clear();
        presolved_to_original_rows_.clear();
    }
private:
    std::vector<ReductionRecord> records_;
    std::vector<std::size_t> presolved_to_original_cols_;
    std::vector<std::size_t> presolved_to_original_rows_;
};

} // namespace markov_cero::presolve
