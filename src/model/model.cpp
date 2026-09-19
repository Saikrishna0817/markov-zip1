#include "markov_cero/model/model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace markov_cero::model {
namespace {
void require_finite(double value, const char* what) {
    if (!std::isfinite(value)) throw std::invalid_argument(std::string(what) + " must be finite");
}
void validate_bound(const Bound& bound, const char* what) {
    if (bound.kind == BoundKind::finite) require_finite(bound.value, what);
}
void validate_names(const std::vector<std::string>& names, std::size_t expected, const char* what) {
    if (names.size() != expected) throw std::invalid_argument(std::string(what) + " count mismatch");
    std::set<std::string> seen;
    for (const auto& name : names) {
        if (name.empty()) throw std::invalid_argument(std::string(what) + " cannot be empty");
        if (!seen.insert(name).second) throw std::invalid_argument(std::string("duplicate ") + what + ": " + name);
    }
}
}

Bound Bound::negative_infinity() noexcept { return {BoundKind::negative_infinity, 0.0}; }
Bound Bound::finite(double value) { require_finite(value, "bound"); return {BoundKind::finite, value}; }
Bound Bound::positive_infinity() noexcept { return {BoundKind::positive_infinity, 0.0}; }
bool Bound::is_finite() const noexcept { return kind == BoundKind::finite; }

void SparseMatrixCSC::validate() const {
    if (column_start.size() != column_count + 1U) throw std::invalid_argument("CSC column_start size mismatch");
    if (column_start.empty() || column_start.front() != 0U) throw std::invalid_argument("CSC must start at zero");
    if (row_index.size() != value.size()) throw std::invalid_argument("CSC index/value size mismatch");
    if (column_start.back() != value.size()) throw std::invalid_argument("CSC terminal pointer mismatch");
    for (std::size_t c = 0; c < column_count; ++c) {
        if (column_start[c] > column_start[c + 1U]) throw std::invalid_argument("CSC pointers must be monotone");
        std::size_t previous = 0U;
        bool has_previous = false;
        for (std::size_t k = column_start[c]; k < column_start[c + 1U]; ++k) {
            if (row_index[k] >= row_count) throw std::invalid_argument("CSC row index out of range");
            if (has_previous && row_index[k] <= previous) throw std::invalid_argument("CSC rows must be strictly increasing per column");
            if (!std::isfinite(value[k]) || value[k] == 0.0) throw std::invalid_argument("CSC values must be finite and nonzero");
            previous = row_index[k]; has_previous = true;
        }
    }
}

std::vector<double> SparseMatrixCSC::multiply(const std::vector<double>& x) const {
    validate();
    if (x.size() != column_count) throw std::invalid_argument("matrix/vector dimension mismatch");
    std::vector<long double> work(row_count, 0.0L);
    for (std::size_t c = 0; c < column_count; ++c) {
        if (!std::isfinite(x[c])) throw std::invalid_argument("vector contains non-finite value");
        for (std::size_t k = column_start[c]; k < column_start[c + 1U]; ++k) work[row_index[k]] += static_cast<long double>(value[k]) * x[c];
    }
    std::vector<double> result(row_count);
    for (std::size_t r = 0; r < row_count; ++r) {
        result[r] = static_cast<double>(work[r]);
        if (!std::isfinite(result[r])) throw std::overflow_error("matrix/vector product is non-finite");
    }
    return result;
}

SparseMatrixBuilder::SparseMatrixBuilder(std::size_t rows, std::size_t columns) : row_count_(rows), column_count_(columns) {}
void SparseMatrixBuilder::add(std::size_t row, std::size_t column, double value) {
    if (row >= row_count_ || column >= column_count_) throw std::out_of_range("sparse entry index out of range");
    require_finite(value, "sparse value");
    entries_.push_back({row, column, value});
}
SparseMatrixCSC SparseMatrixBuilder::build() const {
    std::map<std::pair<std::size_t, std::size_t>, long double> aggregated;
    for (const auto& entry : entries_) aggregated[{entry.column, entry.row}] += entry.value;
    SparseMatrixCSC matrix; matrix.row_count = row_count_; matrix.column_count = column_count_;
    matrix.column_start.assign(column_count_ + 1U, 0U);
    for (const auto& [key, sum] : aggregated) {
        const double value = static_cast<double>(sum);
        if (!std::isfinite(value)) throw std::overflow_error("aggregated sparse value is non-finite");
        if (value == 0.0) continue;
        matrix.row_index.push_back(key.second); matrix.value.push_back(value); ++matrix.column_start[key.first + 1U];
    }
    for (std::size_t c = 0; c < column_count_; ++c) matrix.column_start[c + 1U] += matrix.column_start[c];
    matrix.validate(); return matrix;
}

void Model::validate() const {
    matrix.validate();
    const auto rows = matrix.row_count; const auto columns = matrix.column_count;
    if (!std::isfinite(objective_offset)) throw std::invalid_argument("objective offset must be finite");
    if (objective.size() != columns || variable_lower.size() != columns || variable_upper.size() != columns || variable_type.size() != columns) throw std::invalid_argument("variable metadata dimension mismatch");
    if (row_lower.size() != rows || row_upper.size() != rows) throw std::invalid_argument("row metadata dimension mismatch");
    validate_names(row_name, rows, "row name"); validate_names(variable_name, columns, "variable name");
    for (std::size_t j = 0; j < columns; ++j) {
        require_finite(objective[j], "objective coefficient"); validate_bound(variable_lower[j], "variable lower bound"); validate_bound(variable_upper[j], "variable upper bound");
        if (variable_lower[j].kind == BoundKind::positive_infinity || variable_upper[j].kind == BoundKind::negative_infinity) throw std::invalid_argument("invalid variable infinity direction");
        if (variable_lower[j].is_finite() && variable_upper[j].is_finite() && variable_lower[j].value > variable_upper[j].value) throw std::invalid_argument("inconsistent variable bounds");
        if (variable_type[j] == VariableType::binary) {
            if (!variable_lower[j].is_finite() || !variable_upper[j].is_finite()) throw std::invalid_argument("binary bounds must be finite");
            if (variable_lower[j].value < 0.0 || variable_upper[j].value > 1.0) throw std::invalid_argument("binary bounds must stay within zero and one");
        }
    }
    for (std::size_t i = 0; i < rows; ++i) {
        validate_bound(row_lower[i], "row lower bound"); validate_bound(row_upper[i], "row upper bound");
        if (row_lower[i].kind == BoundKind::positive_infinity || row_upper[i].kind == BoundKind::negative_infinity) throw std::invalid_argument("invalid row infinity direction");
        if (row_lower[i].is_finite() && row_upper[i].is_finite() && row_lower[i].value > row_upper[i].value) throw std::invalid_argument("inconsistent row bounds");
    }
}

const char* to_string(ObjectiveSense sense) noexcept { return sense == ObjectiveSense::minimize ? "minimize" : "maximize"; }
const char* to_string(VariableType type) noexcept {
    switch (type) { case VariableType::continuous: return "continuous"; case VariableType::integer: return "integer"; case VariableType::binary: return "binary"; }
    return "unknown";
}
}
