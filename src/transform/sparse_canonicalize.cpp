#include "markov_cero/transform/sparse_canonical_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace markov_cero::transform {
namespace {

void ensure_finite(double v, const char* message) {
    if (!std::isfinite(v)) {
        throw std::overflow_error(message);
    }
}

} // namespace

void SparseCanonicalModel::validate() const {
    if (matrix.rows != rhs.size() || matrix.columns != objective.size()) {
        throw std::invalid_argument("sparse canonical dimensions disagree");
    }
    if (record.structural_variables > matrix.columns) {
        throw std::invalid_argument("structural variable count exceeds canonical columns");
    }
    if (record.objective_sign != 1.0 && record.objective_sign != -1.0) {
        throw std::invalid_argument("objective sign must be plus or minus one");
    }
    ensure_finite(objective_offset, "canonical offset non-finite");
    matrix.validate();
    for (double v : rhs) {
        ensure_finite(v, "canonical rhs non-finite");
    }
    for (double v : objective) {
        ensure_finite(v, "canonical objective non-finite");
    }
    for (const auto& m : record.variables) {
        ensure_finite(m.offset, "non-finite variable-map offset");
        if (m.canonical_index.size() != m.multiplier.size()) {
            throw std::invalid_argument("variable-map dimension mismatch");
        }
        for (std::size_t q = 0; q < m.canonical_index.size(); ++q) {
            if (m.canonical_index[q] >= record.structural_variables ||
                m.canonical_index[q] >= matrix.columns) {
                throw std::invalid_argument("variable-map index out of range");
            }
            ensure_finite(m.multiplier[q], "non-finite variable-map multiplier");
        }
    }
}

std::vector<double> SparseCanonicalModel::multiply(const std::vector<double>& x) const {
    if (x.size() != matrix.columns) {
        throw std::invalid_argument("dimension mismatch in sparse canonical multiply");
    }
    std::vector<double> y(matrix.rows, 0.0);
    for (std::size_t j = 0; j < matrix.columns; ++j) {
        const double xj = x[j];
        if (xj == 0.0) {
            continue;
        }
        const std::size_t start = matrix.column_offsets[j];
        const std::size_t end = matrix.column_offsets[j + 1];
        for (std::size_t k = start; k < end; ++k) {
            y[matrix.row_indices[k]] += matrix.values[k] * xj;
        }
    }
    for (double v : y) {
        ensure_finite(v, "non-finite result in sparse canonical multiply");
    }
    return y;
}

std::vector<double> SparseCanonicalModel::multiply_transpose(const std::vector<double>& y) const {
    if (y.size() != matrix.rows) {
        throw std::invalid_argument("dimension mismatch in sparse canonical multiply_transpose");
    }
    std::vector<double> x(matrix.columns, 0.0);
    for (std::size_t j = 0; j < matrix.columns; ++j) {
        long double s = 0.0;
        const std::size_t start = matrix.column_offsets[j];
        const std::size_t end = matrix.column_offsets[j + 1];
        for (std::size_t k = start; k < end; ++k) {
            s += static_cast<long double>(matrix.values[k]) * y[matrix.row_indices[k]];
        }
        const double val = static_cast<double>(s);
        ensure_finite(val, "non-finite result in sparse canonical multiply_transpose");
        x[j] = val;
    }
    return x;
}

CanonicalModel SparseCanonicalModel::to_dense() const {
    constexpr std::size_t max_dense_rows = 2048;
    constexpr std::size_t max_dense_cols = 8192;
    if (matrix.rows > max_dense_rows || matrix.columns > max_dense_cols) {
        throw std::length_error("sparse model exceeds dense dimension limits");
    }
    CanonicalModel d;
    d.matrix.rows = matrix.rows;
    d.matrix.columns = matrix.columns;
    d.matrix.values.assign(matrix.rows * matrix.columns, 0.0);
    for (std::size_t j = 0; j < matrix.columns; ++j) {
        const std::size_t start = matrix.column_offsets[j];
        const std::size_t end = matrix.column_offsets[j + 1];
        for (std::size_t k = start; k < end; ++k) {
            d.matrix(matrix.row_indices[k], j) += matrix.values[k];
        }
    }
    d.rhs = rhs;
    d.objective = objective;
    d.objective_offset = objective_offset;
    d.record = record;
    d.validate();
    return d;
}

SparseCanonicalModel sparse_canonicalize(const model::Model& in, bool relax_integrality) {
    in.validate();
    if (!relax_integrality) {
        for (const auto type : in.variable_type) {
            if (type != model::VariableType::continuous) {
                throw std::invalid_argument("continuous variables only supported");
            }
        }
    }

    SparseCanonicalModel out;
    out.original_variable_types = in.variable_type;
    out.record.variables.resize(in.matrix.column_count);
    out.record.objective_sign = in.objective_sense == model::ObjectiveSense::minimize ? 1.0 : -1.0;

    std::vector<double> offset(in.matrix.column_count, 0.0);
    std::size_t structural = 0;
    for (std::size_t j = 0; j < in.matrix.column_count; ++j) {
        auto& m = out.record.variables[j];
        const auto lo = in.variable_lower[j];
        const auto up = in.variable_upper[j];
        if (lo.is_finite() && up.is_finite() && lo.value == up.value) {
            m.offset = lo.value;
            offset[j] = lo.value;
            continue;
        }
        m.offset = lo.is_finite() ? lo.value : (up.is_finite() ? up.value : 0.0);
        offset[j] = m.offset;

        auto add = [&](double sign) {
            m.canonical_index.push_back(structural++);
            m.multiplier.push_back(sign);
        };
        if (lo.is_finite()) {
            add(1.0);
        } else if (up.is_finite()) {
            add(-1.0);
        } else {
            add(1.0);
            add(-1.0);
        }
    }
    out.record.structural_variables = structural;

    // Transpose input matrix into row-oriented adjacency for efficient row-wise canonicalization
    struct RowEntry {
        std::size_t col;
        double val;
    };
    std::vector<std::vector<RowEntry>> row_entries(in.matrix.row_count);
    for (std::size_t j = 0; j < in.matrix.column_count; ++j) {
        const std::size_t c_start = in.matrix.column_start[j];
        const std::size_t c_end = in.matrix.column_start[j + 1];
        for (std::size_t p = c_start; p < c_end; ++p) {
            row_entries[in.matrix.row_index[p]].push_back({j, in.matrix.value[p]});
        }
    }

    struct Triplet {
        std::size_t r;
        std::size_t c;
        double v;
    };
    std::vector<Triplet> triplets;
    std::size_t next_col = structural;

    auto add_canonical_row = [&](std::size_t orig_row, double sign, double bound, bool need_slack) {
        const std::size_t r = out.rhs.size();
        long double shift = 0.0;
        for (const auto& entry : row_entries[orig_row]) {
            shift += static_cast<long double>(entry.val) * offset[entry.col];
            const auto& vmap = out.record.variables[entry.col];
            for (std::size_t q = 0; q < vmap.canonical_index.size(); ++q) {
                const double coeff = sign * entry.val * vmap.multiplier[q];
                if (coeff != 0.0) {
                    triplets.push_back({r, vmap.canonical_index[q], coeff});
                }
            }
        }
        if (need_slack) {
            triplets.push_back({r, next_col++, 1.0});
        }
        const double b = sign * (bound - static_cast<double>(shift));
        ensure_finite(b, "non-finite transformed rhs");
        out.rhs.push_back(b);
    };

    // 1. Structural constraints
    for (std::size_t i = 0; i < in.matrix.row_count; ++i) {
        const auto lo = in.row_lower[i];
        const auto up = in.row_upper[i];
        if (lo.is_finite() && up.is_finite() && lo.value == up.value) {
            add_canonical_row(i, 1.0, lo.value, false);
        } else {
            if (up.is_finite()) {
                add_canonical_row(i, 1.0, up.value, true);
            }
            if (lo.is_finite()) {
                add_canonical_row(i, -1.0, lo.value, true);
            }
        }
    }

    // 2. Box bound constraints on structural variables
    for (std::size_t j = 0; j < in.matrix.column_count; ++j) {
        const auto lo = in.variable_lower[j];
        const auto up = in.variable_upper[j];
        if (lo.is_finite() && up.is_finite() && lo.value != up.value) {
            const std::size_t r = out.rhs.size();
            const auto& vmap = out.record.variables[j];
            for (std::size_t q = 0; q < vmap.canonical_index.size(); ++q) {
                triplets.push_back({r, vmap.canonical_index[q], vmap.multiplier[q]});
            }
            triplets.push_back({r, next_col++, 1.0});
            out.rhs.push_back(up.value - lo.value);
        }
    }

    // 3. Objective vector and constant offset
    const std::size_t total_cols = next_col;
    out.objective.assign(total_cols, 0.0);
    long double obj_shift = 0.0;
    for (std::size_t j = 0; j < in.matrix.column_count; ++j) {
        obj_shift += static_cast<long double>(in.objective[j]) * offset[j];
        const auto& vmap = out.record.variables[j];
        for (std::size_t q = 0; q < vmap.canonical_index.size(); ++q) {
            out.objective[vmap.canonical_index[q]] +=
                out.record.objective_sign * in.objective[j] * vmap.multiplier[q];
        }
    }
    out.objective_offset = out.record.objective_sign * static_cast<double>(obj_shift) + in.objective_offset;

    // 4. Assemble SparseCsc matrix from triplets
    const std::size_t total_rows = out.rhs.size();
    out.matrix.rows = total_rows;
    out.matrix.columns = total_cols;

    std::vector<std::vector<std::pair<std::size_t, double>>> col_entries(total_cols);
    for (const auto& t : triplets) {
        col_entries[t.c].push_back({t.r, t.v});
    }

    out.matrix.column_offsets.assign(total_cols + 1, 0);
    std::size_t current_offset = 0;
    for (std::size_t j = 0; j < total_cols; ++j) {
        out.matrix.column_offsets[j] = current_offset;
        // Combine entries with identical row index in the same column
        auto& entries = col_entries[j];
        if (entries.size() > 1) {
            std::sort(entries.begin(), entries.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            std::size_t write_idx = 0;
            for (std::size_t read_idx = 1; read_idx < entries.size(); ++read_idx) {
                if (entries[read_idx].first == entries[write_idx].first) {
                    entries[write_idx].second += entries[read_idx].second;
                } else {
                    if (entries[write_idx].second != 0.0) {
                        ++write_idx;
                    }
                    entries[write_idx] = entries[read_idx];
                }
            }
            if (!entries.empty() && entries[write_idx].second != 0.0) {
                entries.resize(write_idx + 1);
            } else if (!entries.empty()) {
                entries.resize(write_idx);
            }
        }
        for (const auto& e : entries) {
            if (e.second != 0.0) {
                out.matrix.row_indices.push_back(e.first);
                out.matrix.values.push_back(e.second);
                ++current_offset;
            }
        }
    }
    out.matrix.column_offsets[total_cols] = current_offset;

    out.validate();
    return out;
}

std::vector<double> reconstruct_primal(const SparseCanonicalModel& model,
                                        const std::vector<double>& canonical_primal) {
    if (canonical_primal.size() != model.matrix.columns) {
        throw std::invalid_argument("canonical primal size does not match model columns");
    }
    std::vector<double> out(model.record.variables.size(), 0.0);
    for (std::size_t j = 0; j < model.record.variables.size(); ++j) {
        const auto& map = model.record.variables[j];
        long double val = map.offset;
        for (std::size_t q = 0; q < map.canonical_index.size(); ++q) {
            val += static_cast<long double>(map.multiplier[q]) * canonical_primal[map.canonical_index[q]];
        }
        const double dval = static_cast<double>(val);
        ensure_finite(dval, "non-finite reconstructed primal variable");
        out[j] = dval;
    }
    return out;
}

double reconstruct_objective(const SparseCanonicalModel& model, double canonical_objective) {
    ensure_finite(canonical_objective, "canonical objective non-finite");
    const double out = model.record.objective_sign * canonical_objective;
    ensure_finite(out, "reconstructed objective non-finite");
    return out;
}

} // namespace markov_cero::transform
