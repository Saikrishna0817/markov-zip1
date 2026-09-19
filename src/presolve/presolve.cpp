#include "markov_cero/presolve/presolve.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace markov_cero::presolve {
namespace {

void ensure_finite(double v, const char* message) {
    if (!std::isfinite(v)) {
        throw std::overflow_error(message);
    }
}

} // namespace

PresolveResult presolve(const transform::SparseCanonicalModel& input,
                         const PresolveOptions& options) {
    input.validate();

    PresolveResult result;
    result.status = lp::reference::SolveStatus::optimal;
    result.statistics.original_rows = input.matrix.rows;
    result.statistics.original_cols = input.matrix.columns;

    const std::size_t m = input.matrix.rows;
    const std::size_t n = input.matrix.columns;

    std::vector<bool> row_active(m, true);
    std::vector<bool> col_active(n, true);

    struct ColEntry {
        std::size_t row;
        double val;
    };
    std::vector<std::vector<ColEntry>> cols(n);
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t start = input.matrix.column_offsets[j];
        const std::size_t end = input.matrix.column_offsets[j + 1];
        for (std::size_t k = start; k < end; ++k) {
            cols[j].push_back({input.matrix.row_indices[k], input.matrix.values[k]});
        }
    }

    struct RowEntry {
        std::size_t col;
        double val;
    };
    std::vector<std::vector<RowEntry>> rows(m);
    for (std::size_t j = 0; j < n; ++j) {
        for (const auto& ce : cols[j]) {
            rows[ce.row].push_back({j, ce.val});
        }
    }

    std::vector<double> rhs = input.rhs;
    std::vector<double> obj = input.objective;
    double obj_offset = input.objective_offset;

    std::size_t pass = 0;
    for (; pass < options.max_passes; ++pass) {
        std::size_t reductions = 0;

        // 1. Scan for empty rows
        for (std::size_t i = 0; i < m; ++i) {
            if (!row_active[i]) {
                continue;
            }
            auto& r_entries = rows[i];
            r_entries.erase(
                std::remove_if(r_entries.begin(), r_entries.end(),
                               [&](const RowEntry& e) {
                                   return !col_active[e.col] || std::abs(e.val) <= options.pivot_tolerance;
                               }),
                r_entries.end());

            if (r_entries.empty()) {
                if (std::abs(rhs[i]) > options.feasibility_tolerance) {
                    result.status = lp::reference::SolveStatus::infeasible;
                    result.message = "presolve: empty row with non-zero RHS detected";
                    return result;
                }
                row_active[i] = false;
                result.stack.push(EmptyRowRecord{i, rhs[i]});
                ++result.statistics.empty_rows_removed;
                ++reductions;
            }
        }

        // 2. Scan for empty columns
        for (std::size_t j = 0; j < n; ++j) {
            if (!col_active[j]) {
                continue;
            }
            auto& c_entries = cols[j];
            c_entries.erase(
                std::remove_if(c_entries.begin(), c_entries.end(),
                               [&](const ColEntry& e) {
                                   return !row_active[e.row] || std::abs(e.val) <= options.pivot_tolerance;
                               }),
                c_entries.end());

            if (c_entries.empty()) {
                if (obj[j] < -options.dual_tolerance) {
                    result.status = lp::reference::SolveStatus::unbounded;
                    result.message = "presolve: empty column with negative cost detected";
                    return result;
                }
                col_active[j] = false;
                result.stack.push(EmptyColumnRecord{j, obj[j], 0.0});
                ++result.statistics.empty_cols_removed;
                ++reductions;
            }
        }

        // 3. Scan for row singletons
        for (std::size_t i = 0; i < m; ++i) {
            if (!row_active[i]) {
                continue;
            }
            auto& r_entries = rows[i];
            r_entries.erase(
                std::remove_if(r_entries.begin(), r_entries.end(),
                               [&](const RowEntry& e) {
                                   return !col_active[e.col] || std::abs(e.val) <= options.pivot_tolerance;
                               }),
                r_entries.end());

            if (r_entries.size() == 1) {
                const auto entry = r_entries[0];
                const std::size_t j = entry.col;
                const double a_ij = entry.val;
                double fixed_val = rhs[i] / a_ij;

                if (fixed_val < -options.feasibility_tolerance) {
                    result.status = lp::reference::SolveStatus::infeasible;
                    result.message = "presolve: row singleton implies negative value for canonical non-negative variable";
                    return result;
                }
                if (fixed_val < 0.0) {
                    fixed_val = 0.0;
                }

                result.stack.push(RowSingletonRecord{i, j, a_ij, rhs[i]});
                row_active[i] = false;
                ++result.statistics.row_singletons_removed;
                ++reductions;

                // Fix variable j and substitute into other incident rows
                std::vector<std::size_t> inc_rows;
                std::vector<double> inc_coeffs;
                for (const auto& ce : cols[j]) {
                    if (row_active[ce.row]) {
                        rhs[ce.row] -= ce.val * fixed_val;
                        inc_rows.push_back(ce.row);
                        inc_coeffs.push_back(ce.val);
                    }
                }
                obj_offset += obj[j] * fixed_val;
                result.stack.push(FixedVariableRecord{j, fixed_val, obj[j], std::move(inc_rows), std::move(inc_coeffs)});
                col_active[j] = false;
                ++result.statistics.fixed_vars_removed;
            }
        }

        if (reductions == 0) {
            break;
        }
    }
    result.statistics.passes_executed = pass;

    // Compact remaining active rows and columns
    std::vector<std::size_t> presolved_to_orig_row;
    std::vector<std::size_t> orig_to_presolved_row(m, static_cast<std::size_t>(-1));
    for (std::size_t i = 0; i < m; ++i) {
        if (row_active[i]) {
            orig_to_presolved_row[i] = presolved_to_orig_row.size();
            presolved_to_orig_row.push_back(i);
        }
    }

    std::vector<std::size_t> presolved_to_orig_col;
    std::vector<std::size_t> orig_to_presolved_col(n, static_cast<std::size_t>(-1));
    for (std::size_t j = 0; j < n; ++j) {
        if (col_active[j]) {
            orig_to_presolved_col[j] = presolved_to_orig_col.size();
            presolved_to_orig_col.push_back(j);
        }
    }

    result.stack.set_row_map(presolved_to_orig_row);
    result.stack.set_col_map(presolved_to_orig_col);

    const std::size_t new_m = presolved_to_orig_row.size();
    const std::size_t new_n = presolved_to_orig_col.size();
    result.statistics.presolved_rows = new_m;
    result.statistics.presolved_cols = new_n;

    // Build the compacted model
    result.model.matrix.rows = new_m;
    result.model.matrix.columns = new_n;
    result.model.rhs.resize(new_m);
    for (std::size_t new_i = 0; new_i < new_m; ++new_i) {
        result.model.rhs[new_i] = rhs[presolved_to_orig_row[new_i]];
    }

    result.model.objective.resize(new_n);
    for (std::size_t new_j = 0; new_j < new_n; ++new_j) {
        result.model.objective[new_j] = obj[presolved_to_orig_col[new_j]];
    }

    result.model.objective_offset = obj_offset;
    result.model.record = input.record;

    result.model.matrix.column_offsets.assign(new_n + 1, 0);
    std::size_t offset_count = 0;
    for (std::size_t new_j = 0; new_j < new_n; ++new_j) {
        result.model.matrix.column_offsets[new_j] = offset_count;
        const std::size_t orig_j = presolved_to_orig_col[new_j];
        for (const auto& ce : cols[orig_j]) {
            if (row_active[ce.row] && std::abs(ce.val) > options.pivot_tolerance) {
                const std::size_t new_i = orig_to_presolved_row[ce.row];
                result.model.matrix.row_indices.push_back(new_i);
                result.model.matrix.values.push_back(ce.val);
                ++offset_count;
            }
        }
    }
    result.model.matrix.column_offsets[new_n] = offset_count;

    if (new_m > 0 && new_n > 0) {
        result.model.validate();
    }
    result.message = "presolve complete";
    return result;
}

lp::reference::Result postsolve(
    const PresolveStack& stack,
    const lp::reference::Result& reduced_solution,
    const transform::SparseCanonicalModel& original_model,
    double /*tolerance*/) {

    lp::reference::Result restored = reduced_solution;
    const std::size_t m = original_model.matrix.rows;
    const std::size_t n = original_model.matrix.columns;

    restored.primal.assign(n, 0.0);
    restored.dual.assign(m, 0.0);

    // 1. Copy over un-eliminated reduced solution values
    const auto& col_map = stack.presolved_to_original_cols();
    for (std::size_t new_j = 0; new_j < col_map.size(); ++new_j) {
        if (new_j < reduced_solution.primal.size()) {
            restored.primal[col_map[new_j]] = reduced_solution.primal[new_j];
        }
    }

    const auto& row_map = stack.presolved_to_original_rows();
    for (std::size_t new_i = 0; new_i < row_map.size(); ++new_i) {
        if (new_i < reduced_solution.dual.size()) {
            restored.dual[row_map[new_i]] = reduced_solution.dual[new_i];
        }
    }

    // 2. Pop reduction records in reverse (LIFO) order
    const auto& records = stack.records();
    for (auto it = records.rbegin(); it != records.rend(); ++it) {
        std::visit([&](const auto& rec) {
            using T = std::decay_t<decltype(rec)>;
            if constexpr (std::is_same_v<T, EmptyRowRecord>) {
                restored.dual[rec.original_row_index] = 0.0;
            } else if constexpr (std::is_same_v<T, EmptyColumnRecord>) {
                restored.primal[rec.original_col_index] = rec.fixed_value;
            } else if constexpr (std::is_same_v<T, FixedVariableRecord>) {
                restored.primal[rec.original_col_index] = rec.fixed_value;
            } else if constexpr (std::is_same_v<T, RowSingletonRecord>) {
                // Variable k was fixed by row i: a_{ik} x_k = b_i
                // Satisfy dual optimality: pi_i = (c_k - sum_{r != i} a_{rk} pi_r) / a_{ik}
                const std::size_t i = rec.original_row_index;
                const std::size_t k = rec.variable_index;
                const double a_ik = rec.coefficient;
                const double c_k = original_model.objective[k];

                long double sum_other = 0.0;
                const std::size_t c_start = original_model.matrix.column_offsets[k];
                const std::size_t c_end = original_model.matrix.column_offsets[k + 1];
                for (std::size_t p = c_start; p < c_end; ++p) {
                    const std::size_t r = original_model.matrix.row_indices[p];
                    if (r != i) {
                        sum_other += static_cast<long double>(original_model.matrix.values[p]) * restored.dual[r];
                    }
                }
                const double pi_i = static_cast<double>((c_k - sum_other) / a_ik);
                ensure_finite(pi_i, "non-finite dual multiplier in postsolve");
                restored.dual[i] = pi_i;
            }
        }, *it);
    }

    // 3. Recompute exact objective value in original canonical space
    long double exact_obj = original_model.objective_offset;
    for (std::size_t j = 0; j < n; ++j) {
        exact_obj += static_cast<long double>(original_model.objective[j]) * restored.primal[j];
    }
    restored.objective = static_cast<double>(exact_obj);
    ensure_finite(restored.objective, "non-finite objective in postsolve");

    return restored;
}

} // namespace markov_cero::presolve
