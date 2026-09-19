#include "markov_cero/milp/cuts.hpp"

#include "markov_cero/linalg/dense_lu.hpp"
#include "markov_cero/linalg/sparse_basis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace markov_cero::milp {
namespace {

linalg::SparseCsc extract_basis_matrix(const transform::SparseCanonicalModel& model,
                                      const std::vector<std::size_t>& basis) {
    const std::size_t m = model.matrix.rows;
    std::vector<std::vector<double>> cols;
    cols.reserve(basis.size());
    for (std::size_t col_idx : basis) {
        std::vector<double> col(m, 0.0);
        const std::size_t start = model.matrix.column_offsets[col_idx];
        const std::size_t end = model.matrix.column_offsets[col_idx + 1];
        for (std::size_t p = start; p < end; ++p) {
            col[model.matrix.row_indices[p]] = model.matrix.values[p];
        }
        cols.push_back(std::move(col));
    }
    return linalg::SparseCsc::from_columns(m, cols);
}

} // namespace

std::vector<Cut> generate_gomory_cuts(
    const model::Model& model,
    const std::vector<double>& original_primal,
    const transform::SparseCanonicalModel& canonical,
    const lp::dual::BasisState& basis_state,
    std::size_t max_cuts,
    double min_fractionality) {
    std::vector<Cut> cuts;
    if (basis_state.basic_variables.size() != canonical.matrix.rows) {
        return cuts;
    }

    linalg::SparseBasisOptions s_opts;
    s_opts.singular_tolerance = 1e-11;
    s_opts.maximum_dimension = canonical.matrix.rows + 64;
    s_opts.maximum_nonzeros = 1024 * 1024;
    s_opts.maximum_factor_nonzeros = 1024 * 1024;

    std::unique_ptr<linalg::SparseBasisFactorization> factor;
    try {
        const auto b_mat = extract_basis_matrix(canonical, basis_state.basic_variables);
        factor = std::make_unique<linalg::SparseBasisFactorization>(
            linalg::SparseBasisFactorization::factorize(b_mat, s_opts));
    } catch (...) {
        return cuts;
    }

    std::vector<bool> is_basic(canonical.matrix.columns, false);
    for (std::size_t b_col : basis_state.basic_variables) {
        if (b_col < is_basic.size()) {
            is_basic[b_col] = true;
        }
    }

    // Map canonical columns back to original variable indices
    std::vector<int> col_to_orig(canonical.matrix.columns, -1);
    for (std::size_t j = 0; j < model.matrix.column_count; ++j) {
        if (j < canonical.record.variables.size()) {
            for (std::size_t c_idx : canonical.record.variables[j].canonical_index) {
                if (c_idx < col_to_orig.size()) {
                    col_to_orig[c_idx] = static_cast<int>(j);
                }
            }
        }
    }

    const std::size_t struct_count = canonical.record.structural_variables;
    std::vector<std::vector<std::pair<std::size_t, double>>> row_structural_entries(canonical.matrix.rows);
    for (std::size_t c = 0; c < struct_count; ++c) {
        const std::size_t start = canonical.matrix.column_offsets[c];
        const std::size_t end = canonical.matrix.column_offsets[c + 1];
        for (std::size_t p = start; p < end; ++p) {
            row_structural_entries[canonical.matrix.row_indices[p]].push_back({c, canonical.matrix.values[p]});
        }
    }

    for (std::size_t row_i = 0; row_i < basis_state.basic_variables.size() && cuts.size() < max_cuts; ++row_i) {
        const std::size_t basic_col = basis_state.basic_variables[row_i];
        const int orig_var = col_to_orig[basic_col];
        if (orig_var < 0 || model.variable_type[orig_var] == model::VariableType::continuous) {
            continue;
        }

        const double x_val = original_primal[orig_var];
        const double f0 = x_val - std::floor(x_val);
        if (f0 < min_fractionality || f0 > 1.0 - min_fractionality) {
            continue;
        }

        // BTRAN: solve B^T y = e_{row_i}
        std::vector<double> e(canonical.matrix.rows, 0.0);
        e[row_i] = 1.0;
        std::vector<double> y;
        try {
            y = factor->solve_transpose(e);
        } catch (...) {
            continue;
        }

        // Tableau row: a_bar = y^T A
        const auto a_bar = canonical.multiply_transpose(y);

        // Compute cut coefficients alpha on canonical columns
        std::vector<double> alpha(canonical.matrix.columns, 0.0);
        for (std::size_t col_j = 0; col_j < canonical.matrix.columns; ++col_j) {
            if (is_basic[col_j]) {
                continue;
            }

            const double a_val = a_bar[col_j];
            if (std::abs(a_val) < 1e-12) {
                continue;
            }

            const int orig_j = col_to_orig[col_j];
            const bool is_int = (orig_j >= 0 && model.variable_type[orig_j] != model::VariableType::continuous);

            if (is_int) {
                const double fj = a_val - std::floor(a_val);
                if (fj <= f0) {
                    alpha[col_j] = fj;
                } else {
                    alpha[col_j] = (f0 * (1.0 - fj)) / (1.0 - f0);
                }
            } else {
                if (a_val >= 0.0) {
                    alpha[col_j] = a_val;
                } else {
                    alpha[col_j] = -(f0 / (1.0 - f0)) * a_val;
                }
            }
        }

        // Strictly verify cut contains no non-basic slack variables
        // This guarantees 100% mathematical validity in original variable space.
        bool has_slack = false;
        for (std::size_t col_k = struct_count; col_k < canonical.matrix.columns; ++col_k) {
            if (std::abs(alpha[col_k]) > 1e-8) {
                has_slack = true;
                break;
            }
        }
        if (has_slack) {
            continue;
        }

        // Generate cut in original variables: sum c_j x_j >= rhs
        Cut cut;
        cut.coefficients.assign(model.matrix.column_count, 0.0);
        double cut_rhs = f0;
        for (std::size_t orig_j = 0; orig_j < model.matrix.column_count; ++orig_j) {
            const auto& vmap = canonical.record.variables[orig_j];
            for (std::size_t q = 0; q < vmap.canonical_index.size(); ++q) {
                const std::size_t c_idx = vmap.canonical_index[q];
                if (c_idx < struct_count && std::abs(alpha[c_idx]) > 1e-12) {
                    const double mult = vmap.multiplier[q];
                    cut.coefficients[orig_j] += alpha[c_idx] * mult;
                    cut_rhs += alpha[c_idx] * mult * vmap.offset;
                }
            }
        }

        bool all_finite = std::isfinite(cut_rhs);
        for (double c : cut.coefficients) {
            if (!std::isfinite(c)) {
                all_finite = false;
                break;
            }
        }
        if (!all_finite) {
            continue;
        }

        // Check if cut has nonzeros and actually cuts off current point
        double lhs_val = 0.0;
        bool has_nonzeros = false;
        for (std::size_t j = 0; j < model.matrix.column_count; ++j) {
            if (std::abs(cut.coefficients[j]) > 1e-9) {
                has_nonzeros = true;
                lhs_val += cut.coefficients[j] * original_primal[j];
            }
        }

        cut.rhs = cut_rhs;
        cut.violation = cut.rhs - lhs_val;
        if (has_nonzeros && cut.violation > 1e-4) {
            cuts.push_back(std::move(cut));
        }
    }

    return cuts;
}

void add_cuts_to_model(
    model::Model& model,
    const std::vector<Cut>& cuts) {
    if (cuts.empty()) {
        return;
    }

    const std::size_t initial_rows = model.matrix.row_count;
    const std::size_t cols = model.matrix.column_count;
    const std::size_t new_rows = initial_rows + cuts.size();

    model::SparseMatrixBuilder builder(new_rows, cols);

    // Re-insert existing coefficients
    for (std::size_t j = 0; j < cols; ++j) {
        const std::size_t start = model.matrix.column_start[j];
        const std::size_t end = model.matrix.column_start[j + 1];
        for (std::size_t p = start; p < end; ++p) {
            builder.add(model.matrix.row_index[p], j, model.matrix.value[p]);
        }
    }

    // Append cut rows
    for (std::size_t c = 0; c < cuts.size(); ++c) {
        const std::size_t r = initial_rows + c;
        const auto& cut = cuts[c];
        for (std::size_t j = 0; j < cols; ++j) {
            if (std::abs(cut.coefficients[j]) > 1e-12) {
                builder.add(r, j, cut.coefficients[j]);
            }
        }
        model.row_lower.push_back(model::Bound::finite(cut.rhs));
        model.row_upper.push_back(model::Bound::positive_infinity());
        model.row_name.push_back("GOMORY_CUT_" + std::to_string(c + 1));
    }

    model.matrix = builder.build();
}

} // namespace markov_cero::milp
