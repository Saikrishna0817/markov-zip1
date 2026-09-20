#include "markov_cero/milp/gomory.hpp"

#include "markov_cero/linalg/sparse_basis.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace markov_cero::milp {

std::vector<Cut> generate_gomory_cuts(const model::Model& model,
                                      const std::vector<double>& original_primal,
                                      const transform::SparseCanonicalModel& canonical,
                                      const lp::dual::BasisState& basis_state, std::size_t max_cuts,
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

    // Cache row structural entries and slack info for algebraic substitution
    struct SlackInfo {
        std::size_t row{0};
        double coeff{1.0};
        bool valid{false};
    };
    std::vector<SlackInfo> slack_info(canonical.matrix.columns);
    for (std::size_t c = struct_count; c < canonical.matrix.columns; ++c) {
        const std::size_t start = canonical.matrix.column_offsets[c];
        const std::size_t end = canonical.matrix.column_offsets[c + 1];
        if (end - start == 1) {
            slack_info[c].row = canonical.matrix.row_indices[start];
            slack_info[c].coeff = canonical.matrix.values[start];
            slack_info[c].valid = (std::abs(slack_info[c].coeff) > 1e-12);
        }
    }

    std::vector<std::vector<std::pair<std::size_t, double>>> row_structural_entries(
        canonical.matrix.rows);
    for (std::size_t c = 0; c < struct_count; ++c) {
        const std::size_t start = canonical.matrix.column_offsets[c];
        const std::size_t end = canonical.matrix.column_offsets[c + 1];
        for (std::size_t p = start; p < end; ++p) {
            row_structural_entries[canonical.matrix.row_indices[p]].push_back(
                {c, canonical.matrix.values[p]});
        }
    }

    for (std::size_t row_i = 0;
         row_i < basis_state.basic_variables.size() && cuts.size() < max_cuts; ++row_i) {
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
            const bool is_int =
                (orig_j >= 0 && model.variable_type[orig_j] != model::VariableType::continuous);

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

        // Slack substitution: s_r = (rhs_r - sum A_{r, k} x_k) / coeff_r
        std::vector<double> alpha_struct(struct_count, 0.0);
        for (std::size_t c = 0; c < struct_count; ++c) {
            alpha_struct[c] = alpha[c];
        }
        double rhs_shift = 0.0;
        bool slack_substitution_success = true;

        for (std::size_t c = struct_count; c < canonical.matrix.columns; ++c) {
            if (std::abs(alpha[c]) < 1e-12) {
                continue;
            }
            if (!slack_info[c].valid) {
                slack_substitution_success = false;
                break;
            }
            const std::size_t r = slack_info[c].row;
            const double mult = alpha[c] / slack_info[c].coeff;
            rhs_shift += mult * canonical.rhs[r];
            for (const auto& [str_col, a_rk] : row_structural_entries[r]) {
                alpha_struct[str_col] -= mult * a_rk;
            }
        }

        if (!slack_substitution_success) {
            continue;
        }

        const double cut_rhs_canonical = f0 - rhs_shift;

        // Generate cut in original variables: sum c_j x_j >= rhs
        Cut cut;
        cut.coefficients.assign(model.matrix.column_count, 0.0);
        double cut_rhs = cut_rhs_canonical;
        for (std::size_t orig_j = 0; orig_j < model.matrix.column_count; ++orig_j) {
            const auto& vmap = canonical.record.variables[orig_j];
            for (std::size_t q = 0; q < vmap.canonical_index.size(); ++q) {
                const std::size_t c_idx = vmap.canonical_index[q];
                if (c_idx < struct_count && std::abs(alpha_struct[c_idx]) > 1e-12) {
                    const double mult = vmap.multiplier[q];
                    cut.coefficients[orig_j] += alpha_struct[c_idx] * mult;
                    cut_rhs += alpha_struct[c_idx] * mult * vmap.offset;
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

    return filter_cuts(std::move(cuts), max_cuts);
}

} // namespace markov_cero::milp
