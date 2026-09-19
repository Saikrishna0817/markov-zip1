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

double compute_cut_efficacy(const Cut& cut) noexcept {
    double sum_sq = 0.0;
    for (double c : cut.coefficients) {
        sum_sq += c * c;
    }
    if (sum_sq <= 1e-24) {
        return 0.0;
    }
    return cut.violation / std::sqrt(sum_sq);
}

double compute_cosine_similarity(const Cut& a, const Cut& b) noexcept {
    const std::size_t n = std::min(a.coefficients.size(), b.coefficients.size());
    double dot = 0.0;
    double norm_a_sq = 0.0;
    double norm_b_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        dot += a.coefficients[i] * b.coefficients[i];
        norm_a_sq += a.coefficients[i] * a.coefficients[i];
        norm_b_sq += b.coefficients[i] * b.coefficients[i];
    }
    for (std::size_t i = n; i < a.coefficients.size(); ++i) {
        norm_a_sq += a.coefficients[i] * a.coefficients[i];
    }
    for (std::size_t i = n; i < b.coefficients.size(); ++i) {
        norm_b_sq += b.coefficients[i] * b.coefficients[i];
    }
    const double denom = std::sqrt(norm_a_sq) * std::sqrt(norm_b_sq);
    if (denom <= 1e-24) {
        return 0.0;
    }
    const double cos_sim = std::abs(dot) / denom;
    return std::clamp(cos_sim, 0.0, 1.0);
}

double mir_function(double a, double f0) noexcept {
    if (f0 <= 0.0 || f0 >= 1.0) {
        return std::floor(a);
    }
    const double fl = std::floor(a);
    const double fj = a - fl;
    return fl + std::max(0.0, fj - f0) / (1.0 - f0);
}

std::vector<Cut> filter_cuts(
    std::vector<Cut> candidates,
    std::size_t max_cuts,
    double min_violation,
    double max_parallelism) {
    std::vector<Cut> valid_candidates;
    valid_candidates.reserve(candidates.size());

    for (auto& cut : candidates) {
        if (cut.violation < min_violation) {
            continue;
        }
        bool finite_coeff = std::isfinite(cut.rhs) && std::isfinite(cut.violation);
        double norm_sq = 0.0;
        for (double c : cut.coefficients) {
            if (!std::isfinite(c)) {
                finite_coeff = false;
                break;
            }
            norm_sq += c * c;
        }
        if (finite_coeff && norm_sq > 1e-20) {
            valid_candidates.push_back(std::move(cut));
        }
    }

    std::stable_sort(valid_candidates.begin(), valid_candidates.end(),
                     [](const Cut& a, const Cut& b) {
                         return compute_cut_efficacy(a) > compute_cut_efficacy(b);
                     });

    std::vector<Cut> filtered;
    filtered.reserve(std::min(valid_candidates.size(), max_cuts));

    for (auto& cand : valid_candidates) {
        if (filtered.size() >= max_cuts) {
            break;
        }
        bool is_parallel = false;
        for (const auto& accepted : filtered) {
            if (compute_cosine_similarity(cand, accepted) > max_parallelism) {
                is_parallel = true;
                break;
            }
        }
        if (!is_parallel) {
            filtered.push_back(std::move(cand));
        }
    }

    return filtered;
}

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

std::vector<Cut> generate_mir_cuts(
    const model::Model& model,
    const std::vector<double>& original_primal,
    const transform::SparseCanonicalModel& canonical,
    const lp::dual::BasisState& basis_state,
    std::size_t max_cuts,
    double min_fractionality) {
    std::vector<Cut> candidates;
    if (basis_state.basic_variables.size() != canonical.matrix.rows) {
        return candidates;
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
        return candidates;
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

    // Cache row structural entries and slack info
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

    std::vector<std::vector<std::pair<std::size_t, double>>> row_structural_entries(canonical.matrix.rows);
    for (std::size_t c = 0; c < struct_count; ++c) {
        const std::size_t start = canonical.matrix.column_offsets[c];
        const std::size_t end = canonical.matrix.column_offsets[c + 1];
        for (std::size_t p = start; p < end; ++p) {
            row_structural_entries[canonical.matrix.row_indices[p]].push_back({c, canonical.matrix.values[p]});
        }
    }

    for (std::size_t row_i = 0; row_i < basis_state.basic_variables.size(); ++row_i) {
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

        // Evaluate MIR cut in direct and complement orientations (Marchand & Wolsey 2001)
        for (int orientation = 0; orientation < 2; ++orientation) {
            const double sign = (orientation == 0) ? 1.0 : -1.0;
            const double eff_f0 = (orientation == 0) ? f0 : (1.0 - f0);

            std::vector<double> w(canonical.matrix.columns, 0.0);
            for (std::size_t col_j = 0; col_j < canonical.matrix.columns; ++col_j) {
                if (is_basic[col_j]) {
                    continue;
                }
                const double a_val = sign * a_bar[col_j];
                if (std::abs(a_val) < 1e-12) {
                    continue;
                }

                const int orig_j = col_to_orig[col_j];
                const bool is_int = (orig_j >= 0 && model.variable_type[orig_j] != model::VariableType::continuous);

                if (is_int) {
                    const double alpha_val = mir_function(a_val, eff_f0);
                    w[col_j] = a_val - alpha_val;
                } else {
                    if (a_val >= 0.0) {
                        w[col_j] = a_val;
                    } else {
                        w[col_j] = -(eff_f0 / (1.0 - eff_f0)) * a_val;
                    }
                }
            }

            // Slack substitution: s_r = (rhs_r - sum A_{r, k} x_k) / coeff_r
            std::vector<double> w_struct(struct_count, 0.0);
            for (std::size_t c = 0; c < struct_count; ++c) {
                w_struct[c] = w[c];
            }
            double rhs_shift = 0.0;
            bool slack_substitution_success = true;

            for (std::size_t c = struct_count; c < canonical.matrix.columns; ++c) {
                if (std::abs(w[c]) < 1e-12) {
                    continue;
                }
                if (!slack_info[c].valid) {
                    slack_substitution_success = false;
                    break;
                }
                const std::size_t r = slack_info[c].row;
                const double mult = w[c] / slack_info[c].coeff;
                rhs_shift += mult * canonical.rhs[r];
                for (const auto& [str_col, a_rk] : row_structural_entries[r]) {
                    w_struct[str_col] -= mult * a_rk;
                }
            }

            if (!slack_substitution_success) {
                continue;
            }

            const double cut_rhs_canonical = eff_f0 - rhs_shift;

            // Map canonical structural columns to original model variables
            Cut cut;
            cut.coefficients.assign(model.matrix.column_count, 0.0);
            double cut_rhs = cut_rhs_canonical;

            for (std::size_t orig_j = 0; orig_j < model.matrix.column_count; ++orig_j) {
                const auto& vmap = canonical.record.variables[orig_j];
                for (std::size_t q = 0; q < vmap.canonical_index.size(); ++q) {
                    const std::size_t c_idx = vmap.canonical_index[q];
                    if (c_idx < struct_count && std::abs(w_struct[c_idx]) > 1e-12) {
                        const double mult = vmap.multiplier[q];
                        cut.coefficients[orig_j] += w_struct[c_idx] * mult;
                        cut_rhs += w_struct[c_idx] * mult * vmap.offset;
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

            // Check nonzeros and violation at current point
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
                candidates.push_back(std::move(cut));
            }
        }
    }

    return filter_cuts(std::move(candidates), max_cuts);
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
