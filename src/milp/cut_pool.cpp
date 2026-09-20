#include "markov_cero/milp/cut_pool.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace markov_cero::milp {

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

std::vector<Cut> filter_cuts(std::vector<Cut> candidates, std::size_t max_cuts,
                             double min_violation, double max_parallelism) {
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

void add_cuts_to_model(model::Model& model, const std::vector<Cut>& cuts) {
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
