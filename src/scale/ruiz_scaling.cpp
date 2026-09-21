#include "markov_cero/scale/ruiz_scaling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace markov_cero::scale {
namespace {

void ensure_finite(double v, const char* message) {
    if (!std::isfinite(v)) {
        throw std::overflow_error(message);
    }
}

} // namespace

RuizScalers equilibrate(transform::SparseCanonicalModel& model, const RuizOptions& options) {
    const std::size_t m = model.matrix.rows;
    const std::size_t n = model.matrix.columns;

    RuizScalers scalers;
    scalers.row_scale.assign(m, 1.0);
    scalers.col_scale.assign(n, 1.0);
    scalers.inv_row_scale.assign(m, 1.0);
    scalers.inv_col_scale.assign(n, 1.0);

    if (m == 0 || n == 0 || model.matrix.values.empty()) {
        return scalers;
    }

    std::size_t iter = 0;
    for (; iter < options.max_iterations; ++iter) {
        // 1. Compute row inf-norms
        std::vector<double> r_norm(m, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t start = model.matrix.column_offsets[j];
            const std::size_t end = model.matrix.column_offsets[j + 1];
            for (std::size_t k = start; k < end; ++k) {
                const std::size_t r = model.matrix.row_indices[k];
                r_norm[r] = std::max(r_norm[r], std::abs(model.matrix.values[k]));
            }
        }
        for (std::size_t i = 0; i < m; ++i) {
            if (r_norm[i] == 0.0) {
                r_norm[i] = 1.0;
            }
        }

        // 2. Compute col inf-norms
        std::vector<double> c_norm(n, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            double max_v = 0.0;
            const std::size_t start = model.matrix.column_offsets[j];
            const std::size_t end = model.matrix.column_offsets[j + 1];
            for (std::size_t k = start; k < end; ++k) {
                max_v = std::max(max_v, std::abs(model.matrix.values[k]));
            }
            if (max_v == 0.0) {
                max_v = 1.0;
            }
            c_norm[j] = max_v;
        }

        // 3. Check convergence
        double max_r_err = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            max_r_err = std::max(max_r_err, std::abs(1.0 - r_norm[i]));
        }
        double max_c_err = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            max_c_err = std::max(max_c_err, std::abs(1.0 - c_norm[j]));
        }

        if (max_r_err < options.tolerance && max_c_err < options.tolerance) {
            scalers.converged = true;
            break;
        }

        // 4. Compute delta scalers: delta_R = 1/sqrt(r_norm), delta_C = 1/sqrt(c_norm)
        std::vector<double> delta_r(m);
        for (std::size_t i = 0; i < m; ++i) {
            delta_r[i] = 1.0 / std::sqrt(r_norm[i]);
            scalers.row_scale[i] *= delta_r[i];
        }
        std::vector<double> delta_c(n);
        for (std::size_t j = 0; j < n; ++j) {
            delta_c[j] = 1.0 / std::sqrt(c_norm[j]);
            scalers.col_scale[j] *= delta_c[j];
        }

        // Update matrix values: A_{ij} = delta_r[i] * A_{ij} * delta_c[j]
        for (std::size_t j = 0; j < n; ++j) {
            const double dc = delta_c[j];
            const std::size_t start = model.matrix.column_offsets[j];
            const std::size_t end = model.matrix.column_offsets[j + 1];
            for (std::size_t k = start; k < end; ++k) {
                const std::size_t r = model.matrix.row_indices[k];
                model.matrix.values[k] *= (delta_r[r] * dc);
            }
        }
    }
    scalers.iterations_executed = iter;

    // Apply numerical safeguard clamping to prevent extreme scaling multipliers
    for (std::size_t i = 0; i < m; ++i) {
        scalers.row_scale[i] =
            std::clamp(scalers.row_scale[i], options.min_scale, options.max_scale);
        scalers.inv_row_scale[i] = 1.0 / scalers.row_scale[i];
        model.rhs[i] *= scalers.row_scale[i];
        ensure_finite(model.rhs[i], "non-finite scaled RHS");
    }

    for (std::size_t j = 0; j < n; ++j) {
        scalers.col_scale[j] =
            std::clamp(scalers.col_scale[j], options.min_scale, options.max_scale);
        scalers.inv_col_scale[j] = 1.0 / scalers.col_scale[j];
        model.objective[j] *= scalers.col_scale[j];
        ensure_finite(model.objective[j], "non-finite scaled objective");
    }

    return scalers;
}

void unscale_solution(const RuizScalers& scalers, lp::reference::Result& solution) {
    // 1. Primal variables: x = D_C * x_bar
    for (std::size_t j = 0; j < solution.primal.size() && j < scalers.col_scale.size(); ++j) {
        solution.primal[j] *= scalers.col_scale[j];
        ensure_finite(solution.primal[j], "non-finite unscaled primal variable");
    }

    // 2. Dual multipliers: pi = D_R * pi_bar
    for (std::size_t i = 0; i < solution.dual.size() && i < scalers.row_scale.size(); ++i) {
        solution.dual[i] *= scalers.row_scale[i];
        ensure_finite(solution.dual[i], "non-finite unscaled dual multiplier");
    }

    // 3. Rays and certificates
    for (std::size_t j = 0; j < solution.ray.size() && j < scalers.col_scale.size(); ++j) {
        solution.ray[j] *= scalers.col_scale[j];
    }
    for (std::size_t i = 0; i < solution.certificate.size() && i < scalers.row_scale.size(); ++i) {
        solution.certificate[i] *= scalers.row_scale[i];
    }

    // Note: objective value c^T x = (D_C c)^T (D_C^{-1} x) is mathematically invariant under
    // diagonal scaling
}

RuizScalers equilibrate_model(model::Model& model, const RuizOptions& options) {
    const std::size_t m = model.matrix.row_count;
    const std::size_t n = model.matrix.column_count;

    RuizScalers scalers;
    scalers.row_scale.assign(m, 1.0);
    scalers.col_scale.assign(n, 1.0);
    scalers.inv_row_scale.assign(m, 1.0);
    scalers.inv_col_scale.assign(n, 1.0);

    if (m == 0 || n == 0 || model.matrix.value.empty()) {
        return scalers;
    }

    std::size_t iter = 0;
    for (; iter < options.max_iterations; ++iter) {
        std::vector<double> r_norm(m, 0.0);
        std::vector<double> c_norm(n, 0.0);
        for (std::size_t col = 0; col < n; ++col) {
            for (std::size_t p = model.matrix.column_start[col];
                 p < model.matrix.column_start[col + 1]; ++p) {
                const double v = std::abs(model.matrix.value[p]);
                c_norm[col] = std::max(c_norm[col], v);
                r_norm[model.matrix.row_index[p]] = std::max(r_norm[model.matrix.row_index[p]], v);
            }
        }
        for (std::size_t i = 0; i < m; ++i) {
            if (r_norm[i] == 0.0) r_norm[i] = 1.0;
        }
        for (std::size_t j = 0; j < n; ++j) {
            if (c_norm[j] == 0.0) c_norm[j] = 1.0;
        }

        double max_r_err = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            max_r_err = std::max(max_r_err, std::abs(1.0 - r_norm[i]));
        }
        double max_c_err = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            max_c_err = std::max(max_c_err, std::abs(1.0 - c_norm[j]));
        }

        if (max_r_err < options.tolerance && max_c_err < options.tolerance) {
            scalers.converged = true;
            break;
        }

        std::vector<double> delta_r(m);
        for (std::size_t i = 0; i < m; ++i) {
            delta_r[i] = 1.0 / std::sqrt(r_norm[i]);
            scalers.row_scale[i] *= delta_r[i];
        }
        std::vector<double> delta_c(n);
        for (std::size_t j = 0; j < n; ++j) {
            delta_c[j] = 1.0 / std::sqrt(c_norm[j]);
            scalers.col_scale[j] *= delta_c[j];
        }

        for (std::size_t col = 0; col < n; ++col) {
            const double dc = delta_c[col];
            for (std::size_t p = model.matrix.column_start[col];
                 p < model.matrix.column_start[col + 1]; ++p) {
                const std::size_t r = model.matrix.row_index[p];
                model.matrix.value[p] *= delta_r[r] * dc;
            }
        }
    }
    scalers.iterations_executed = iter;

    for (std::size_t i = 0; i < m; ++i) {
        scalers.inv_row_scale[i] = 1.0 / scalers.row_scale[i];
    }
    for (std::size_t j = 0; j < n; ++j) {
        scalers.inv_col_scale[j] = 1.0 / scalers.col_scale[j];
    }

    for (std::size_t i = 0; i < m; ++i) {
        if (model.row_lower[i].kind == model::BoundKind::finite) {
            model.row_lower[i].value *= scalers.row_scale[i];
        }
        if (model.row_upper[i].kind == model::BoundKind::finite) {
            model.row_upper[i].value *= scalers.row_scale[i];
        }
    }
    for (std::size_t j = 0; j < n; ++j) {
        if (model.variable_lower[j].kind == model::BoundKind::finite) {
            model.variable_lower[j].value *= scalers.inv_col_scale[j];
        }
        if (model.variable_upper[j].kind == model::BoundKind::finite) {
            model.variable_upper[j].value *= scalers.inv_col_scale[j];
        }
        model.objective[j] *= scalers.col_scale[j];
    }

    return scalers;
}

void unscale_model_solution(const RuizScalers& scalers,
                            std::vector<double>& primal,
                            std::vector<double>& dual) {
    for (std::size_t j = 0; j < primal.size() && j < scalers.col_scale.size(); ++j) {
        primal[j] *= scalers.col_scale[j];
    }
    for (std::size_t i = 0; i < dual.size() && i < scalers.row_scale.size(); ++i) {
        dual[i] *= scalers.row_scale[i];
    }
}

} // namespace markov_cero::scale
