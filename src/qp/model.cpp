#include "markov_cero/qp/model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace markov_cero::qp {

void SparseSymmetricMatrix::validate(std::size_t maximum_nonzeros) const {
    if (dimension > 4096) {
        throw std::invalid_argument("SparseSymmetricMatrix dimension exceeds 4096 limit");
    }
    if (column_offsets.size() != dimension + 1) {
        throw std::invalid_argument("SparseSymmetricMatrix invalid column_offsets size");
    }
    if (column_offsets[0] != 0) {
        throw std::invalid_argument("SparseSymmetricMatrix column_offsets[0] must be 0");
    }
    if (row_indices.size() != values.size()) {
        throw std::invalid_argument("SparseSymmetricMatrix row/values size mismatch");
    }
    if (values.size() > maximum_nonzeros) {
        throw std::invalid_argument("SparseSymmetricMatrix nonzeros exceed maximum limit");
    }
    for (std::size_t j = 0; j < dimension; ++j) {
        const std::size_t start = column_offsets[j];
        const std::size_t end = column_offsets[j + 1];
        if (start > end || end > row_indices.size()) {
            throw std::invalid_argument("SparseSymmetricMatrix invalid column offset range");
        }
        std::size_t last_row = 0;
        for (std::size_t k = start; k < end; ++k) {
            const std::size_t r = row_indices[k];
            if (r > j) {
                throw std::invalid_argument(
                    "SparseSymmetricMatrix must be upper triangular (i<=j)");
            }
            if (k > start && r <= last_row) {
                throw std::invalid_argument("SparseSymmetricMatrix rows not strictly increasing");
            }
            last_row = r;
        }
    }
}

double SparseSymmetricMatrix::evaluate_energy(const std::vector<double>& x) const {
    if (x.size() != dimension) {
        throw std::invalid_argument("Vector size mismatch in evaluate_energy");
    }
    double sum = 0.0;
    for (std::size_t j = 0; j < dimension; ++j) {
        const double xj = x[j];
        const std::size_t start = column_offsets[j];
        const std::size_t end = column_offsets[j + 1];
        for (std::size_t k = start; k < end; ++k) {
            const std::size_t i = row_indices[k];
            const double v = values[k];
            if (i == j) {
                sum += v * xj * xj;
            } else {
                sum += 2.0 * v * x[i] * xj;
            }
        }
    }
    return sum;
}

std::vector<double> SparseSymmetricMatrix::multiply(const std::vector<double>& x) const {
    if (x.size() != dimension) {
        throw std::invalid_argument("Vector size mismatch in SparseSymmetricMatrix::multiply");
    }
    std::vector<double> y(dimension, 0.0);
    for (std::size_t j = 0; j < dimension; ++j) {
        const double xj = x[j];
        const std::size_t start = column_offsets[j];
        const std::size_t end = column_offsets[j + 1];
        for (std::size_t k = start; k < end; ++k) {
            const std::size_t i = row_indices[k];
            const double v = values[k];
            y[i] += v * xj;
            if (i != j) {
                y[j] += v * x[i];
            }
        }
    }
    return y;
}

linalg::SparseCsc SparseSymmetricMatrix::to_full_sparse_csc() const {
    std::vector<std::vector<std::pair<std::size_t, double>>> cols(dimension);
    for (std::size_t j = 0; j < dimension; ++j) {
        const std::size_t start = column_offsets[j];
        const std::size_t end = column_offsets[j + 1];
        for (std::size_t k = start; k < end; ++k) {
            const std::size_t i = row_indices[k];
            const double v = values[k];
            cols[j].emplace_back(i, v);
            if (i != j) {
                cols[i].emplace_back(j, v);
            }
        }
    }
    linalg::SparseCsc full;
    full.rows = dimension;
    full.columns = dimension;
    full.column_offsets.push_back(0);
    for (std::size_t j = 0; j < dimension; ++j) {
        std::sort(cols[j].begin(), cols[j].end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& [r, v] : cols[j]) {
            full.row_indices.push_back(r);
            full.values.push_back(v);
        }
        full.column_offsets.push_back(full.values.size());
    }
    return full;
}

void QuadraticModel::validate() const {
    P.validate();
    A.validate();
    const std::size_t n = num_variables();
    const std::size_t m = num_constraints();
    if (P.dimension != n) {
        throw std::invalid_argument("QuadraticModel P dimension does not match variable count");
    }
    if (A.columns != n) {
        throw std::invalid_argument("QuadraticModel A columns do not match variable count");
    }
    if (A.rows != m || l.size() != m || u.size() != m) {
        throw std::invalid_argument("QuadraticModel constraint dimension mismatch");
    }
    for (std::size_t i = 0; i < m; ++i) {
        if (l[i] > u[i] + 1e-12) {
            throw std::invalid_argument(
                "QuadraticModel constraint lower bound exceeds upper bound");
        }
    }
}

bool check_convexity(const SparseSymmetricMatrix& P, double tolerance) {
    const std::size_t n = P.dimension;
    if (n == 0) {
        return true;
    }
    // Check diagonal elements first: P_ii < -tolerance implies non-convexity immediately
    std::vector<double> diag(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t start = P.column_offsets[j];
        const std::size_t end = P.column_offsets[j + 1];
        for (std::size_t k = start; k < end; ++k) {
            if (P.row_indices[k] == j) {
                diag[j] = P.values[k];
                if (diag[j] < -tolerance) {
                    return false;
                }
            }
        }
    }

    // Dense LDL^T factorization check for n <= 4096
    // Work with active symmetric submatrix
    std::vector<double> A_dense(n * n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t start = P.column_offsets[j];
        const std::size_t end = P.column_offsets[j + 1];
        for (std::size_t k = start; k < end; ++k) {
            const std::size_t i = P.row_indices[k];
            const double v = P.values[k];
            A_dense[i * n + j] = v;
            A_dense[j * n + i] = v;
        }
    }

    // Diagonal Gaussian elimination
    for (std::size_t k = 0; k < n; ++k) {
        const double pivot = A_dense[k * n + k];
        if (pivot < -tolerance) {
            return false;
        }
        if (std::abs(pivot) <= tolerance) {
            // Check if entire row k is zero. If non-zero entry exists, matrix is indefinite.
            for (std::size_t j = k + 1; j < n; ++j) {
                if (std::abs(A_dense[k * n + j]) > tolerance) {
                    return false;
                }
            }
            continue;
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            const double factor = A_dense[i * n + k] / pivot;
            if (std::abs(factor) <= 1e-15) {
                continue;
            }
            for (std::size_t j = i; j < n; ++j) {
                const double update = factor * A_dense[k * n + j];
                A_dense[i * n + j] -= update;
                if (i != j) {
                    A_dense[j * n + i] -= update;
                }
            }
        }
    }
    return true;
}

QuadraticModel make_quadratic_model(const model::Model& model) {
    QuadraticModel qp;
    qp.name = model.name;
    qp.sense = model.objective_sense;
    qp.objective_offset = model.objective_offset;

    const std::size_t n = model.matrix.column_count;
    const std::size_t m_orig = model.matrix.row_count;
    const double sign = (model.objective_sense == model::ObjectiveSense::maximize) ? -1.0 : 1.0;

    // Linear objective q
    qp.q.resize(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        if (j < model.objective.size()) {
            qp.q[j] = sign * model.objective[j];
        }
    }

    // Quadratic objective P (upper triangular)
    qp.P.dimension = n;
    qp.P.column_offsets.assign(n + 1, 0);
    if (model.has_quadratic_objective && model.quadratic_matrix.column_count == n) {
        std::vector<std::vector<std::pair<std::size_t, double>>> upper_entries(n);
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t start = model.quadratic_matrix.column_start[j];
            const std::size_t end = model.quadratic_matrix.column_start[j + 1];
            for (std::size_t k = start; k < end; ++k) {
                const std::size_t i = model.quadratic_matrix.row_index[k];
                const double v = sign * model.quadratic_matrix.value[k];
                if (i <= j) {
                    upper_entries[j].emplace_back(i, v);
                } else {
                    upper_entries[i].emplace_back(j, v);
                }
            }
        }
        for (std::size_t j = 0; j < n; ++j) {
            std::sort(upper_entries[j].begin(), upper_entries[j].end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            // Merge duplicate entries
            for (const auto& [r, v] : upper_entries[j]) {
                if (!qp.P.row_indices.empty() && qp.P.row_indices.back() == r &&
                    qp.P.column_offsets[j] < qp.P.row_indices.size()) {
                    qp.P.values.back() += v;
                } else {
                    qp.P.row_indices.push_back(r);
                    qp.P.values.push_back(v);
                }
            }
            qp.P.column_offsets[j + 1] = qp.P.values.size();
        }
    }

    // Combined constraints: m = m_orig + n (general constraints + variable box bounds)
    const std::size_t m_total = m_orig + n;
    qp.l.resize(m_total);
    qp.u.resize(m_total);

    for (std::size_t i = 0; i < m_orig; ++i) {
        qp.l[i] = (i < model.row_lower.size() && model.row_lower[i].is_finite())
                      ? model.row_lower[i].value
                      : -std::numeric_limits<double>::infinity();
        qp.u[i] = (i < model.row_upper.size() && model.row_upper[i].is_finite())
                      ? model.row_upper[i].value
                      : std::numeric_limits<double>::infinity();
    }
    for (std::size_t j = 0; j < n; ++j) {
        qp.l[m_orig + j] = (j < model.variable_lower.size() && model.variable_lower[j].is_finite())
                               ? model.variable_lower[j].value
                               : -std::numeric_limits<double>::infinity();
        qp.u[m_orig + j] = (j < model.variable_upper.size() && model.variable_upper[j].is_finite())
                               ? model.variable_upper[j].value
                               : std::numeric_limits<double>::infinity();
    }

    // Combined constraint matrix A: [A_orig; I_n]
    qp.A.rows = m_total;
    qp.A.columns = n;
    qp.A.column_offsets.assign(n + 1, 0);
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t start = (j < model.matrix.column_start.size())
                                      ? model.matrix.column_start[j]
                                      : 0;
        const std::size_t end = (j + 1 < model.matrix.column_start.size())
                                    ? model.matrix.column_start[j + 1]
                                    : start;
        for (std::size_t k = start; k < end; ++k) {
            qp.A.row_indices.push_back(model.matrix.row_index[k]);
            qp.A.values.push_back(model.matrix.value[k]);
        }
        // Identity entry at row m_orig + j
        qp.A.row_indices.push_back(m_orig + j);
        qp.A.values.push_back(1.0);
        qp.A.column_offsets[j + 1] = qp.A.values.size();
    }

    qp.variable_types = model.variable_type;
    qp.variable_names = model.variable_name;
    qp.constraint_names = model.row_name;
    for (std::size_t j = 0; j < n; ++j) {
        std::string vname = (j < model.variable_name.size() && !model.variable_name[j].empty())
                                ? model.variable_name[j]
                                : ("x" + std::to_string(j));
        qp.constraint_names.push_back("bnd_" + vname);
    }
    return qp;
}

} // namespace markov_cero::qp
