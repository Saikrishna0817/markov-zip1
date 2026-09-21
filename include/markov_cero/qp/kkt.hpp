#pragma once

#include "markov_cero/linalg/sparse_basis.hpp"
#include "markov_cero/qp/model.hpp"

#include <cstddef>
#include <vector>

namespace markov_cero::qp {

/// Symmetric quasi-definite KKT linear system solver:
///   [ P + sigma*I       A^T       ] [  x  ] = [ rhs_x ]
///   [     A         -diag(rho)^-1 ] [ nu  ]   [ rhs_z ]
///
/// Uses Davis's sparse LDL^T factorization (Algorithm 849).
/// Because the system is symmetric quasi-definite (SQD), non-singular LDL^T
/// factorization is guaranteed without numerical pivot searching.
class KktSolver {
public:
    KktSolver() = default;

    /// Constructs and factorizes the augmented KKT matrix.
    /// Returns true if factorization is successful.
    bool factorize(const SparseSymmetricMatrix& P,
                   const linalg::SparseCsc& A,
                   double sigma,
                   const std::vector<double>& rho);

    /// Re-factorizes when only rho or sigma values change (sparsity pattern unchanged).
    bool update_numeric(const SparseSymmetricMatrix& P,
                        const linalg::SparseCsc& A,
                        double sigma,
                        const std::vector<double>& rho);

    /// Solves K * [sol_x; sol_nu] = [rhs_x; rhs_z].
    void solve(const std::vector<double>& rhs_x,
               const std::vector<double>& rhs_z,
               std::vector<double>& sol_x,
               std::vector<double>& sol_nu) const;

    [[nodiscard]] std::size_t num_variables() const noexcept { return n_; }
    [[nodiscard]] std::size_t num_constraints() const noexcept { return m_; }
    [[nodiscard]] std::size_t dimension() const noexcept { return total_dim_; }
    [[nodiscard]] bool is_factorized() const noexcept { return factorized_; }
    [[nodiscard]] std::size_t nonzeros_L() const noexcept;

private:
    std::size_t n_{0};
    std::size_t m_{0};
    std::size_t total_dim_{0};
    bool factorized_{false};

    // Upper-triangular CSC representation of K
    std::vector<std::size_t> kkt_col_ptr_;
    std::vector<std::size_t> kkt_row_ind_;
    std::vector<double> kkt_val_;

    // Factorization results
    std::vector<std::size_t> L_col_ptr_;
    std::vector<std::size_t> L_row_ind_;
    std::vector<double> L_val_;
    std::vector<double> D_;
    std::vector<std::size_t> parent_;

    // Permutation (identity if unused)
    std::vector<std::size_t> perm_;
    std::vector<std::size_t> pinv_;

    void build_kkt_matrix(const SparseSymmetricMatrix& P,
                          const linalg::SparseCsc& A,
                          double sigma,
                          const std::vector<double>& rho);
};

} // namespace markov_cero::qp
