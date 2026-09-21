#include "markov_cero/qp/kkt.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace markov_cero::qp {

namespace {

void ldl_symbolic(std::size_t n,
                  const std::vector<std::size_t>& Ap,
                  const std::vector<std::size_t>& Ai,
                  std::vector<std::size_t>& Lp,
                  std::vector<std::size_t>& Parent,
                  std::vector<std::size_t>& Lnz,
                  std::vector<std::size_t>& Flag,
                  const std::vector<std::size_t>* P,
                  std::vector<std::size_t>* Pinv) {
    if (P && Pinv) {
        for (std::size_t k = 0; k < n; ++k) {
            (*Pinv)[(*P)[k]] = k;
        }
    }
    for (std::size_t k = 0; k < n; ++k) {
        Parent[k] = std::numeric_limits<std::size_t>::max();
        Flag[k] = k;
        Lnz[k] = 0;
        const std::size_t kk = P ? (*P)[k] : k;
        const std::size_t p2 = Ap[kk + 1];
        for (std::size_t p = Ap[kk]; p < p2; ++p) {
            std::size_t i = Pinv ? (*Pinv)[Ai[p]] : Ai[p];
            if (i < k) {
                for (; Flag[i] != k; i = Parent[i]) {
                    if (Parent[i] == std::numeric_limits<std::size_t>::max()) {
                        Parent[i] = k;
                    }
                    ++Lnz[i];
                    Flag[i] = k;
                }
            }
        }
    }
    Lp[0] = 0;
    for (std::size_t k = 0; k < n; ++k) {
        Lp[k + 1] = Lp[k] + Lnz[k];
    }
}

bool ldl_numeric(std::size_t n,
                 const std::vector<std::size_t>& Ap,
                 const std::vector<std::size_t>& Ai,
                 const std::vector<double>& Ax,
                 const std::vector<std::size_t>& Lp,
                 const std::vector<std::size_t>& Parent,
                 std::vector<std::size_t>& Lnz,
                 std::vector<std::size_t>& Li,
                 std::vector<double>& Lx,
                 std::vector<double>& D,
                 std::vector<double>& Y,
                 std::vector<std::size_t>& Pattern,
                 std::vector<std::size_t>& Flag,
                 const std::vector<std::size_t>* P,
                 const std::vector<std::size_t>* Pinv) {
    for (std::size_t k = 0; k < n; ++k) {
        Y[k] = 0.0;
        std::size_t top = n;
        Flag[k] = k;
        Lnz[k] = 0;
        const std::size_t kk = P ? (*P)[k] : k;
        const std::size_t p2 = Ap[kk + 1];
        for (std::size_t p = Ap[kk]; p < p2; ++p) {
            std::size_t i = Pinv ? (*Pinv)[Ai[p]] : Ai[p];
            if (i <= k) {
                Y[i] += Ax[p];
                std::size_t len = 0;
                for (; Flag[i] != k; i = Parent[i]) {
                    Pattern[len++] = i;
                    Flag[i] = k;
                }
                while (len > 0) {
                    Pattern[--top] = Pattern[--len];
                }
            }
        }
        D[k] = Y[k];
        Y[k] = 0.0;
        for (; top < n; ++top) {
            const std::size_t i = Pattern[top];
            const double yi = Y[i];
            Y[i] = 0.0;
            const std::size_t l_end = Lp[i] + Lnz[i];
            for (std::size_t p = Lp[i]; p < l_end; ++p) {
                Y[Li[p]] -= Lx[p] * yi;
            }
            const double l_ki = yi / D[i];
            D[k] -= l_ki * yi;
            const std::size_t p_store = Lp[i] + Lnz[i];
            Li[p_store] = k;
            Lx[p_store] = l_ki;
            ++Lnz[i];
        }
        if (std::abs(D[k]) < 1e-15 || !std::isfinite(D[k])) {
            return false;
        }
    }
    return true;
}

void ldl_lsolve(std::size_t n,
                std::vector<double>& x,
                const std::vector<std::size_t>& Lp,
                const std::vector<std::size_t>& Li,
                const std::vector<double>& Lx) {
    for (std::size_t j = 0; j < n; ++j) {
        const double xj = x[j];
        const std::size_t p2 = Lp[j + 1];
        for (std::size_t p = Lp[j]; p < p2; ++p) {
            x[Li[p]] -= Lx[p] * xj;
        }
    }
}

void ldl_dsolve(std::size_t n,
                std::vector<double>& x,
                const std::vector<double>& D) {
    for (std::size_t j = 0; j < n; ++j) {
        x[j] /= D[j];
    }
}

void ldl_ltsolve(std::size_t n,
                 std::vector<double>& x,
                 const std::vector<std::size_t>& Lp,
                 const std::vector<std::size_t>& Li,
                 const std::vector<double>& Lx) {
    for (std::size_t j = n; j > 0; --j) {
        const std::size_t col = j - 1;
        double sum = 0.0;
        const std::size_t p2 = Lp[col + 1];
        for (std::size_t p = Lp[col]; p < p2; ++p) {
            sum += Lx[p] * x[Li[p]];
        }
        x[col] -= sum;
    }
}

} // namespace

void KktSolver::build_kkt_matrix(const SparseSymmetricMatrix& P,
                                 const linalg::SparseCsc& A,
                                 double sigma,
                                 const std::vector<double>& rho) {
    n_ = P.dimension;
    m_ = A.rows;
    total_dim_ = n_ + m_;

    // Triplets for upper-triangular entries of K (row <= col)
    std::vector<std::map<std::size_t, double>> col_entries(total_dim_);

    // 1. P + sigma*I block in top-left (dimension n x n)
    for (std::size_t j = 0; j < n_; ++j) {
        col_entries[j][j] += sigma;
        if (j < P.column_offsets.size() - 1) {
            const std::size_t start = P.column_offsets[j];
            const std::size_t end = P.column_offsets[j + 1];
            for (std::size_t k = start; k < end; ++k) {
                const std::size_t i = P.row_indices[k];
                if (i <= j) {
                    col_entries[j][i] += P.values[k];
                }
            }
        }
    }

    // 2. A^T block in top-right (col n + i, row j) for i in [0, m), j in [0, n)
    for (std::size_t j = 0; j < n_; ++j) {
        if (j < A.column_offsets.size() - 1) {
            const std::size_t start = A.column_offsets[j];
            const std::size_t end = A.column_offsets[j + 1];
            for (std::size_t k = start; k < end; ++k) {
                const std::size_t i = A.row_indices[k];
                if (i < m_) {
                    col_entries[n_ + i][j] += A.values[k];
                }
            }
        }
    }

    // 3. -diag(rho)^-1 block in bottom-right (col n + i, row n + i)
    for (std::size_t i = 0; i < m_; ++i) {
        const double r = (i < rho.size() && rho[i] > 0.0) ? rho[i] : 1e-3;
        col_entries[n_ + i][n_ + i] -= 1.0 / r;
    }

    // Assemble upper-triangular CSC representation
    kkt_col_ptr_.assign(total_dim_ + 1, 0);
    kkt_row_ind_.clear();
    kkt_val_.clear();

    for (std::size_t j = 0; j < total_dim_; ++j) {
        for (const auto& [r, v] : col_entries[j]) {
            if (std::abs(v) > 1e-20) {
                kkt_row_ind_.push_back(r);
                kkt_val_.push_back(v);
            }
        }
        kkt_col_ptr_[j + 1] = kkt_val_.size();
    }
}

bool KktSolver::factorize(const SparseSymmetricMatrix& P,
                          const linalg::SparseCsc& A,
                          double sigma,
                          const std::vector<double>& rho) {
    build_kkt_matrix(P, A, sigma, rho);

    // Symbolic factorization
    L_col_ptr_.assign(total_dim_ + 1, 0);
    parent_.assign(total_dim_, std::numeric_limits<std::size_t>::max());
    std::vector<std::size_t> lnz(total_dim_, 0);
    std::vector<std::size_t> flag(total_dim_, 0);

    ldl_symbolic(total_dim_, kkt_col_ptr_, kkt_row_ind_, L_col_ptr_, parent_, lnz, flag,
                 nullptr, nullptr);

    const std::size_t total_lnz = L_col_ptr_[total_dim_];
    L_row_ind_.assign(total_lnz, 0);
    L_val_.assign(total_lnz, 0.0);
    D_.assign(total_dim_, 0.0);

    // Numeric factorization
    std::vector<double> Y(total_dim_, 0.0);
    std::vector<std::size_t> Pattern(total_dim_, 0);
    flag.assign(total_dim_, 0);
    lnz.assign(total_dim_, 0);

    factorized_ = ldl_numeric(total_dim_, kkt_col_ptr_, kkt_row_ind_, kkt_val_, L_col_ptr_,
                              parent_, lnz, L_row_ind_, L_val_, D_, Y, Pattern, flag,
                              nullptr, nullptr);

    return factorized_;
}

bool KktSolver::update_numeric(const SparseSymmetricMatrix& P,
                              const linalg::SparseCsc& A,
                              double sigma,
                              const std::vector<double>& rho) {
    build_kkt_matrix(P, A, sigma, rho);

    const std::size_t total_lnz = L_col_ptr_[total_dim_];
    L_row_ind_.assign(total_lnz, 0);
    L_val_.assign(total_lnz, 0.0);
    D_.assign(total_dim_, 0.0);

    std::vector<double> Y(total_dim_, 0.0);
    std::vector<std::size_t> Pattern(total_dim_, 0);
    std::vector<std::size_t> flag(total_dim_, 0);
    std::vector<std::size_t> lnz(total_dim_, 0);

    factorized_ = ldl_numeric(total_dim_, kkt_col_ptr_, kkt_row_ind_, kkt_val_, L_col_ptr_,
                              parent_, lnz, L_row_ind_, L_val_, D_, Y, Pattern, flag,
                              nullptr, nullptr);

    return factorized_;
}

void KktSolver::solve(const std::vector<double>& rhs_x,
                      const std::vector<double>& rhs_z,
                      std::vector<double>& sol_x,
                      std::vector<double>& sol_nu) const {
    if (!factorized_) {
        throw std::runtime_error("KktSolver::solve called on unfactorized system");
    }

    std::vector<double> work(total_dim_, 0.0);
    for (std::size_t j = 0; j < n_ && j < rhs_x.size(); ++j) {
        work[j] = rhs_x[j];
    }
    for (std::size_t i = 0; i < m_ && i < rhs_z.size(); ++i) {
        work[n_ + i] = rhs_z[i];
    }

    // 1. Forward substitution: L * v = rhs
    ldl_lsolve(total_dim_, work, L_col_ptr_, L_row_ind_, L_val_);

    // 2. Diagonal solve: D * w = v
    ldl_dsolve(total_dim_, work, D_);

    // 3. Backward substitution: L^T * [x; nu] = w
    ldl_ltsolve(total_dim_, work, L_col_ptr_, L_row_ind_, L_val_);

    // Extract solutions
    sol_x.resize(n_);
    for (std::size_t j = 0; j < n_; ++j) {
        sol_x[j] = work[j];
    }

    sol_nu.resize(m_);
    for (std::size_t i = 0; i < m_; ++i) {
        sol_nu[i] = work[n_ + i];
    }
}

std::size_t KktSolver::nonzeros_L() const noexcept {
    return L_col_ptr_.empty() ? 0 : L_col_ptr_.back();
}

} // namespace markov_cero::qp
