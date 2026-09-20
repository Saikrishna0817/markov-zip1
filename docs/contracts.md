# Subsystem Contracts & Interface Invariants — markov-cero

Single source of truth for solver component contracts, numerical policies, and invariants.

---

## 1. Model & I/O Contracts

### Free-Format MPS Parser Contract (`src/io/mps.cpp`)
- **Supported Sections**: `NAME`, `OBJSENSE`, `ROWS`, `COLUMNS`, `RHS`, `RANGES`, `BOUNDS`, `ENDATA`.
- **Integrality Markers**: `MARKER 'INTORG'` activates integer typing; `'INTEND'` terminates it.
- **Bound Types**: `LO` (lower bound), `UP` (upper bound), `FX` (fixed), `FR` (free: $[-\infty, \infty]$),
  `BV` (binary: $[0, 1]$), `UI` (upper integer bound), `LI` (lower integer bound).
- **Invariants**: All indices bounded by parsed column/row limits. No memory allocation exceeds
  input file size times a fixed constant factor. Duplicate identifiers trigger typed parse errors.

### Immutable Model Contract (`src/model/model.cpp`)
- Matrix representation is strictly Compressed Sparse Column (CSC):
  - `column_offsets` has size `columns + 1` with `column_offsets[0] == 0`.
  - `row_indices` and `values` have size `column_offsets[columns]`.
  - Row indices within each column are strictly monotonically increasing.
- All variable bounds satisfy $l_j \le u_j$. Infeasible bounds ($l_j > u_j$) are rejected at model construction.

---

## 2. Linear Algebra & Basis Factorization Contracts

### Sparse Basis Factorization (`src/linalg/sparse_basis.cpp`)
- Factorization guarantees $P B Q = L U$ with unit lower-triangular $L$ and upper-triangular $U$.
- Eta matrix chain:
  - Maximum update chain length is strictly bounded (default: 50 updates).
  - Density threshold trigger: refactorization forced when Eta nonzeros exceed 3x original factor nonzeros.
  - Updates are transactional: if a numerical failure occurs during an update, the basis state rolls
    back to the previous valid factorization.

---

## 3. Solver Engine Contracts

### Continuous Simplex Contracts (`src/lp`)
- **Status Certifications**:
  - `Optimal`: Primal feasible, dual feasible, and complementary slackness verified.
  - `Infeasible`: Certified by Farkas dual ray $y$ with $A^T y \le 0$ and $b^T y > 0$.
  - `Unbounded`: Certified by primal direction vector $d$ with $A d = 0$, $d \ge 0$, and $c^T d < 0$.
- **Pivot Rule**:
  - Primal: Bland's anti-cycling rule (smallest index among eligible candidates).
  - Dual: Harris two-pass ratio test with dynamic pivoting threshold.

### MILP Branch-and-Cut Contracts (`src/milp`)
- **Integrality**:
  - Variable $x_j$ is declared integer if and only if marked in model metadata.
  - An LP relaxation solution is integer-feasible if $\max_{j \in I} |x_j - \text{round}(x_j)| \le 10^{-5}$.
- **Cutting Planes**:
  - GMI and MIR cuts must be valid globally across the entire branch-and-bound tree.
  - Orthogonality filter rejects parallel cuts ($\cos \theta > 0.999$) and cuts with bad numerical dynamics.

---

## 4. Numerical Policies & Tolerances

| Tolerance | Value | Meaning |
|---|---|---|
| Feasibility ($\epsilon_{\text{feas}}$) | $10^{-7}$ | Maximum absolute constraint violation $\|Ax - b\|_\infty$ |
| Dual Feasibility ($\epsilon_{\text{dual}}$) | $10^{-7}$ | Maximum absolute reduced cost violation |
| Optimality Gap ($\epsilon_{\text{opt}}$) | $10^{-7}$ | Relative objective gap $|c^T x - b^T y| / (1 + |c^T x|)$ |
| Pivot Zero ($\epsilon_{\text{pivot}}$) | $10^{-12}$ | Numerical threshold below which pivot elements are zeroed |
| Integrality ($\epsilon_{\text{int}}$) | $10^{-5}$ | Maximum distance from nearest integer for integer variables |
| Ruiz Scaling Clamp | $[10^{-4}, 10^4]$ | Bounds on allowable diagonal equilibration factors |
