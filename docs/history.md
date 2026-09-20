# Development History & Milestone Lineage

Chronological record of markov-cero development, architectural evolution, and audit remediations.

---

## 1. Chronological Milestones

### Milestone M0: Clean-Room Foundation & Governance (2026-09-13)
- Established clean-room C++20 architecture with strict zero-dependency policy.
- Implemented build metadata emission, foundation contracts, and deterministic packaging.
- Outlawed all external solver libraries (HiGHS, GLPK, Clp, CPLEX, Gurobi, etc.).

### Milestone M1: Model Representation & Parser (2026-09-14)
- Developed strict free-format MPS parser with bounds checking and memory ceilings.
- Implemented immutable Compressed Sparse Column (CSC) model representation.
- Built independent original primal feasibility verifier (`verify_primal`).

### Milestone M2: Canonicalization & Dense Oracle (2026-09-15)
- Standardized linear programs into canonical standard equality form ($Ax = b, x \ge 0$).
- Implemented reference dense Gaussian LU factorization oracle with partial pivoting.
- Verified canonical mapping bijection and sign preservation.

### Milestone M3: Certified Primal Revised Simplex (2026-09-16)
- Implemented certified primal revised simplex engine with Bland's anti-cycling rule.
- Added Farkas dual ray certification for certified infeasibility detection.
- Introduced dual-gated verification: canonical witness and independent model-space checks.

### Milestone M4: Dual Warm Simplex & Serialization (2026-09-17)
- Implemented dual revised simplex with Harris two-pass ratio test for numerical stability.
- Built basis state serialization and warm-starting for rapid re-optimization.
- Validated hot/cold parity across randomized perturbation benchmarks.

### Milestone M5: Sparse Basis Linear Algebra (2026-09-18)
- Developed sparse LU factorization with product-form Eta column updates.
- Added reach-restricted FTRAN/BTRAN skipping graph-disconnected triangular components.
- Implemented density and update-count refactorization triggers.

### Phase 3: Sovereign MILP Branch-and-Cut (2026-09-19)
- Built sovereign Mixed-Integer Linear Programming Branch-and-Cut solver (`MilpSolver`).
- Implemented Gomory Mixed-Integer (GMI) cutting plane generator with slack substitution.
- Added search tree management (best-bound, DFS, plunge) and dual-tier primal heuristics.
- Validated on canonical MIPLIB benchmark problems (`stein9`, `stein15`, `flugpl`).

### Phase 4: Multi-Engine Expansion (2026-09-19)
- Implemented matrix-free First-Order PDLP (Primal-Dual Hybrid Gradient) for large-scale LPs.
- Added Mixed-Integer Rounding (MIR) cutting planes with cosine orthogonal filtering.
- Implemented strong branching candidate scoring and reliability pseudo-costs.
- Built parallel branch-and-cut tree search using C++20 `std::jthread`.
- Added reversible multi-pass presolve and Ruiz $\ell_\infty$ condition equilibration scaling.

---

## 2. Critical Audit Remediation (The False Optimal / Infeasible Story)

During internal mathematical audits prior to v0.5.2, two critical certification vulnerabilities
were uncovered in edge-case models:

1. **False Optimal Vulnerability**:
   - *Defect*: Pivot tolerance thresholding allowed tiny reduced costs ($< 10^{-11}$) to be
     classified as optimal even when residual drift violated dual feasibility.
   - *Fix*: Introduced scale-aware dual tolerance verification enforcing strict residual bounds
     relative to matrix norm and objective magnitude.

2. **False Infeasible Vulnerability**:
   - *Defect*: Unbounded Farkas ray certificates with near-zero norm were incorrectly accepted
     as certifying problem infeasibility.
   - *Fix*: Enforced strict non-zero ray certificates ($\|c_B^T B^{-1} A - c\|_\infty > \epsilon$)
     and mandatory dual ray verification against canonical bounds.

3. **Sparse Update State Transactionality**:
   - *Defect*: Failed basis update operations could leave the Eta chain in an inconsistent state.
   - *Fix*: Made basis factorization updates strictly transactional—failed pivots roll back
     the update counter and preserve factorization integrity.

These fixes were captured in permanent audit regression tests (`tests/regression_test.cpp`)
and are verified on every CI run.
