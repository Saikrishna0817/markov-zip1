# Architecture & System Design — markov-cero

Comprehensive technical specification of the markov-cero optimization solver architecture.

---

## 1. Architectural Principles & Invariants

1. **Clean-Room Sovereign Engineering**:
   Zero code, data, symbols, or bindings from third-party optimization libraries (HiGHS, GLPK,
   Clp, SCIP, Gurobi, CPLEX, etc.). Only standard C++20 and POSIX threads.
2. **Deterministic Execution**:
   Bit-exact reproducibility across runs under identical compiler and floating-point flags.
3. **Zero-Trust Verification**:
   All solutions and infeasibility/unboundedness certificates are validated by independent
   verifiers operating in original model space without sharing linear algebra solver state.

---

## 2. Layered Architectural Subsystems

```
+-------------------------------------------------------------------------+
|                  Applications / CLI (markov-cero-solve)                 |
+-------------------------------------------------------------------------+
|     Zero-Trust Verification (verify_primal, verify_reference_result)    |
+-------------------------------------------------------------------------+
| MILP Engine: Branch-and-Cut, GMI/MIR Cuts, Strong Branching, Heuristics |
| Parallel Tree Search (C++20 std::jthread, concurrent incumbent state)   |
+-------------------------------------------------------------------------+
| Continuous LP Engines:                                                  |
|   - Primal Revised Simplex (Bland rule anti-cycling)                    |
|   - Dual Revised Simplex (Harris ratio test, basis serialization)       |
|   - First-Order PDLP (Matrix-free Chambolle-Pock, diagonal scaling)     |
+-------------------------------------------------------------------------+
| Linear Algebra: Sparse LU + Product-form Eta updates, Dense LU Oracle  |
+-------------------------------------------------------------------------+
| Presolve & Scaling: Reversible LIFO reductions, Ruiz L-infinity scaling |
+-------------------------------------------------------------------------+
| Model & Parser: Strict free-format MPS parser, CSC Immutable Model     |
+-------------------------------------------------------------------------+
```

### Layer 1: Model & Input Representation (`src/model`, `src/io`)
- **MPS Parser (`src/io/mps.cpp`)**: RFC-style bounded free-format parser supporting `NAME`,
  `ROWS`, `COLUMNS`, `RHS`, `RANGES`, `BOUNDS` (including integer markers and variable bounds),
  and `ENDATA`.
- **Model Representation (`src/model/model.cpp`)**: Immutable Compressed Sparse Column (CSC)
  representation with column pointers, row indices, and double-precision coefficients.

### Layer 2: Transformation, Presolve & Scaling (`src/transform`, `src/presolve`, `src/scale`)
- **Canonicalization (`src/transform/canonicalize.cpp`, `sparse_canonicalize.cpp`)**: Converts
  arbitrary bounded linear programs into canonical standard equality form ($Ax = b, x \ge 0$).
- **Presolve (`src/presolve/presolve.cpp`)**: Multi-pass reduction removing empty rows/columns,
  fixing singleton bounds, and pushing inverse postsolve operations onto a typed LIFO stack.
- **Ruiz Scaling (`src/scale/ruiz_scaling.cpp`)**: Iterative $\ell_\infty$ equilibration scaling
  matrix rows and columns into $[1 - \epsilon, 1 + \epsilon]$ with $[10^{-4}, 10^4]$ safeguards.

### Layer 3: Linear Algebra Substrate (`src/linalg`)
- **Dense LU (`src/linalg/dense_lu.cpp`)**: Gaussian elimination with row partial pivoting used
  as numerical reference oracle and verification baseline.
- **Sparse Basis (`src/linalg/sparse_basis.cpp`)**: Sparse LU factorization with product-form Eta
  matrices, chronological forward transformation (FTRAN), reverse-order backward transformation
  (BTRAN), and fill-ratio refactorization triggers.

### Layer 4: Continuous LP Engines (`src/lp`)
- **Primal Revised Simplex (`src/lp/reference/revised_simplex.cpp`)**: Two-phase revised simplex
  with Bland's smallest-subscript rule preventing degeneracy cycles.
- **Dual Revised Simplex (`src/lp/dual/dual_simplex.cpp`)**: Dual simplex with Harris two-pass
  ratio test, basis state serialization, warm-starting, and Farkas certificate generation.
- **First-Order PDLP (`src/lp/first_order/pdlp.cpp`)**: Matrix-free Primal-Dual Hybrid Gradient
  (PDHG / Chambolle-Pock) executing sparse matrix-vector multiplications without matrix inversion.

### Layer 5: Mixed-Integer Programming Engine (`src/milp`)
- **Branch-and-Cut (`src/milp/milp_solver.cpp`)**: Tree search manager supporting best-bound,
  depth-first, and best-bound plunge node selection strategies.
- **Cutting Planes (`src/milp/cuts.cpp`)**: Gomory Mixed-Integer (GMI) cuts with algebraic slack
  substitution, and Mixed-Integer Rounding (MIR) cuts with cosine orthogonal filtering.
- **Branching Rules (`src/milp/strong_branching.cpp`, `branch_selector.cpp`)**: Reliability
  pseudo-costs with fallback to strong branching candidate score evaluation.
- **Primal Heuristics (`src/milp/heuristics.cpp`)**: Fast Simple Rounding and iterative
  Feasibility Pump with cycle-detection perturbation.
- **Parallel Tree Search (`src/milp/parallel_tree_search.cpp`)**: Multithreaded work-stealing
  tree search using C++20 `std::jthread` and atomic incumbent objective synchronization.

### Layer 6: Independent Zero-Trust Verification (`src/verify`)
- **Canonical Verifier (`src/verify/reference_lp_verifier.cpp`)**: Validates primal residual
  $\|Ax - b\|_\infty$, dual residual $\|A^T y + s - c\|_\infty$, and complementarity $x^T s$.
- **Original Model Verifier (`src/verify/primal_verifier.cpp`)**: Evaluates the computed solution
  vector directly against the unscaled, unpresolved original problem constraints and variable
  bounds in double precision.
