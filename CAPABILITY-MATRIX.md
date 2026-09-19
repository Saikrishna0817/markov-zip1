# Capability matrix — v0.5.1 / M5 qualification prototype

| Capability | State | Evidence |
|---|---|---|
| Strict MPS subset | Implemented | `src/io/mps.cpp`, M1 tests |
| Continuous LP model | Implemented | `src/model/model.cpp` |
| Canonicalization | Implemented | `src/transform/canonicalize.cpp` |
| Primal revised simplex | Implemented | `src/lp/reference/revised_simplex.cpp` |
| Dual warm simplex | Implemented | `src/lp/dual/dual_simplex.cpp` |
| Sparse basis LU + eta updates | Implemented | `src/linalg/sparse_basis.cpp` |
| Independent canonical verifier | Implemented | `src/verify/reference_lp_verifier.cpp` |
| Original primal verifier | Implemented | `src/verify/primal_verifier.cpp` |
| Solve CLI | Implemented | `apps/markov_cero_solve.cpp` |
| Refinery demo | Implemented | `examples/refinery/`, `run-qualification-demo.sh` |
| Dual steepest-edge (incremental) | **Not implemented** | Tableau-norm recomputed each pivot; M9 |
| Presolve / postsolve | Planned M6 | Guarded |
| CPU PDLP | Planned M7 | Guarded (`src/lp/pdlp`) |
| CUDA / GPU | Planned M8 | No number claimed; no CUDA in tree |
| MILP branch-and-bound | Planned M9 | Guarded (`src/mip`) |
| Cuts / heuristics | Planned M10 | Guarded |
| Convex QP | Planned M11 | Guarded (`src/qp`) |

## `markov-cero-solve` exit codes

| Code | Status |
|---|---|
| 0 | Optimal (verified) |
| 1 | Infeasible |
| 2 | Unbounded |
| 3 | InvalidModel |
| 4 | InvalidOptions |
| 5 | ResourceLimit |
| 6 | IterationLimit |
| 7 | NumericalFailure |
| 8 | Usage / I/O |
