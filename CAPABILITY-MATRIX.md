# Capability matrix — v0.5.2 / Phase 4 Sovereign Multi-Engine

| Capability | State | Evidence |
|---|---|---|
| Strict MPS free-format | Implemented | `src/io/mps.cpp`, M1 tests, fuzz targets |
| Continuous LP model (CSC) | Implemented | `src/model/model.cpp` |
| Canonicalization & Sparse CSC | Implemented | `src/transform/canonicalize.cpp`, `src/transform/sparse_canonicalize.cpp` |
| Primal revised simplex | Implemented | `src/lp/reference/revised_simplex.cpp`, 12/12 Netlib pass |
| Dual warm simplex | Implemented | `src/lp/dual/dual_simplex.cpp`, warm-start serialization |
| Sparse basis LU + eta updates | Implemented | `src/linalg/sparse_basis.cpp`, M5 property tests |
| Matrix-free First-Order PDLP | Implemented | `src/lp/first_order/pdlp.cpp`, `tests/pdlp_test.cpp` |
| Sovereign MILP Branch-and-Cut | Implemented | `src/milp/milp_solver.cpp`, 3/3 MIPLIB pass |
| Parallel Tree Search | Implemented | `src/milp/parallel_tree_search.cpp`, `tests/parallel_tree_search_test.cpp` |
| Gomory Mixed-Integer (GMI) Cuts | Implemented | `src/milp/cuts.cpp` with algebraic slack substitution |
| Mixed-Integer Rounding (MIR) Cuts | Implemented | `src/milp/cuts.cpp`, `tests/strong_branching_test.cpp` |
| Strong Branching & Pseudo-Costs | Implemented | `src/milp/strong_branching.cpp`, `src/milp/branch_selector.cpp` |
| Primal Heuristics & Pump | Implemented | `src/milp/heuristics.cpp` with cycle perturbation |
| Presolve / postsolve | Implemented | `src/presolve/presolve.cpp`, `tests/presolve_test.cpp` |
| Ruiz matrix equilibration | Implemented | `src/scale/ruiz_scaling.cpp`, `tests/ruiz_scaling_test.cpp` |
| Independent canonical verifier | Implemented | `src/verify/reference_lp_verifier.cpp` |
| Original primal verifier | Implemented | `src/verify/primal_verifier.cpp` |
| Solve CLI & JSON Telemetry | Implemented | `apps/markov_cero_solve.cpp` |
| Refinery demo (MRPL models) | Implemented | `examples/refinery/`, `run-qualification-demo.sh` |
| CUDA / GPU | Planned | Architecturally prepared via matrix-free PDLP |
| Convex QP | Planned | Planned future milestone |

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
