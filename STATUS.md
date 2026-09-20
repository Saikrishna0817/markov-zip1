# Status & Capability Matrix — markov-cero v0.5.2

Single source of truth for solver status, capabilities, CLI options, and prototype boundaries.

---

## 1. Implemented Capabilities

| Capability | Status | Implementation / Evidence |
|---|---|---|
| Free-format MPS parser | Implemented | `src/io/mps.cpp`, fuzz targets, edge tests |
| Immutable model (CSC) | Implemented | `src/model/model.cpp` |
| Sparse canonical model | Implemented | `src/transform/sparse_canonicalize.cpp` |
| Primal revised simplex | Implemented | `src/lp/reference/revised_simplex.cpp`, Netlib pass |
| Dual warm simplex | Implemented | `src/lp/dual/dual_simplex.cpp`, basis serialization |
| Sparse basis LU + Eta updates | Implemented | `src/linalg/sparse_basis.cpp`, property tests |
| Matrix-free PDLP (CPU) | Implemented | `src/lp/first_order/pdlp.cpp`, `tests/pdlp_test.cpp` |
| Sovereign MILP Branch-and-Cut | Implemented | `src/milp/milp_solver.cpp`, 3/3 MIPLIB pass |
| Parallel tree search | Implemented | `src/milp/parallel_tree_search.cpp`, C++20 jthread |
| Gomory Mixed-Integer (GMI) cuts | Implemented | `src/milp/gomory.cpp` with slack substitution |
| Mixed-Integer Rounding (MIR) cuts | Implemented | `src/milp/mir.cpp`, cosine filtering |
| Strong branching | Implemented | `src/milp/strong_branching.cpp`, pseudo-costs |
| Primal heuristics & pump | Implemented | `src/milp/heuristics.cpp` with cycle perturbation |
| Reversible presolve / postsolve | Implemented | `src/presolve/presolve.cpp`, `presolve_test.cpp` |
| Ruiz matrix scaling | Implemented | `src/scale/ruiz_scaling.cpp`, `ruiz_scaling_test.cpp` |
| Independent canonical verifier | Implemented | `src/verify/reference_lp_verifier.cpp` |
| Original primal verifier | Implemented | `src/verify/primal_verifier.cpp` |
| Solve CLI & JSON output | Implemented | `apps/markov_cero_solve.cpp` |
| Refinery qualification demo | Implemented | `examples/refinery/`, `run-qualification-demo.sh` |

---

## 2. Explicit Prototype Limitations

JSON telemetry emitted by `markov-cero-solve` reports:
```json
"limitations":"CPU sovereign LP and MILP Branch-and-Cut engine; GPU and QP are not implemented."
```

The following capabilities are **not implemented** in the current release:

1. **GPU Acceleration**:
   - **Status**: **Not implemented** (Zero CUDA files in solver core).
   - **Roadmap**: Scheduled for [Phase 5 (GPU Acceleration)](docs/gpu.md).
   - **Strategy**: Offload matrix-free PDHG (SpMV, vector axpy, reductions) to CUDA.

2. **Convex Quadratic Programming (QP)**:
   - **Status**: **Not implemented**.
   - **Roadmap**: Scheduled for [Phase 6 (Convex QP)](#phase-6-convex-qp-roadmap).
   - **Strategy**: ADMM operator-splitting (OSQP-style) with sparse KKT factorizations.

3. **Machine Learning-Assisted Branching**:
   - **Status**: **Not implemented**.
   - **Roadmap**: Scheduled for Phase 7.
   - **Strategy**: Offline-trained gradient-boosted tree ranker on strong branching scores.

4. **Dual Steepest-Edge Pricing Recurrence**:
   - **Status**: Recomputes full tableau norm in $O(m^2)$ work per pivot.
   - **Roadmap**: Scheduled for Phase 8 ($O(m)$ Forrest–Goldfarb recurrence).

---

## 3. Roadmaps for Deferred Capabilities

### Phase 5: GPU Acceleration Roadmap
- Detailed roadmap and design: [docs/gpu.md](docs/gpu.md).
- Native C++20 matrix-free PDLP engine (`src/lp/first_order/pdlp.cpp`) serves as the CPU on-ramp.
- Target: CUDA kernel SpMV (`A * x`, `A^T * y`), vector axpy, and dot-product reductions.
- Primal revised simplex remains CPU-bound due to serial sparse basis updates.

### Phase 6: Convex QP Roadmap
- Target: Convex quadratic programs ($\min \frac{1}{2} x^T Q x + c^T x$ with $l \le A x \le u$).
- Parser: Add strict `QUADOBJ` / `QMATRIX` support to the MPS parser (`src/io/mps.cpp`).
- Algorithm: ADMM operator-splitting with quasi-definite KKT linear system solves.
- Verification: Independent KKT certificate verifier checking primal/dual residuals and
  complementarity conditions.

---

## 4. MPS Parser Dialect & Boundaries

- **Supported Sections**: `NAME`, `ROWS`, `COLUMNS`, `RHS`, `RANGES`, `BOUNDS`, and `ENDATA`.
- **Integer Markers**: Binary and general integers via `'INTORG'`, `'INTEND'`, `BV`, `UI`, `LI`.
- **Security Boundaries**: Hostile input safeguards with bounded memory allocation, checked
  arithmetic dimensions, and RFC 8259-compliant JSON telemetry.

---

## 5. CLI Interface (`markov-cero-solve --help`)

The CLI interface matches `markov-cero-solve --help` exactly:

```
usage: markov-cero-solve MODEL.mps [options]
options:
  --output result.json     Write output JSON to file
  --engine primal|dual|pdlp|milp|parallel|auto Select solver engine (default: auto)
  --threads N              Worker threads for parallel tree search (default: 4)
  --branching most_fractional|pseudo_cost|strong_branching|reliability Branching variable selection rule (default: pseudo_cost)
  --iteration-limit N      Maximum simplex iterations
  --max-nodes N            Maximum branch-and-cut search nodes (default: 50000)
  --time-limit SEC         Maximum search time limit in seconds (default: 60.0)
  --cuts, --no-cuts        Enable or disable Gomory & MIR mixed-integer cuts (default: enabled)
  --heuristics, --no-heuristics Enable or disable primal heuristics (default: enabled)
  --warm-start FILE        Load warm-start basis from file (dual engine)
  --save-basis FILE        Save optimal basis to file
  --presolve, --no-presolve Enable or disable presolve reductions (default: enabled)
  --scale, --no-scale       Enable or disable Ruiz matrix scaling (default: enabled)
  --max-presolve-passes N   Maximum presolve passes (default: 5)
  --ruiz-iterations N       Maximum Ruiz equilibration iterations (default: 10)
  --help, -h               Show this help
```

---

## 6. Exit Codes

`markov-cero-solve` returns deterministic exit codes:

| Code | Status | Meaning |
|---|---|---|
| `0` | Optimal | Verified optimal solution found |
| `1` | Infeasible | Certified infeasible by Farkas certificate / dual ray |
| `2` | Unbounded | Certified unbounded by primal ray |
| `3` | InvalidModel | Parse error, duplicate names, or malformed MPS |
| `4` | InvalidOptions | Invalid command-line arguments or parameters |
| `5` | ResourceLimit | Node limit or time limit exceeded |
| `6` | IterationLimit | Iteration limit reached without optimality |
| `7` | NumericalFailure | Singular basis, numerical drift, or precision loss |
| `8` | Usage / I/O | File not found, unreadable path, or help requested |
