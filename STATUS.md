# Status & Capability Matrix — markov-cero v0.5.2

Single source of truth for solver status, capabilities, CLI options, and boundaries.

Status vocabulary (remediation plan):

| Tier | Meaning |
|------|---------|
| **Prototype** | Code exists, compiles, has at least one unit test |
| **Verified (N instances)** | Passes correctness checks on a named, counted benchmark set |
| **Hardware-verified** | Reproducible artifact generated on real target hardware with hardware spec recorded |

No performance number or "passed" claim appears here unless produced by a command that can be re-run.

---

## 1. Capability matrix

| Capability | Status | Evidence / notes |
|---|---|---|
| Free-format MPS parser | Verified (unit + fuzz smoke) | `src/io/mps.cpp`, `tests/mps_parser_test.cpp`, fuzz targets |
| Immutable model (CSC) | Prototype | `src/model/model.cpp`, model tests |
| Sparse canonical model | Prototype | `src/transform/sparse_canonicalize.cpp` |
| Primal revised simplex | Verified (7 Netlib instances in CI suite) | `src/lp/reference/revised_simplex.cpp`; `ctest -R netlib` |
| Dual revised simplex | Prototype | `src/lp/dual/dual_simplex.cpp`, dual/warm-start tests |
| Sparse basis LU + Eta | Prototype | `src/linalg/sparse_basis.cpp`, property tests |
| Matrix-free PDLP (CPU) | Prototype | `src/lp/first_order/pdlp.cpp`, `tests/pdlp_test.cpp` |
| MILP branch-and-cut | Verified (3 MIPLIB instances in CI suite) | `src/milp/milp_solver.cpp`; `ctest -R miplib` (stein9, stein15, flugpl) |
| Parallel tree search | Prototype | `src/milp/parallel_tree_search.cpp` |
| GMI cuts | Prototype | `src/milp/gomory.cpp` |
| MIR cuts | Prototype | `src/milp/mir.cpp` |
| Strong branching | Prototype | `src/milp/strong_branching.cpp` |
| Primal heuristics / pump | Prototype | `src/milp/heuristics.cpp` |
| Reversible presolve | Prototype | `src/presolve/presolve.cpp` |
| Ruiz scaling | Prototype | `src/scale/ruiz_scaling.cpp` |
| Independent verifiers | Prototype | `src/verify/` |
| Solve CLI + JSON | Prototype | `apps/markov_cero_solve.cpp` |
| Refinery demo models | Prototype | `examples/refinery/` |
| GPU PDLP path | Deferred (not in build) | CUDA path removed from active tree; restore from git history when hardware work resumes. Prior speedup claims must not be cited. |
| Convex QP (ADMM) | Prototype | `src/qp/`, `tests/qp_test.cpp` |
| Sparse LDLᵀ KKT | Prototype | `src/qp/kkt.cpp` |
| MIQP branch-and-cut | Prototype | Uses continuous QP relaxations at nodes |

**CI suite snapshot:** Netlib runner uses the 7 instances in `CMakeLists.txt`; MIPLIB uses stein9, stein15, flugpl. GPU targets are not built. Phase 2 regressions: `regression_simplex_scale200_pricing`, `regression_backend_actually_used`.

---

## 2. Explicit limitations

- GPU performance numbers previously published (including any "29,xxx×" speedup) are **not citeable**. They could not be verified as having been produced on real CUDA hardware with a trustworthy simplex baseline. Do not cite them in README, STATUS, CHANGELOG, or public claims. Replacement artifacts require a real NVIDIA device and a re-run (GPU work is deferred; code recoverable from git history).

- Dense primal reference simplex workspace is capped (`maximum_expanded_elements`); generated
  scale_2000 (~1050 rows after expansion) still returns ResourceLimit. Pricing path is fixed
  for the ~200-row regime (scale_200 regression). Full sparse industrial scale is out of Phase 2.
- Simplex scaling past moderate sizes has a dense reference dimension limit (see Phase 2 of the remediation plan).
- Full Netlib (~90) and MIPLIB benchmark-tag sets are not yet wired; only the small CI subsets above are claimed as Verified.

---

## 3. Deferred items

1. Machine-learning-assisted branching (roadmap Phase 7).
2. Dual steepest-edge pricing recurrence (roadmap Phase 8).
3. Hardware-verified GPU timing (deferred; requires real NVIDIA device; restore `gpu/` from git).

---

## 4. MPS dialect (parser)

Supported sections: `NAME`, `ROWS`, `COLUMNS`, `RHS`, `RANGES`, `BOUNDS`, `ENDATA`, plus quadratic `QUADOBJ` / `QMATRIX`.
Integer markers: `'INTORG'`, `'INTEND'`, `BV`, `UI`, `LI`.
Hostile-input safeguards: bounded allocation, checked dimensions.

---

## 5. CLI (`markov-cero-solve --help`)

```
usage: markov-cero-solve MODEL.mps [options]
options:
  --output result.json     Write output JSON to file
  --engine primal|dual|pdlp|milp|parallel|qp|miqp|auto
  --threads N              Worker threads for parallel tree search (default: 4)
  --branching RULE         most_fractional|pseudo_cost|strong_branching|reliability
  --iteration-limit N
  --max-nodes N            (default: 50000)
  --time-limit SEC         (default: 60.0)
  --cuts, --no-cuts
  --heuristics, --no-heuristics
  --warm-start FILE / --save-basis FILE
  --presolve, --no-presolve / --max-presolve-passes N
  --scale, --no-scale / --ruiz-iterations N
  --tolerance TOL          PDLP relative KKT tolerance (default: 1e-4)
  --backend cpu|gpu        PDLP backend (default: cpu; gpu currently runs CPU with cpu_fallback)
  --help, -h
```

---

## 6. Exit codes

| Code | Meaning |
|------|---------|
| 0 | Optimal (verified) |
| 1 | Infeasible |
| 2 | Unbounded |
| 3 | InvalidModel |
| 4 | InvalidOptions |
| 5 | ResourceLimit |
| 6 | IterationLimit |
| 7 | NumericalFailure |
| 8 | Usage / I/O |
