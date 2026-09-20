# Changelog

## 0.5.2 — 2026-09-19

### Phase 4: Sovereign Scaling & Audit Remediation (commit 23c8921)
- Implemented matrix-free first-order PDLP/PDHG solver with diagonal preconditioning.
- Implemented Mixed-Integer Rounding (MIR) cuts with cosine-similarity filtering.
- Implemented Strong Branching lookahead evaluation and domain reduction.
- Implemented multithreaded parallel tree search using C++20 std::jthread.
- Fixed Gomory cut generation by replacing slack discard with algebraic substitution.
- Added feasibility pump cycle prevention with ambiguous variable perturbation.
- Added CLI options: --engine pdlp|parallel, --threads, --branching rules.
- Synchronized capability documentation across README and CAPABILITY-MATRIX.

### Qualification Demo & Tooling Fixes (commit ec50874)
- Built markov-cero-info binary target and integrated with verification scripts.
- Updated qualification demo script and solver capability notice.

### Phase 3: Sovereign MILP Branch-and-Cut (commit 7dc74cd)
- Implemented sovereign branch-and-cut MILP solver with dual simplex node relaxations.
- Implemented node selection: best-bound, depth-first, and best-bound plunge.
- Implemented reliability pseudo-cost branching with most-fractional fallback.
- Implemented primal heuristics: simple rounding and feasibility pump.
- Implemented Gomory Mixed-Integer (GMI) cutting plane generator.
- Added MIPLIB benchmark runner and test suite (stein9, stein15, flugpl).
- Completed clean-room project rename to markov-cero across all sources.

## 0.5.1 — 2026-09-19

- Stopped tracking CMake `_m5-*` build trees; added `.gitignore`.
- Tightened `no_solver_guard.py` (forbids external solvers; allows clean-room lp).
- Reformatted M3/M4 simplex sources; renamed dual pricing `tableau_norm`.
- Added `markov-cero-solve`, refinery qualification models, and `run-qualification-demo.sh`.
- Documented current M5 capability vs planned M6–M11.

## 0.0.1 — 2026-09-13

- Established Apache-2.0 clean-room governance.
- Added canonical LP/QP, status/certificate, and numerical-policy specifications.
- Added threat, dependency-license, provenance, research, competitor,
  benchmark, evidence, and documentation schemas.
- Added CMake/CTest foundation and deterministic release verification/packaging scripts.
- Added no-solver guard and M0 acceptance report.

## 0.1.0 — 2026-09-13

- Added immutable model and CSC validation.
- Added strict MPS subset and diagnostics.
- Added independent primal/objective/integrality verifier.
- Added CLI, Python inspection path, tests, and fuzz target.

## 0.2.0 — 2026-09-13

- Added dense partial-pivoting LU, FTRAN/BTRAN, diagnostics, and residual checks.
- Added reversible LP canonicalization for objective sense, row senses,
  fixed/free/bounded variables, slacks, and postsolve.
- Added analytic and randomized M2 oracle tests.
