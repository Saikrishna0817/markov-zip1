# Changelog

## 0.5.1 — 2026-09-19

- Stopped tracking CMake `_m5-*` build trees; added `.gitignore`.
- Tightened `no_solver_guard.py` (still forbids MIP/QP/GPU/PDLP; allows `src/lp/reference` and `src/lp/dual`).
- Reformatted M3/M4 simplex sources; renamed dual pricing `tableau_norm`.
- Added `markov-cero-solve`, refinery qualification models, and `run-qualification-demo.sh`.
- Documented current M5 capability vs planned M6–M11. GPU/MILP/QP remain unimplemented.

## 0.0.1 — 2026-09-13

- Established Apache-2.0 clean-room governance.
- Added canonical LP/QP, status/certificate, and numerical-policy specifications.
- Added threat, dependency-license, provenance, research, competitor, benchmark, evidence, and documentation schemas.
- Added CMake/CTest foundation and deterministic release verification/packaging scripts.
- Added no-solver guard and M0 acceptance report.

## 0.1.0 — 2026-09-13

- Added immutable model and CSC validation.
- Added strict MPS subset and diagnostics.
- Added independent primal/objective/integrality verifier.
- Added CLI, Python inspection path, tests, and fuzz target.

## 0.2.0 — 2026-09-13

- Added dense partial-pivoting LU, FTRAN/BTRAN, diagnostics, and residual checks.
- Added reversible LP canonicalization for objective sense, row senses, fixed/free/bounded variables, slacks, and postsolve.
- Added analytic and randomized M2 oracle tests.
