# Known failures and blockers

## Current prototype (M5 RC)

1. Independent mathematical and source/security reviewers have not signed the exact candidate commit.
2. GPU, MILP, convex QP, and reversible presolve are **not implemented**. They are future milestones (M6–M11), not missing bugfixes.
3. Dual leaving-row pricing that is not Bland uses a **tableau-row norm** `||A^T B^{-T} e_i||^2`. That is not conventional dual steepest-edge `||B^{-T} e_i||^2`.
4. Canonical models remain dense; sparse LU is the basis substrate only. Production sparse performance is not claimed.
5. Incremental dual steepest-edge updates are not implemented (O(rows²) tableau-norm work per pivot, capped at 1024 rows).
6. Strict MPS subset: no fixed-column MPS, no vendor extensions.
7. Commercial-oracle access, judging logistics, and a first live MRPL dataset remain unresolved.
8. Exact paper section/equation mapping stays blocked where no pinned lawful full text exists.
9. NumericalPolicy defaults remain provisional pending independent review.
10. QP Hessian / PSD policy is unspecified because QP is not in this prototype.

## Historical M0–M1 notes

M0 contained no solver by design; that statement is no longer the current product state. M3–M5 implement certified continuous simplex.

Earlier sandbox notes (missing CMake/Clang in some audit environments) do not describe this deployment machine, which has GCC, Clang, CMake, and CTest.
