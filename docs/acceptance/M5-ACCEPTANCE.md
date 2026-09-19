# M5 acceptance record

Status: **release candidate; qualification CLI in progress; M6 not authorized**.
Current identity: version `0.5.1`, milestone M5. Solver algorithms exist; a user-facing `markov-cero-solve` path is part of the qualification prototype. GPU/MILP/QP remain out of scope.

M5 must provide canonical sparse basis storage, deterministic sparse LU, sparse and reach-restricted FTRAN/BTRAN, bounded product-form column updates, refactorization triggers, M4 warm-dual integration, cumulative verification, deployment-machine CMake/GCC/Clang builds, ASan/UBSan, coverage-guided fuzzing, and independent mathematical and source/security reviews.

## Locally completed
- Original sparse CSC, sparse partial-pivot LU, row/column factor adjacency, FTRAN/BTRAN, and finite/residual checks.
- Reach propagation skips triangular equations outside a sparse RHS dependency graph.
- Product-form eta updates use chronological FTRAN and reverse-order BTRAN.
- Count and density refactorization triggers plus pivot and fill ceilings.
- M4 dual warm path uses the sparse substrate and retains certified M3 cold fallback and independent result verification.
- 250 direct random sparse factorization cases, 2,400 accepted multi-update sequences with repeated FTRAN/BTRAN comparisons, row-pivot examples, malformed-input tests, cumulative M0-M4 regression, and local GCC ASan/UBSan passed.

## Blocking final acceptance
- Isolated reviewer sessions did not share `/data/markov-cero`; both reported repository-unavailable blockers and could not produce independent sign-off.
- Evidence JSON may still name an older candidate commit until the qualification freeze regenerates it from HEAD.

M5 remains the active milestone. M6 (presolve) is not authorized until the CPU-LP qualification prototype is frozen. Final packaging requires `scripts/m5-deployment-gate.sh` plus independent reviews of the exact freeze commit.
