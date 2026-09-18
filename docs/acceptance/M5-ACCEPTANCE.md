# M5 acceptance record

Status: **release candidate; final gate pending**.

M5 must provide canonical sparse basis storage, deterministic sparse LU, sparse and reach-restricted FTRAN/BTRAN, bounded product-form column updates, refactorization triggers, M4 warm-dual integration, cumulative verification, deployment-machine CMake/GCC/Clang builds, ASan/UBSan, coverage-guided fuzzing, and independent mathematical and source/security reviews.

## Locally completed
- Original sparse CSC, sparse partial-pivot LU, row/column factor adjacency, FTRAN/BTRAN, and finite/residual checks.
- Reach propagation skips triangular equations outside a sparse RHS dependency graph.
- Product-form eta updates use chronological FTRAN and reverse-order BTRAN.
- Count and density refactorization triggers plus pivot and fill ceilings.
- M4 dual warm path uses the sparse substrate and retains certified M3 cold fallback and independent result verification.
- 250 direct random sparse factorization cases, 2,400 accepted multi-update sequences with repeated FTRAN/BTRAN comparisons, row-pivot examples, malformed-input tests, cumulative M0-M4 regression, and local GCC ASan/UBSan passed.

## Blocking final acceptance
- This runtime has GCC but no CMake, CTest, or Clang. Therefore the required CMake/GCC/Clang matrix and Clang libFuzzer coverage-guided run could not execute.
- Isolated reviewer sessions did not share `/data/sihopt`; both reported repository-unavailable blockers and could not produce independent sign-off.

M5 must remain the active milestone. M6 is not authorized. Final M5 packaging and acceptance require running `scripts/m5-deployment-gate.sh` on an accessible deployment machine and completing both independent reviews against the exact candidate commit.

Candidate source commit: `69fdee4a423227e1647646a451db324044e69b1c`.
