# M5 mathematics and verification

## Sparse factorization contract
For nonsingular square basis `B`, M5 computes `P B = L U`, where `P` is a row permutation, `L` is unit lower triangular, and `U` is upper triangular. FTRAN solves `B x=b` by solving `L y=P b` and `U x=y`. BTRAN solves `B^T x=b` by solving `U^T y=b`, `L^T z=y`, and `x=P^T z`.

## Product-form update
Replacing basis column `p` by `a` gives `d=B^-1 a` and `B_new=B E`, where `E` is identity except column `p` equals `d`. The update is nonsingular exactly when `d_p` is nonzero. To solve `E x=y`, use `x_p=y_p/d_p` and `x_i=y_i-d_i x_p`. To solve `E^T z=b`, preserve `z_i=b_i` for `i!=p` and use `z_p=(b_p-sum_{i!=p} d_i b_i)/d_p`.

For a chain `B_k=B_0 E_1...E_k`, FTRAN applies eta inverses from `E_1` through `E_k`; BTRAN applies transpose inverses in reverse order before the base transpose solve. Reversing either order is a mathematical defect exposed by multi-update differential tests.

## Reach-based sparse triangular solves
A nonzero at node `i` activates only structurally reachable dependent equations through stored L/U adjacency. Zero RHS components outside this reach need not be evaluated. Correctness is unchanged because skipped equations have zero RHS and no active predecessor contribution.

## Verification strategy
- analytic 3x3 FTRAN/BTRAN and residual identities;
- row-pivot and singular examples;
- 250 deterministic random sparse diagonally dominant systems compared with M2 DenseLu;
- multi-update FTRAN/BTRAN compared with fresh dense and sparse refactorization;
- update/refactor trigger tests;
- malformed CSC, explicit zero, duplicate/unsorted index, non-finite, dimension, singularity, and pivot failures;
- cumulative M0-M4 regression tests;
- independent mathematical and source/security reviews.

## Residuals
`||B x-b||_infinity` and `||B^T x-b||_infinity` are accumulated with long-double products and reject every non-finite operand or result. A small residual supports a computed solve but is not an absolute proof for arbitrarily ill-conditioned matrices.

## Limits
Partial pivoting improves robustness but does not make every nonsingular floating-point matrix solvable. M5 returns a numerical failure for pivots below policy. There is no iterative refinement or condition estimator yet. DenseLu remains an independent small-instance oracle, not a hidden production backend.
