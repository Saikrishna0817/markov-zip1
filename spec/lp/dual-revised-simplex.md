# Dual revised simplex — SPEC-M4-LP-01

M4 reoptimizes the M2 standard-form problem `min c^T x + c0`, `Ax=b`, `x>=0` from a validated dual-feasible basis. A cold request is bootstrapped by the independently verified M3 oracle; RHS-only hot starts reuse an M4 basis because the matrix and objective fingerprint is unchanged.

For basis B, compute `xB=B^-1 b`, `y=B^-T cB`, and reduced costs `r=c-A^T y`. Dual feasibility is `r>=0`. If a basic value is negative, choose a leaving row. For its tableau row `alpha=e_p^T B^-1 A`, an entering nonbasic column must satisfy `alpha_j<0`. The exact dual ratio is `r_j/(-alpha_j)`.

The conservative Harris mode first computes a relaxed admissible bound using dual tolerance, then selects the largest safe pivot from the admissible window. Every candidate basis is refactorized, and a final independent verifier must accept any certified status. With no eligible entering column, `-B^-T e_p` is a Farkas certificate.

Tableau-norm leaving-row weights `||A^T B^{-T} e_i||^2` are recomputed each iteration. This is not conventional dual steepest-edge `||B^{-T} e_i||^2`. Bland mode is available for deterministic least-index selection. M5 replaces per-iteration dense refactorization with sparse updates.
