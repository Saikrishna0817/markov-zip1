# Convex QP contract — SPEC-M0-QP-01

The intended model is `min 0.5*x^T*P*x + q^T*x + c0` subject to the LP row and variable bounds. Stationarity is `P*x + q + A^T(alpha-beta) + gamma-delta = 0`.

M11 is blocked until an approved ADR freezes Hessian storage triangle, symmetry scales, PSD test, near-indefinite classification, nonconvex rejection, regularization semantics, objective interpretation, and KKT residual scaling. A materially changed model must never be solved silently.
