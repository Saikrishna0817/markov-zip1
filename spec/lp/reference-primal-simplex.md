# Reference primal revised simplex — SPEC-M3-LP-01

## Contract
The M3 oracle solves the continuous standard-form minimization problem

A x = b, x >= 0, minimize c^T x + c0.

It consumes only a validated M2 `CanonicalModel`. Original bounded variables are represented by M2 affine maps and explicit range rows. Consequently, native upper-bound flips are not required in this standard-form oracle; the telemetry field is retained and remains zero. Native bounded-variable flips belong to the production bounded simplex path.

## Phase I
Rows with negative right-hand sides are multiplied by -1. Artificial identity columns provide an initial basic feasible solution for the auxiliary objective minimize sum(a). If its optimum exceeds feasibility tolerance, the phase-I dual vector is returned as a Farkas certificate satisfying A^T y <= 0 and b^T y > 0.

Zero artificial basics are pivoted out when possible. Rows with no eligible original pivot are redundant and removed from the internal working system.

## Phase II
At each iteration, factor B by the independently tested M2 dense LU oracle, compute x_B=B^-1 b and y=B^-T c_B, and evaluate reduced costs r_j=c_j-a_j^T y. A negative reduced cost enters. The direction d=B^-1 a_j determines the ratio test. If d has no positive component, the returned ray has entering component one and basic components -d.

Bland ordering is the deterministic anti-cycling default. Every iteration records phase, objective, minimum reduced cost, entering/leaving variables, and degeneracy.

## Status policy
`Optimal`, `Unbounded`, and `Infeasible` are accepted only when the independent M3 verifier validates the primal/dual conditions, improving ray, or Farkas certificate. Limits and numerical failures are non-conclusive.

## Review hardening
The minimum-ratio test never treats a strictly larger floating-point ratio as a tie. Ratios must be finite; overflow returns `NumericalFailure`. Bland's leaving-index rule applies only to exactly equal computed minima. A direction with any positive component that cannot produce a finite admissible ratio is not an unboundedness proof.

Success statuses are passed through the independent verifier inside the public solve boundary and downgraded to `NumericalFailure` if the witness fails. Unbounded rays require exactly nonnegative components and a homogeneous row residual scaled by the absolute row activity, without an absolute-one floor.

The reference engine admits at most 1,024 rows, 8,192 original columns, 4,194,304 expanded dense elements, 1,000,000 iterations, and 10,000 retained telemetry records. Dimension addition and multiplication are checked before allocation. Tolerances must be positive, finite, at most 1e-4, and pivot tolerance cannot exceed feasibility tolerance. Telemetry beyond the configured retained limit is discarded and reported with `telemetry_truncated`.
