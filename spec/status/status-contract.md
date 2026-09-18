# Status and certificate contract — SPEC-M0-STATUS-01

Certified terminal statuses: `Optimal`, `Infeasible`, `Unbounded`, and provisionally `OptimalWithinGap` for MILP.

Operational/non-conclusive statuses: `Feasible`, `IterationLimit`, `TimeLimit`, `NodeLimit`, `MemoryLimit`, `Interrupted`, `NumericalFailure`, `Unsupported`, `InvalidModel`, `InvalidWarmStart`, `NonConvex`, `Unknown`, `InternalError`.

A certified status requires independent verification on the original model. `Optimal` requires finite values, primal/dual feasibility, stationarity or reduced costs, complementarity, objective recomputation, and acceptable gap. `Infeasible` requires a validated alternative-system witness. `Unbounded` requires a validated feasible improving recession ray. MILP proof requires a verified incumbent, open-node/global-bound accounting, and approved gap. Engine suspicion without a valid witness is non-conclusive.
