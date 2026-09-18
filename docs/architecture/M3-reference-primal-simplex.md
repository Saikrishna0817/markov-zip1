# M3 reference primal revised simplex architecture

The M3 component is a deliberately dense, deterministic correctness oracle. It consumes the M2 continuous standard form and keeps solver state private. `revised_simplex.hpp` exposes options, truthful statuses, result/certificate payloads, basis indices, and typed iteration telemetry. `revised_simplex.cpp` owns the working matrix, row-sign map, Phase I artificial columns, basis factorization, pricing, ratio test, artificial removal, and ray construction.

The independent `reference_lp_verifier` depends only on the canonical model and public result. It does not call simplex internals. It verifies optimality through primal equality/nonnegativity, dual reduced costs, complementarity, and objective recomputation; unboundedness through a feasible anchor and improving recession ray; and infeasibility through a Farkas vector.

Complexity is intentionally O(iterations*(m^3+mn)) because the dense basis is refactorized each iteration. M3 has no sparse update, warm start, concurrency, GPU path, native bounded-variable flip, or production performance claim.

Independent review added solver-local checked size arithmetic and explicit limits before expanded or basis allocation. Iteration and telemetry budgets are independently capped. The solve boundary invokes the separate verifier before returning any certified status. An unsafe approximate ratio tie was replaced by a strict numerical minimum with exact-equality Bland tie-breaking.
