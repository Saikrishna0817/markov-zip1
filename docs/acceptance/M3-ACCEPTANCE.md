# M3 acceptance record

Candidate scope: dense reference primal revised simplex with crash basis, Phase I/II, deterministic Bland anti-cycling, optional Dantzig pricing, ratio tests, redundant-row cleanup, optimal dual payload, Farkas certificate, improving unbounded ray, independent verifier, and iteration telemetry.

Native upper-bound flips are not applicable to the M2 standard-form input because finite upper bounds are explicit equations with nonnegative slacks. The result retains a bound-flip statistic fixed at zero. Native bounded-variable simplex remains outside M3.

Required local evidence: cumulative GCC C++20 warnings-as-errors, analytic status tests, cycling and degeneracy tests, randomized exact differential tests, row-permutation metamorphism, corrupted witness rejection, ASan/UBSan, deterministic package reproduction, fresh extraction, and independent mathematical/security review.

Deployment CMake/Clang remains deferred to M5 by explicit approval. M4 may not begin before M3 sign-off and explicit approval.

Review remediation requires: strict-minimum ratio regression, ratio-overflow fail-closed regression, excessive-tolerance rejection, bounded telemetry with truncation disclosure, solver-local expanded-workspace limits, independent verification before certified status return, scale-aware homogeneous ray checks, complete M3 documentation-manifest records, and an external Git bundle plus release attestation for commit verification.
