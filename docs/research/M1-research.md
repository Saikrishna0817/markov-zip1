# M1 research interpretation

IBM and MOSEK MPS references define section roles, row types, bounds, ranges, integer markers, defaults, duplicate-entry behavior, and dialect differences. markov-cero deliberately supports a documented strict free-format subset and rejects ambiguity instead of guessing.

IEEE 754 motivates rejecting NaN/infinite coefficients while representing mathematical infinity as a bound kind. Higham motivates independent residual recomputation and higher-precision accumulation. CWE-190 motivates limits and checked dimensions before allocation. LLVM libFuzzer motivates a byte-array parser target with bounded work and no persistent state.

Exact edition/section mappings and reviewer approval remain required for final gate acceptance.
