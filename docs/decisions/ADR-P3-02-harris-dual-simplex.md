# ADR-P3-02 — Dual simplex / Harris ratio test (math check)

**Status:** optimum pass; path telemetry incomplete  
**Date:** 2026-09-22  
**Instance:** same `tiny_simplex` as ADR-P3-01

## Hand derivation

Dual of the LP has the same optimal value −9 by strong duality.  
Harris two-pass ratio test is a pivot rule; it does not change the optimum of a non-degenerate LP that terminates optimally.

Expected: Optimal, objective −9, primal (1, 3).

## Code (measured)

```
./build/markov-cero-solve tiny_simplex.mps --engine dual
```

- engine field: dual  
- status: Optimal  
- objective: −9  
- primal: ≈ (1, 3)  
- message: `reference primal revised simplex optimum`  

## Verdict

**Optimum pass.** The message string is the primal-reference message, which suggests the dual CLI path may cold-start through the primal reference solver rather than exercising Harris pivots on this instance.

**Not claimed:** Harris ratio-test pivot sequence verified. Needs dual-only pivot log or unit test (follow-up).
