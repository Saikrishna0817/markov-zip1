# ADR-P3-05 — PDHG / PDLP (math check)

**Status:** pass within tolerance  
**Date:** 2026-09-22  
**Instance:** same `tiny_simplex` as ADR-P3-01

## Hand derivation

Same LP optimum $(1,3)$, objective $-9$ (see ADR-P3-01).  
With relative KKT tolerance $10^{-6}$, first-order PDHG may leave a small residual near the vertex.

## Code (measured)

```
./build/markov-cero-solve tiny_simplex.mps --engine pdlp --tolerance 1e-6
```

- status: Optimal  
- objective: ≈ −9.0000013  
- primal: ≈ (1.000008, 2.999989)  
- message: PDLP converged  
- verified: true  

## Verdict

**Pass.** Within $O(10^{-5})$ of the exact vertex; consistent with the tolerance.
