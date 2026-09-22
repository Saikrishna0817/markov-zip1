# ADR-P3-01 — Primal revised simplex (math check)

**Status:** pass (hand derivation matches code)  
**Date:** 2026-09-22  
**Instance:** `tiny_simplex` (2 vars, 2 inequalities)

## Problem (minimization MPS form)

$$
\min -3x_1 - 2x_2
\quad\text{s.t.}\quad
x_1+x_2 \le 4,\quad
2x_1+x_2 \le 5,\quad
x \ge 0
$$

## Hand derivation

Feasible vertices and objectives:

| Point | Objective |
|-------|-----------|
| (0,0) | 0 |
| (0,4) | −8 |
| (2.5,0) | −7.5 |
| (1,3) | **−9** |

Intersection of $x_1+x_2=4$ and $2x_1+x_2=5$: $x_1=1$, $x_2=3$.  
No better feasible vertex exists → unique optimum $(1,3)$, objective $-9$.

## Code (measured)

```
./build/markov-cero-solve tiny_simplex.mps --engine primal
```

- status: Optimal  
- objective: −9  
- primal: ≈ (1, 3)  
- verified: true  

## Verdict

**Pass.** Algebra and solver agree within floating-point noise.
