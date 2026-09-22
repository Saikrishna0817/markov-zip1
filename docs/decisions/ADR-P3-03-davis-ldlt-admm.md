# ADR-P3-03 — Davis LDLᵀ + ADMM QP (math check)

**Status:** pass  
**Date:** 2026-09-22  
**Instance:** `tiny_qp`

## Problem

$$
\min \tfrac12 x_1^2 + \tfrac12 x_2^2 - x_1 - x_2
\quad\text{s.t.}\quad
x_1+x_2 \le 1,\quad x \ge 0
$$

(P = I in MPS QUADOBJ; q = (−1,−1).)

## Hand derivation

Unconstrained stationarity: $x = (1,1)$, outside the feasible set.  
Active constraint $x_1+x_2=1$: substitute $x_2=1-x_1$,

$$
f(x_1)=\tfrac12 x_1^2 + \tfrac12(1-x_1)^2 - x_1 - (1-x_1) = x_1^2 - x_1 - \tfrac12.
$$

$f'(x_1)=2x_1-1=0 \Rightarrow x_1=\tfrac12$, $x_2=\tfrac12$.  
Objective: $\tfrac14 - 1 = -\tfrac34$.

## Code (measured)

```
./build/markov-cero-solve tiny_qp.mps --engine qp
```

- status: Optimal  
- objective: ≈ −0.74999  
- primal: ≈ (0.5, 0.5)  
- verified: true  

## Verdict

**Pass.** Matches $-3/4$ and $(1/2,1/2)$.
