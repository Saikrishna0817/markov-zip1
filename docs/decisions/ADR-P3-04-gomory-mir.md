# ADR-P3-04 — Gomory / MIR cuts (math check)

**Status:** pass (integer optimum + cut fired)  
**Date:** 2026-09-22  
**Instance:** `tiny_milp`

## Problem

$$
\min -x \quad\text{s.t.}\quad 2x \le 3,\quad x \in \mathbb{Z}_{\ge 0},\quad x \le 10
$$

## Hand derivation

LP relaxation: $x \le 1.5$, objective $-1.5$.  
From $2x + s = 3$: $x + 0.5 s = 1.5$ → Gomory: $x \le 1$.  
Integer optimum: $x=1$, objective $-1$.

## Code (measured)

```
./build/markov-cero-solve tiny_milp.mps --engine milp
```

- status: Optimal  
- objective: −1  
- primal: [1]  
- cuts_generated: 1  
- nodes_explored: 1  
- verified: true  

## Verdict

**Pass.** Integer optimum matches; one cut closed the gap at the root.  
**Not claimed:** isolation of GMI vs MIR coefficient vectors (telemetry does not name the family).
