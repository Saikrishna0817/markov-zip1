# M3 mathematics and verification

For standard form min c^T x+c0 subject to Ax=b and x>=0, a basis B determines x_B=B^-1 b. Dual multipliers satisfy B^T y=c_B, and reduced costs are r=c-A^T y. Primal feasibility requires x_B>=0. Optimality requires r>=0. Complementarity follows because nonbasic variables are zero and basic reduced costs are zero.

An entering variable has r_j<0. Its direction is d=B^-1 a_j. If no d_i>0, x+tq remains feasible for all t>=0 with q_j=1 and q_B=-d, while c^Tq=r_j<0; this is a verified unbounded ray. Otherwise the minimum x_Bi/d_i preserves nonnegativity. The ratio test always preserves the strict numerical minimum; Bland’s basic-index rule is applied only when two computed ratios are exactly equal. This prevents a tolerance-sized but strictly larger step from violating primal feasibility while retaining deterministic anti-cycling for genuine ties.

Phase I minimizes the sum of artificial variables. If the optimum is positive, its dual y satisfies A^T y<=0 and b^T y>0, proving no x>=0 can satisfy Ax=b. The verifier checks this certificate on the original canonical rows after undoing internal row sign flips.

Analytic optimal/infeasible/unbounded cases, the Beale cycling example, degeneracy, redundant rows, empty systems, corrupt certificates, iteration limits, 200 random bounded two-variable LPs, exact basis enumeration, and row-permutation metamorphism are tested.

Independent adversarial review added two permanent counterexamples: a pair of close but unequal ratios where tolerance-based tie handling chose an infeasible step, and an overflowed ratio that could be mistaken for an unbounded direction. M3 now requires the strict computed minimum, finite ratios, and exact ray nonnegativity; homogeneous ray residuals are scaled by actual row activity.
