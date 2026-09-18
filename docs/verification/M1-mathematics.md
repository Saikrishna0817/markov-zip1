# M1 mathematics and verification

For candidate x, row activity is a=Ax and objective is c^T x+c0. The verifier checks variable and row lower/upper violations with absolute-plus-relative allowances, recomputes the objective, and checks integrality distance |x-round(x)| for integer variables.

M1 proves only primal feasibility, objective consistency, and integrality consistency. It cannot certify Optimal, Infeasible, or Unbounded because dual multipliers, Farkas witnesses, and rays are not yet produced.

Beginner example: for 50 units each of two crudes, total is 100 and sulfur is 2; the verifier accepts the claimed cost 3500. It rejects 100 units of the high-sulfur crude and rejects a corrupted objective of 3499.
