# M3 research interpretation record

## Primary sources
- George B. Dantzig, *Linear Programming and Extensions*, revised-simplex foundations.
- Robert G. Bland, “New finite pivoting rules for the simplex method,” Mathematics of Operations Research 2(2), 1977, DOI 10.1287/moor.2.2.103.
- John J. H. Forrest and Donald Goldfarb, “Steepest-edge simplex algorithms for linear programming,” Mathematical Programming 57, 1992.
- Qi Huangfu and J. A. Julian Hall, “Parallelizing the dual revised simplex method,” Mathematical Programming Computation 10, 2018; arXiv:1503.01889.

## Clean-room interpretation
M3 independently derives a small dense primal revised-simplex oracle from standard linear-algebra identities. No external solver source is used. Bland's least-index rule is used for deterministic anti-cycling. Dantzig pricing is available only when explicitly requested. The M2 LU oracle supplies B^-1 products; no factorization-update algorithm is implemented in M3.

## Deliberate limits
The solver accepts M2 standard form, not a native bounded-variable representation. Upper bounds are explicit canonical rows, so native bound flips are unnecessary and the count is zero. Sparse updates, Devex, steepest edge, Harris ratio logic, and warm starts belong to M4/M5.
