# M4 research interpretation record

Primary basis: Qi Huangfu and J. A. Julian Hall, “Parallelizing the dual revised simplex method,” arXiv:1503.01889; Robert Bixby, dual-simplex engineering literature; Paula Harris, two-pass ratio-test ideas; Forrest and Goldfarb, steepest-edge pricing.

Clean-room interpretation: M4 independently derives dual reoptimization from `B^-1 b`, `B^-T cB`, reduced costs, and a tableau row. Exact dense steepest-edge weights are recomputed rather than copying a production update formula. Harris mode is conservative and witness-gated. M3 remains the cold-start oracle and final mathematical comparator.

No external solver source, tests, constants, layouts, or control flow are used. Sparse factorization, eta/Forrest–Tomlin updates, hypersparsity, and parallel suboptimization remain M5 work.
