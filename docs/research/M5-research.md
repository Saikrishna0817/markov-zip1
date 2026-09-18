# M5 research papers and standards

## Sources and clean-room interpretation
M5 uses public mathematical descriptions, not competitor source code. The design is independently written from factorization identities, simplex basis-update literature, and secure C++ engineering standards.

### Wilkinson and Higham: Gaussian elimination
Partial pivoting chooses a large available pivot to control division by tiny values. Growth and minimum/maximum pivot diagnostics expose numerical warning signs. Literature does not guarantee safety for all matrices, so the implementation also has pivot/fill ceilings and failure statuses.

### Bartels-Golub and product-form updates
A simplex pivot changes one basis column. The replacement can be represented by an elementary eta matrix built from `B^-1 a`, avoiding a full factorization after every pivot. Update chains must be bounded because error and work accumulate.

### Forrest-Tomlin context
Production simplex systems use more sophisticated sparse update forms and refactorization policies. M5 implements the simpler auditable product form; it does not claim Forrest-Tomlin performance. The interface and telemetry create a future comparison boundary.

### Gilbert-Peierls and sparse triangular reach
Sparse triangular solves can restrict work to graph nodes reachable from a sparse RHS. M5 stores both row and column adjacency and propagates active nodes in dependency order. The dense return vector is an API choice, not evidence of dense factorization.

### Huangfu-Hall dual revised simplex
Their dependency analysis motivates separating BTRAN, pricing, ratio testing, FTRAN, and basis update. M5 replaces only the basis-linear-algebra component in the existing M4 dual state machine; result certificates remain checked by the independent verifier.

### IEEE 754 and secure-resource guidance
Every public numeric input must be finite. NaN cannot be treated as zero or ordered normally. Dimensions, nonzeros, fill, and update chains are bounded before unsafe amplification. Integer sizes and vector indexing are validated before use.

## Research-to-test traceability
| Claim | Executable evidence |
| --- | --- |
| `P B=L U` solve order | dense-oracle FTRAN/BTRAN parity |
| transpose permutation/order | random BTRAN and row-pivot cases |
| eta product form | fresh-refactor parity after updates |
| bounded update error/work | count and density trigger tests |
| canonical CSC | malformed storage rejection |
| finite arithmetic boundary | NaN/infinity rejection |
| cumulative compatibility | M0-M4 regression suite |

## Deliberate deviations and limitations
No Markowitz/AMD ordering, supernodes, numerical dropping, iterative refinement, Forrest-Tomlin update, GPU kernels, or parallel factorization is claimed. M5 is a sparse, deterministic, reviewable correctness substrate. More advanced performance mechanisms require later milestone approval and parity against this implementation.
