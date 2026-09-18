# M5 sparse basis architecture

## Scope
M5 replaces the M4 warm dual path's per-iteration dense basis refactorization with an original deterministic sparse basis substrate. It does not implement MILP, QP, GPU, concurrent solves, or third-party solver code.

## Components
- `SparseCsc`: canonical compressed sparse column storage with strictly increasing row indices, no explicit zeros, finite values, and dimension/nonzero ceilings.
- `SparseLu`: row-map Gaussian elimination with deterministic partial pivoting, sparse L/U row and column adjacency, fill ceiling, pivot/growth diagnostics, FTRAN, BTRAN, and reach-based skipping for sparse right-hand sides.
- `SparseBasisFactorization`: a base sparse LU followed by a bounded chain of product-form eta matrices. Column replacement records `d=B^-1 a` and uses its leaving-position component as update pivot.
- M4 integration: a validated warm basis is converted to CSC once. Pivots append eta updates; update count or density triggers a full sparse refactorization. Cold fallback remains the M3 certified oracle.

## Data and execution flow
1. Validate matrix/options before allocation.
2. Convert selected canonical columns into basis CSC.
3. Factor `P B = L U` with deterministic largest-magnitude partial pivot selection.
4. FTRAN applies `P`, solves `L`, solves `U`, then applies eta inverses in chronological order.
5. BTRAN applies eta-transpose inverses in reverse order, solves `U^T`, solves `L^T`, and maps through `P^T`.
6. A replacement column is copied into canonical CSC without materializing an `n x n` dense update workspace.
7. Update-chain count and density trigger a sparse refactorization.

## Security and resource controls
Dimension, input nonzero, factor fill, update count, finite-value, pivot, and canonical-index checks fail closed. Checked API ceilings precede factor allocation. Telemetry reports refactorizations and update-chain state. Exact zeros are omitted; numerical dropping is deliberately absent in M5 to avoid silently changing equations.

## Honest limitations
The symbolic ordering is natural order plus row partial pivoting; there is no AMD/colamd ordering. Elimination uses deterministic ordered maps and can fill substantially. Reach-based triangular solves skip inactive rows but return dense result vectors because a sparse RHS can produce a dense solution. Eta vectors may be dense and therefore trigger early refactorization. These choices prioritize auditable correctness over claimed production speed.
