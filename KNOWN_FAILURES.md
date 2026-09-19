# Known Failures, Blockers, and Prototype Boundaries

## Current Prototype Status (v0.5.1 / M5.1)

1. **Solver Capabilities Implemented**:
   - Continuous LP Primal Revised Simplex (`sihopt-solve --engine primal`)
   - Continuous LP Dual Simplex with Basis Serialization and Warm Starts (`sihopt-solve --engine dual --warm-start FILE --save-basis FILE`)
   - Sparse basis representation (`SparseLu` with product-form eta updates)
   - Dual steepest-edge pricing via tableau-norm weighting
   - Harris ratio test for numerically stable pivot selection
   - Independent verification gating (`verify_reference_result` and `verify_primal`)

2. **Known Architectural Limitations**:
   - **Canonical representation is dense**: `CanonicalModel` uses dense matrices capped at 2048 dimensions and 4,194,304 elements. Sparse canonicalization is scheduled for Phase 2.
   - **Dual steepest-edge pricing is $O(\text{rows}^2)$ per pivot**: Full tableau-norm is recomputed per pivot instead of the Forrest–Goldfarb recurrence update.
   - **No Presolve / Scaling yet**: Reversible presolve reductions (Andersen & Andersen 1995) and Ruiz scaling are Phase 2 targets.
   - **No MILP (Branch-and-Bound)**: Integer variables are not branched on; MILP is scheduled for Phase 3.
   - **No GPU acceleration / PDLP**: Device-resident first-order solving is scheduled for Phase 4.
   - **No Convex QP**: Quadratic objectives/constraints are not supported yet.

3. **MPS Parser Boundaries**:
   - Strict MPS dialect: Standard NAME, ROWS, COLUMNS, RHS, RANGES, BOUNDS, ENDATA sections supported.
   - Hostile input safeguards: Bounded memory allocations, checked arithmetic dimensions, RFC 8259-compliant JSON output.

4. **Historical Clarifications**:
   - M0 contained no solver by design; M3–M5 implement certified continuous simplex.
   - All 20 test targets pass under Release and Sanitize builds.
