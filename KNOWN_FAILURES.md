# Known Failures, Blockers, and Prototype Boundaries

## Current Prototype Status (v0.5.1 / Phase 3 MILP Sovereign Release)

1. **Solver Capabilities Implemented**:
   - Sovereign Mixed-Integer Linear Programming (MILP) Branch-and-Cut Engine (`markov-cero-solve --engine milp` or `--engine auto`)
   - Branch-and-Bound search tree manager with best-bound, depth-first, and best-bound plunge node selection (`BranchNode`, `NodeCompareBestBound`)
   - Reliability pseudo-cost variable branching selection with most-fractional fallback (`VariablePseudoCost`, `select_branching_variable`)
   - Dual-tier Primal Heuristics: Fast Simple Rounding and iterative Feasibility Pump (`simple_rounding`, `feasibility_pump`)
   - Simplex tableau Gomory Mixed-Integer (GMI) cutting plane generator with mathematical validity guard (`generate_gomory_cuts`, `add_cuts_to_model`)
   - Node LP dual warm-starting with parent basis states (`lp::dual::solve`)
   - Continuous LP Primal Revised Simplex (`markov-cero-solve --engine primal`)
   - Continuous LP Dual Simplex with Basis Serialization and Warm Starts (`markov-cero-solve --engine dual --warm-start FILE --save-basis FILE`)
   - Sparse Canonical Model representation (`SparseCanonicalModel` CSC format) removing dense memory bottlenecks
   - Reversible multi-pass Presolve (`markov_cero::presolve::presolve`) with typed LIFO postsolve restoration (empty rows, empty columns, fixed variables, row singletons)
   - Ruiz condition equilibration scaling (`markov_cero::scale::equilibrate`) with $[10^{-4}, 10^4]$ clamp safeguarding and exact unscaling
   - Sparse basis representation (`SparseLu` with product-form eta updates)
   - Dual steepest-edge pricing via tableau-norm weighting
   - Harris ratio test for numerically stable pivot selection
   - Zero-trust dual-gated verification: canonical primal/dual feasibility witness check and independent original-model verification (`verify_primal`) with row, column, and integrality tolerances
   - Comprehensive Netlib benchmark harness (`scripts/run_netlib.py`) passing canonical Netlib instances to machine precision
   - Comprehensive MIPLIB benchmark harness (`scripts/run_miplib.py`) passing canonical MIPLIB instances (`stein9`, `stein15`, `flugpl`) to certified optimality

2. **Known Architectural Limitations**:
   - **Dual steepest-edge pricing is $O(\text{rows}^2)$ per pivot**: Full tableau-norm is recomputed per pivot instead of the Forrest–Goldfarb recurrence update.
   - **No GPU acceleration / PDLP**: Device-resident first-order solving is scheduled for Phase 4.
   - **No Convex QP**: Quadratic objectives/constraints are not supported yet.

3. **MPS Parser Boundaries**:
   - Strict MPS dialect: Standard NAME, ROWS, COLUMNS, RHS, RANGES, BOUNDS (including binary/integer markers `INTORG`/`INTEND`, `BV`, `UI`, `LI`), ENDATA sections supported.
   - Hostile input safeguards: Bounded memory allocations, checked arithmetic dimensions, RFC 8259-compliant JSON output.

4. **Historical Clarifications**:
   - M0 contained no solver by design; M3–M5 implement certified continuous simplex; Phase 3 implements sovereign MILP Branch-and-Cut.
   - All 28 test targets pass under Release and Sanitize builds (100% CTest pass rate).
