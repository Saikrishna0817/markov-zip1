# M1 model, parser, and verifier

M1 adds an immutable validated model, canonical CSC sparse matrix, strict free-format MPS subset, independent primal verifier, inspection CLI, and thin Python subprocess interface. It adds no optimization algorithm.

The model owns all arrays. Validation rejects non-finite coefficients, invalid CSC pointers/indices, duplicate names, inconsistent bounds, and dimension mismatches. Sparse builder aggregates duplicates and removes exact zero sums. The verifier recomputes activities and objective independently with long-double accumulation.

The parser supports NAME, OBJSENSE, OBJNAME, ROWS (N/E/L/G), COLUMNS, one RHS, one RANGES, one BOUNDS, ENDATA, integer markers, and LO/UP/FX/FR/MI/PL/BV/LI/UI bounds. Unknown sections, multiple rim vectors, malformed records, non-finite numbers, and resource-limit violations fail with line-numbered diagnostics.
