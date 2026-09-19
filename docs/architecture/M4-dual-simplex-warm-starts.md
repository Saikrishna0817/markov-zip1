# M4 dual revised simplex and warm starts

M4 adds a dense dual revised-simplex reoptimization engine above the M2 canonical model and M3 verifier. Cold requests delegate to the certified M3 oracle. A validated dual-feasible basis can then be reused when the right-hand side changes, which is the primary hot-start case needed later by MILP node relaxations.

The public contract includes a matrix/objective fingerprint, strict basis serialization with checksum, Bland or tableau-norm leaving-row pricing (weight `||A^T B^{-T} e_i||^2`, not conventional dual steepest-edge), conservative two-pass Harris ratio selection, iteration telemetry, refactorization counts, and explicit cold-fallback disclosure.

Every iteration refactorizes the dense basis with M2 LU. This is intentionally expensive but removes hidden update state before M5. Every certified result is checked through the independent M3 verifier. Invalid, stale, duplicate, singular, or dimension-mismatched basis states fail closed or visibly use the configured cold fallback.

The warm path treats a 0×0 basis as valid and bypasses non-applicable pivot-condition diagnostics. Model/options validation is separated from basis-validation fallback so invalid numerical options are never mislabeled as warm-start failures. Telemetry reports the complete objective including its constant offset. Serialization and parsing enforce dimensions, fingerprint syntax, unique indices, and index ranges; model-specific fingerprint and singularity checks occur in `solve`.
