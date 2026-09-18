# M1 acceptance

Implemented: model, CSC validation/builder, strict MPS subset, line diagnostics, primal/objective/integrality verifier, CLI, Python inspection path, tests, and fuzz target. No solver algorithm.

Recommendation before external review: CONTINUE WITHIN SAME MILESTONE. Final acceptance requires deployment-machine GCC/Clang/CMake, sanitizer/fuzz runs, exact research mapping, and named review.

Continuation evidence: fixed content-after-ENDATA acceptance, rejected non-ASCII names and objective-row rim ambiguity, added checked byte accounting, 250 randomized CSC/dense property trials, GCC -fanalyzer, and fresh-release execution. All passed. Clang/CMake/libFuzzer and sanitizer-runtime execution remain environment-blocked.
