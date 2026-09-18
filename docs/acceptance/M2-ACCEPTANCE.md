# M2 acceptance

Local GCC warnings-as-errors, analytic FTRAN/BTRAN, 100 randomized diagonally dominant systems, singular rejection, residual checks, canonicalization, and postsolve tests are required. Deployment CMake/Clang remains deferred to M5.

Independent-review remediation added checked dense-size arithmetic, explicit dense-oracle limits, finite operand/intermediate/result checks, multi-stage-pivot FTRAN regression, pivoted BTRAN regression, near-singular boundaries, empty dimensions, ranged and one-sided transformation cases, malformed record validation, and explicit rejection of integer/binary canonicalization. Cumulative M2 ASan/UBSan passed with auditable evidence.
Final record-validation remediation rejects structural-variable counts larger than canonical columns and requires objective sign to be exactly +1 or -1; both malformed cases have regression tests.
