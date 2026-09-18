# M0-M5 audit remediation — v0.5.1 RC2

Fixed the reproduced false Optimal and false Infeasible certifications with scale-aware reduced-cost and certificate checks; rejected tolerance overflow; made sparse updates transactional; enforced update-chain limits internally; corrected the exact MPS byte boundary; fixed the healthy information-command exit code; aligned version metadata; and added permanent audit regressions.

The cumulative GCC warnings-as-errors and functional/property suites pass. This runtime still lacks CMake, CTest, Clang, libFuzzer, and a usable ASan runtime. Independent mathematical and source/security review also remains pending. Therefore this is a stable candidate, not an unqualified stable release. M6 is not authorized.
