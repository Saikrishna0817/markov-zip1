# Known failures and blockers

1. Mathematical reviewer and security/release reviewer are unassigned; final M0 acceptance is blocked.
2. Exact section/equation mapping is blocked for sources without a pinned lawful full-text archive.
3. QP Hessian storage, symmetry/PSD, near-indefinite, and regularization policies are provisional.
4. NumericalPolicy defaults are proposed, not independently reviewed.
5. Judging date, dashboard, commercial-oracle access, final benchmark limits, and first refinery use case are unresolved.
6. The deployment machine lacks Ninja and the CUDA toolkit; neither is required for M0.
7. Sandbox verification records the sandbox toolchain, not the separately reported Omarchy deployment machine.
8. No solver exists in M0; all LP/MILP/QP/GPU capability claims are intentionally absent.

## M1 environment blockers

- The execution sandbox lacks CMake, Clang, libFuzzer, and dynamic ASan/UBSan runtimes.
- Deployment-machine dual-compiler, CMake/CTest, sanitizer, and coverage-guided fuzz evidence remains required.
- Fixed-column MPS and vendor extensions are intentionally unsupported in the M1 strict subset.
