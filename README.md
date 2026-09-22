# markov-cero

[![CI][ci-badge]][ci-link]

[ci-badge]: https://github.com/Saikrishna0817/markov-zip1/actions/workflows/ci.yml/badge.svg
[ci-link]: https://github.com/Saikrishna0817/markov-zip1/actions/workflows/ci.yml

Clean-room C++20 LP / MILP / QP solver core for SIH 2026 problem SIH26119 (MRPL).

Current release: **v0.5.2**. Code for continuous LP (primal/dual revised simplex, CPU PDLP),
MILP branch-and-cut, convex QP/MIQP, and CPU PDLP are present. GPU/CUDA path is deferred. Status of each piece is
tracked in [STATUS.md](STATUS.md) using an explicit Prototype / Verified / Hardware-verified
vocabulary. Do not treat the presence of code as a performance or scale claim.

Substantial implementation was AI-assisted under human direction; see
[docs/governance/provenance-and-verification.md](docs/governance/provenance-and-verification.md).

## What works today (honest summary)

- Free-format MPS parser (including integer markers and quadratic sections).
- Primal revised simplex — exercised on the 7 Netlib instances in the CI suite.
- MILP branch-and-cut — exercised on the 3 MIPLIB instances in the CI suite.
- CPU PDLP, dual simplex, presolve, Ruiz scaling, cuts, heuristics, QP ADMM: code + unit tests
  (Prototype).
- GPU/CUDA path: **deferred** (not in the active build). Prior unpublished speedup claims must not be cited.

Full capability table and limitations: **[STATUS.md](STATUS.md)**.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Solve

```bash
./build/markov-cero-solve examples/blend.mps --engine primal
./build/markov-cero-solve data/miplib/stein9.mps --engine milp
```

See `STATUS.md` for CLI options and exit codes.
