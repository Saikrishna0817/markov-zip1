# markov-cero

[![CI][ci-badge]][ci-link]

[ci-badge]: https://github.com/Saikrishna0817/markov-zip1/actions/workflows/ci.yml/badge.svg
[ci-link]: https://github.com/Saikrishna0817/markov-zip1/actions/workflows/ci.yml

Clean-room C++20 LP / MILP / QP solver core for SIH 2026 problem SIH26119 (MRPL).

Current release: **v0.5.2**. Code for continuous LP (primal/dual revised simplex, CPU PDLP),
MILP branch-and-cut, convex QP/MIQP, and a CUDA PDLP path is present. Status of each piece is
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
- GPU PDLP kernels and host wrappers: code + unit tests under CPU fallback. No CUDA hardware
  timing is published until re-run on real hardware; prior figures are quarantined under
  `evidence/benchmarks/_unverified/`.

Full capability table and limitations: **[STATUS.md](STATUS.md)**.

## Build and solve

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/markov-cero-solve examples/blend.mps
./build/markov-cero-solve examples/qp_portfolio.mps --engine qp
./build/markov-cero-solve examples/refinery/refinery-feasible.mps
```

Offline qualification script:

```sh
bash run-qualification-demo.sh
```

Verify the test suite:

```sh
ctest --test-dir build --output-on-failure
```

See [BUILDING.md](BUILDING.md) and [STATUS.md](STATUS.md).

## Documentation

| File | Purpose |
|------|---------|
| [STATUS.md](STATUS.md) | Capability matrix with tier labels and evidence |
| [docs/architecture.md](docs/architecture.md) | System design |
| [docs/mathematics.md](docs/mathematics.md) | Algorithm notes |
| [docs/gpu.md](docs/gpu.md) | GPU design notes (numbers pending real hardware) |
| [docs/governance/provenance-and-verification.md](docs/governance/provenance-and-verification.md) | Provenance, AI disclosure, release checks |
| [docs/decisions/](docs/decisions/) | ADRs |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
