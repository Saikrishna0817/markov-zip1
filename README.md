# SIHOpt

Clean-room C++20 solver core for SIH 2026 problem SIH26119 (MRPL indigenous LP/MILP/QP).

Current release: **v0.5.1**, milestone **M5** (release candidate). Continuous LP is implemented and independently verified. GPU, MILP, QP, and presolve remain planned and must not be claimed.

## Scope

- Immutable model, strict free-format MPS subset, CSC storage
- Reversible canonicalization to standard form
- Certified primal revised simplex (M3) and dual warm path (M4) on a sparse basis substrate (M5)
- Independent canonical witness checks and original-model primal verification
- `sihopt-info`, `sihopt-mps-inspect`, and `sihopt-solve`

Not in this prototype: CUDA/GPU acceleration, PDHG/PDLP, MILP, convex QP, production presolve.

## Solve a model

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
./build/sihopt-solve examples/blend.mps
./build/sihopt-solve examples/refinery/refinery-feasible.mps --output /tmp/result.json
```

Judge demo (offline):

```sh
bash run-qualification-demo.sh
```

## Verify

```sh
./scripts/verify-release.sh
```

See `KNOWN_FAILURES.md`, `CAPABILITY-MATRIX.md`, and `QUICKSTART.md`.
