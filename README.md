# markov-cero

[![CI][ci-badge]][ci-link]

[ci-badge]: https://github.com/Saikrishna0817/markov-zip1/actions/workflows/ci.yml/badge.svg
[ci-link]: https://github.com/Saikrishna0817/markov-zip1/actions/workflows/ci.yml

Clean-room C++20 solver core for SIH 2026 problem SIH26119 (MRPL indigenous LP/MILP/QP).

Current release: **v0.5.2** (Phase 5 GPU-Accelerated Multi-Engine Release). Continuous LP,
First-Order PDLP (CPU & CUDA GPU), and Mixed-Integer Linear Programming (MILP) Branch-and-Cut
are fully implemented and independently verified. Convex QP is scheduled for Phase 6.

## Scope & Capabilities

- **Model & Storage**: Immutable model, free-format MPS parser, Compressed Sparse Column (CSC).
- **Presolve & Scaling**: Reversible multi-pass presolve with LIFO postsolve stack; Ruiz scaling.
- **Continuous LP Engines**:
  - Certified Primal Revised Simplex with Bland's rule anti-cycling.
  - Dual Revised Simplex with Harris two-pass ratio test, basis serialization, and warm-starts.
  - Sparse basis substrate with row-map Gaussian LU and product-form Eta updates.
  - Matrix-free First-Order PDLP (Chambolle-Pock) with diagonal preconditioning.
- **GPU-Accelerated LP Engine**:
  - Sovereign CUDA First-Order PDLP with device-resident loop (`--engine pdlp --backend gpu`).
  - Warp-per-row CSR SpMV and transpose-SpMV kernels with shuffle reduction.
  - Deterministic two-stage parallel reductions and adaptive restart strategy.
  - Four-part timing telemetry (H2D, kernel, D2H, total) and verified scale crossover study.
- **MILP Branch-and-Cut Engines**:
  - Sovereign Sequential Branch-and-Cut (`--engine milp`).
  - Multithreaded Parallel Tree Search (`--engine parallel --threads N`) with C++20 `std::jthread`.
  - Cutting Planes: Gomory Mixed-Integer (GMI) cuts and Mixed-Integer Rounding (MIR) cuts.
  - Variable Selection: Strong Branching domain reduction and Reliability Pseudo-Costs.
  - Dual-tier Primal Heuristics: Simple Rounding and Feasibility Pump with cycle perturbation.
- **Zero-Trust Independent Verification**: Dual-gated verification in canonical and original space.
- **Applications**: `markov-cero-info`, `markov-cero-mps-inspect`, and `markov-cero-solve`.

Planned for future milestones: convex Quadratic Programming (QP) and ML-assisted branching.

## Solve a model

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
./build/markov-cero-solve examples/blend.mps
./build/markov-cero-solve examples/refinery/refinery-feasible.mps --output /tmp/result.json
```

Judge demo (offline):

```sh
bash run-qualification-demo.sh
```

## Verify

```sh
./scripts/verify-release.sh
```

See `STATUS.md` and `QUICKSTART.md`.
