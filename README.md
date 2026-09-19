# markov-cero

Clean-room C++20 solver core for SIH 2026 problem SIH26119 (MRPL indigenous LP/MILP/QP).

Current release: **v0.5.2** (Phase 4 Sovereign Multi-Engine Release). Continuous LP, First-Order PDLP, and Mixed-Integer Linear Programming (MILP) Branch-and-Cut are fully implemented and independently verified. CUDA/GPU acceleration and convex QP remain planned.

## Scope & Capabilities

- **Model & Storage**: Immutable model, free-format MPS parser, Compressed Sparse Column (CSC) storage.
- **Presolve & Scaling**: Reversible multi-pass presolve with LIFO postsolve stack; Ruiz $\ell_\infty$ condition equilibration.
- **Continuous LP Engines**:
  - Certified Primal Revised Simplex with Bland's rule anti-cycling.
  - Dual Revised Simplex with Harris two-pass ratio test, basis serialization, and warm-starts.
  - Sparse basis substrate with row-map Gaussian LU and product-form Eta updates.
  - Matrix-free First-Order PDLP (Primal-Dual Hybrid Gradient / Chambolle-Pock) with diagonal preconditioning.
- **MILP Branch-and-Cut Engines**:
  - Sovereign Sequential Branch-and-Cut (`--engine milp`).
  - Multithreaded Parallel Tree Search (`--engine parallel --threads N`) with C++20 `std::jthread` and lock-free incumbent management.
  - Cutting Planes: Gomory Mixed-Integer (GMI) cuts with algebraic slack substitution and Mixed-Integer Rounding (MIR) cuts.
  - Variable Selection: Strong Branching domain reduction, Reliability Pseudo-Costs, and Most-Fractional branching rules.
  - Dual-tier Primal Heuristics: Simple Rounding and Feasibility Pump with cycle-detection perturbation.
- **Zero-Trust Independent Verification**: Dual-gated verification validating solutions in both canonical form and original model space.
- **Applications**: `markov-cero-info`, `markov-cero-mps-inspect`, and `markov-cero-solve`.

Planned for future milestones: CUDA/GPU acceleration and convex Quadratic Programming (QP).

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

See `KNOWN_FAILURES.md`, `CAPABILITY-MATRIX.md`, and `QUICKSTART.md`.
