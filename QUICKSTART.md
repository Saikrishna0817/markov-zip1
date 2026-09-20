# Quickstart Guide — markov-cero v0.5.2

5 commands from zero to a verified optimal solve on the MRPL refinery qualification model.

---

## 5 Commands to a Verified Solve

```sh
# 1. Clone the repository
git clone https://github.com/Saikrishna0817/markov-zip1.git markov-cero && cd markov-cero

# 2. Configure the build (C++20, Release mode)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 3. Build solver core library and CLI binaries
cmake --build build -j

# 4. Solve the SIH26119 refinery qualification model
./build/markov-cero-solve examples/refinery/refinery-feasible.mps

# 5. Verify the entire test suite (31/31 CTest targets)
ctest --test-dir build --output-on-failure
```

---

## What You See (Command 4 Output)

When Command 4 runs, `markov-cero-solve` outputs JSON telemetry to stdout and a verdict to stderr:

```json
{
  "version": "0.5.2",
  "engine": "primal",
  "status": "Optimal",
  "verified": true,
  "message": "reference primal revised simplex optimum",
  "objective": 2600.0,
  "primal": [38.095238095238095, 21.428571428571431],
  "canonical_verified": true,
  "original_verified": true,
  "original_message": "original primal verified",
  "maximum_primal_violation": 0.0,
  "maximum_variable_violation": 0.0,
  "runtime_ms": 0.12,
  "nodes_explored": 1,
  "lp_iterations": 8,
  "best_bound": 2600.0,
  "relative_gap": 0.0,
  "limitations": "CPU sovereign LP and MILP Branch-and-Cut engine; GPU and QP are not implemented."
}
```

Stderr verdict:
```text
markov-cero 0.5.2 Optimal VERIFIED
```

- **Stdout**: RFC 8259-compliant JSON record with dual-gated mathematical verification.
- **Stderr**: Single-line human verdict: `markov-cero 0.5.2 Optimal VERIFIED`.
- **Exit Code**: `0` (`0` = Verified Optimal; typed non-zero codes for infeasible, unbounded, etc.).

---

## Automated Qualification Demo

To execute the self-contained SIH26119 qualification scenario in one command:

```sh
bash run-qualification-demo.sh
```

This runs the two-crude CDU blend with petrol, diesel, ATF, and sulfur specifications, validates
the independent canonical witness certificate, and asserts original-model primal feasibility.

---

## Common Options

```sh
# Solve a MILP problem with sovereign branch-and-cut
./build/markov-cero-solve data/miplib/stein9.mps --engine milp

# Solve with multithreaded parallel tree search (4 threads)
./build/markov-cero-solve data/miplib/stein15.mps --engine parallel --threads 4

# Solve using matrix-free first-order PDLP
./build/markov-cero-solve data/netlib/afiro.mps --engine pdlp

# Inspect model dimensions, matrix density, and bounds
./build/markov-cero-mps-inspect examples/refinery/refinery-feasible.mps

# Run the full release verification suite (compilation, sanitizers, sovereignty)
bash scripts/verify-release.sh
```

---

## System Requirements

- **Compiler**: GCC $\ge 11$ or Clang $\ge 14$ (C++20 standard library required).
- **Build System**: CMake $\ge 3.20$.
- **Python**: Python $\ge 3.8$ (for benchmark runners and sovereignty verification).
- **External Solvers**: **None**. Zero external linear algebra or solver dependencies.
