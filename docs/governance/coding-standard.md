# C++ Coding Standards & Anti-Overengineering Contract

Strict engineering standards and quality contract for markov-cero development.

---

## 1. Structural Standards

1. **One Concept Per File**: A file implements one algorithm or one data structure.
2. **Hard Line Cap**: Maximum 500 LOC per `.cpp` implementation file, 300 LOC per `.hpp` header.
   Exceeding this indicates the file combines multiple distinct concepts.
3. **1:1 Header-Source Mirroring**: `include/` and `src/` mirror each other 1:1. No header without an
   implementation file; no implementation file without a header (except `apps/`).
4. **Free Functions Over Classes**: Classes are reserved strictly for encapsulating mutable state
   with invariants (`SparseBasisFactorization`, `PresolveStack`). All other algorithms are free
   functions taking inputs by `const&` and returning a typed result struct.
5. **No Inheritance or Virtual Functions**: No class hierarchies, virtual tables, or complex templates
   in the solver core. Backend dispatch uses plain function pointers or conditional branching.
6. **No Design Patterns By Name**: No factories, visitors, observers, or dependency injection
   containers. If an algorithm requires a pattern to be understood, it must be simplified.

---

## 2. Interface Standards

7. **Standard Algorithm Signature**:
   ```cpp
   Result solve(const Input& input, const Options& options);
   ```
   Inputs are passed by `const&`. Options are plain aggregates with defaulted members. Results are
   plain aggregates carrying status, values, and telemetry. No output parameters or input mutation.
8. **Centralized Tolerances**: All numerical tolerances derive strictly from `core::NumericalPolicy`.
   Literal numerical constants in algorithmic code are prohibited.
9. **No Infinity Sentinels**: Use `std::optional<double>` or explicit flags rather than large values
   such as `1e30`.
10. **Error Handling Discipline**: Invalid programmer/user inputs throw descriptive exceptions
    naming the faulty parameter. Mathematical failures return a typed status code (`numerical_failure`)
    and must never throw.

---

## 3. Engineering Discipline

11. **Zero External Solver Dependencies**: Dependency set is strictly `{C++20 stdlib, Threads, CUDA}`.
    External packages (Eigen, Boost, fmt, spdlog, GoogleTest, nlohmann/json, libtorch) are barred.
12. **No Premature Abstraction**: Implement concrete use cases directly before extracting common
    utilities. Two similar functions are preferable to a premature generalized framework.
13. **Defensive Documentation**: Comments explain *why* an invariant exists, never *what* the code does.
14. **Deterministic Execution**: Identical input, options, and thread count must produce bit-identical
    results across runs. Parallel algorithms must support a deterministic execution mode.
15. **Whiteboard Defensibility**: Every file must be simple enough for a second-year computer science
    student to read and explain on a whiteboard in five minutes.
