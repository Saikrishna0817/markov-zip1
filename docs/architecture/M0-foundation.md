# M0 code and architecture

Scope: build metadata, schemas, governance documents, record validation, no-solver guard, checksum verification, CMake/CTest, and deterministic packaging. The only C++ component (`COMP-M0-INFO`) exposes immutable compile-time version/milestone metadata. It allocates no dynamic solver state, performs no I/O except the CLI's single output line, has no synchronization, no CUDA, and no external dependencies.

Dependency direction is `markov-cero-info -> markov_cero_foundation`. Future solver responsibilities are documented but absent. Tests: `foundation_contract`, `json_records`, `no_solver_guard`. Resource limits are enforced by the release script using `timeout` where available. Rollback is local Git checkout; release archives exclude `.git`.
