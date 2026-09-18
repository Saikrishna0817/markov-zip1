# SIHOpt M1 Model, Parser, and Verifier

Clean-room foundation for SIH 2026 problem SIH26119.

## Scope

This `v0.1.0` cumulatively contains M0 governance plus an immutable model, strict MPS subset, validated CSC storage, independent primal verifier, inspection CLI, Python-facing inspection path, tests, and a fuzz target. It still contains no optimization algorithm.

## Verify

```sh
./scripts/verify-release.sh
```

The script verifies the internal manifest, detects the environment, configures and builds with available GCC and Clang compilers, runs CTest, validates JSON records, and writes `evidence/local-verification-report.txt`.

## Deadline tracks

- Submission prototype: gated M0→M2, narrow M3 only after prerequisites, then a verified crude-blending LP demo if approved.
- Full production: M0→M14 without weakened gates after the 2026-09-23 submission deadline.

See `KNOWN_FAILURES.md` and `docs/acceptance/M0-ACCEPTANCE.md` before relying on this archive.
# markov-zip1
