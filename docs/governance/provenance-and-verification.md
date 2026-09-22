# Provenance and verification

## Provenance

- Baseline: empty original repository created for markov-cero M0 on 2026-09-13.
- Implementation: independent clean-room session. External solver source inspected
  during implementation: none.
- Inputs admitted: approved project requirements, mathematical statements written
  independently, public bibliographic metadata, and sanitized behavior-level
  competitor observations.
- Excluded: competitor code, pseudocode, tests, identifiers, constants, layouts,
  and control-flow recipes.
- Production external-solver path: prohibited and absent.

**AI-assisted implementation:** Substantial portions of the codebase were produced
with AI code-generation assistance under human direction and review. The authors
retain responsibility for design decisions, mathematical correctness checks,
and verification of every claim that appears in documentation or benchmarks.
This disclosure is intentional; concealment would be more damaging than use.

Each later change should record sources consulted, source exposure, derivation
references, affected invariants, and reviewer status. Contamination triggers
quarantine and clean reimplementation.

## Release verification

Run `./scripts/verify-release.sh` from the project root. A passing report
establishes only that required files are intact and the governance/build/schema
checks pass in the local environment. It does **not** establish solver
correctness or performance.

Any source change that affects the packaged release requires regenerating the
source manifest, committing, and repackaging.
