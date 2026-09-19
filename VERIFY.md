# Release verification

Run `./scripts/verify-release.sh` from the extracted `markov-cero` directory. A passing report establishes only that M0 files are intact and the governance/build/schema checks pass in the local environment. It does not establish solver correctness or performance.

The checked source manifest excludes its own digest and generated local reports. Any source change requires regenerating `provenance/source-manifest.sha256`, committing, and repackaging.
