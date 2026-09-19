#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BUILD="${SIHOPT_BUILD_DIR:-$ROOT/build}"
MODEL="$ROOT/examples/refinery/refinery-feasible.mps"
OUT="${TMPDIR:-/tmp}/sihopt-qualification-result.json"

echo "=== Problem ==="
echo "SIH26119 qualification slice: two-crude CDU blend with petrol, diesel, ATF, and sulfur."
echo

if [[ ! -x "$BUILD/sihopt-solve" ]]; then
  echo "=== Build ==="
  cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSIHOPT_WARNINGS_AS_ERRORS=ON
  cmake --build "$BUILD" --target sihopt-solve --parallel
  echo
fi

echo "=== Solver route ==="
"$BUILD/sihopt-info" || true
echo "Engine: certified primal revised simplex (M3) after M2 canonicalization."
echo "Independent checks: canonical witness + original-model primal."
echo

echo "=== Inputs ==="
echo "Model: $MODEL"
echo

echo "=== Solve ==="
set +e
"$BUILD/sihopt-solve" "$MODEL" --output "$OUT"
status=$?
set -e
echo
echo "=== Result file ==="
echo "$OUT"
cat "$OUT"
echo
echo "=== Limitations ==="
echo "CPU continuous LP only. GPU, MILP, QP, and production presolve are not implemented."
echo "Exit code $status (0 = verified Optimal)."
exit "$status"
