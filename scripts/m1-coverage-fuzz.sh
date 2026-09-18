#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
BUILD="${TMPDIR:-/tmp}/sihopt-m1-coverage-fuzz"
SECONDS_TO_RUN="${1:-60}"
rm -rf "$BUILD"; mkdir -p "$BUILD"
CXX=${CXX:-g++}
COMMON=(-std=c++20 -O1 -g -Wall -Wextra -Wpedantic -Werror -I"$ROOT/include")
"$CXX" "${COMMON[@]}" -fsanitize-coverage=trace-pc -c "$ROOT/src/model/model.cpp" -o "$BUILD/model.o"
"$CXX" "${COMMON[@]}" -fsanitize-coverage=trace-pc -c "$ROOT/src/io/mps.cpp" -o "$BUILD/mps.o"
"$CXX" "${COMMON[@]}" "$ROOT/tests/fuzz/mps_coverage_fuzz.cpp" "$BUILD/model.o" "$BUILD/mps.o" -o "$BUILD/mps_coverage_fuzz"
"$BUILD/mps_coverage_fuzz" "$SECONDS_TO_RUN"
