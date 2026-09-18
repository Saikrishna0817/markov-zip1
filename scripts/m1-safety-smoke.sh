#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
BUILD="$ROOT/_verify-m1-safety"
rm -rf "$BUILD"; mkdir -p "$BUILD"
CXX=${CXX:-g++}
FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Werror -I"$ROOT/include" -fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g)
objects=()
for source in src/foundation/build_info.cpp src/model/model.cpp src/io/mps.cpp src/verify/primal_verifier.cpp src/linalg/dense_lu.cpp src/linalg/sparse_basis.cpp src/transform/canonicalize.cpp src/lp/reference/revised_simplex.cpp src/verify/reference_lp_verifier.cpp src/lp/dual/dual_simplex.cpp; do
  object="$BUILD/$(basename "${source%.cpp}").o"
  "$CXX" "${FLAGS[@]}" -c "$ROOT/$source" -o "$object"
  objects+=("$object")
done
for test in tests/m1_test.cpp tests/m1_edge_test.cpp tests/fuzz/mps_fuzz_smoke.cpp tests/m2_test.cpp tests/m3_test.cpp tests/m3_property_test.cpp tests/m4_test.cpp tests/m4_property_test.cpp tests/m5_test.cpp tests/m5_property_test.cpp; do
  exe="$BUILD/$(basename "${test%.cpp}")"
  "$CXX" "${FLAGS[@]}" "$ROOT/$test" "${objects[@]}" -o "$exe"
  ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 "$exe"
done
echo "Cumulative M5 ASan/UBSan and deterministic fuzz smoke passed"
