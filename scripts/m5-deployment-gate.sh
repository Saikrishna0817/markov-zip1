#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
command -v cmake >/dev/null
command -v ctest >/dev/null
command -v g++ >/dev/null
command -v clang++ >/dev/null
for compiler in g++ clang++; do
  tag=${compiler//+/x}
  build="$ROOT/_m5-deploy-$tag"
  rm -rf "$build"
  cmake -S "$ROOT" -B "$build" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER="$compiler" -DMARKOV_CERO_WARNINGS_AS_ERRORS=ON
  cmake --build "$build" --parallel 2
  ctest --test-dir "$build" --output-on-failure
  sanitize="$ROOT/_m5-deploy-$tag-sanitize"
  rm -rf "$sanitize"
  cmake -S "$ROOT" -B "$sanitize" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER="$compiler" -DMARKOV_CERO_ENABLE_ASAN_UBSAN=ON
  cmake --build "$sanitize" --parallel 2
  ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir "$sanitize" --output-on-failure
done
fuzz="$ROOT/_m5-deploy-fuzz"
rm -rf "$fuzz"
cmake -S "$ROOT" -B "$fuzz" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=clang++ -DMARKOV_CERO_BUILD_FUZZER=ON
cmake --build "$fuzz" --parallel 2
"$fuzz/sparse_basis_fuzz" -max_total_time=120 -timeout=5 "$ROOT/tests/fuzz/corpus/sparse_basis"
"$fuzz/mps_fuzz" -max_total_time=120 -timeout=5 "$ROOT/tests/fuzz/corpus/mps"
echo "M5 deployment gate passed"
