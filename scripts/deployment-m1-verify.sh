#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
OUT="${1:-$ROOT/evidence/deployment-m1-verification.txt}"
exec > >(tee "$OUT") 2>&1
cd "$ROOT"
echo "UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 --version
cmake --version | head -1
for compiler in g++ clang++; do "$compiler" --version | head -1; done
for compiler in g++ clang++; do
  name=${compiler//+/p}
  for mode in Debug Release; do
    build="_deployment-${name}-${mode}"
    rm -rf "$build"
    cmake -S . -B "$build" -DCMAKE_BUILD_TYPE="$mode" -DCMAKE_CXX_COMPILER="$compiler"
    cmake --build "$build" --parallel
    ctest --test-dir "$build" --output-on-failure
  done
done
LD_LIBRARY_PATH="$(dirname "$(g++ -print-file-name=libasan.so.6.0.0)"):${LD_LIBRARY_PATH:-}" ./scripts/m1-safety-smoke.sh
./scripts/m1-coverage-fuzz.sh 60
echo "DEPLOYMENT M1 VERIFICATION PASSED"
