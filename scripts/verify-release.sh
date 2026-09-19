#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
REPORT="$ROOT/evidence/local-verification-report.txt"
ENVJSON="$ROOT/evidence/environment-local.json"
: > "$REPORT"
log() { printf '%s\n' "$*" | tee -a "$REPORT"; }
run() { log "+ $*"; "$@" >>"$REPORT" 2>&1; }
log "markov-cero cumulative M5 local verification"
log "UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 - "$ENVJSON" <<'PY'
import json, platform, sys
from pathlib import Path
payload = {"os": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
run python3 scripts/check-manifest.py "$ROOT"
run python3 scripts/validate_records.py "$ROOT"
run python3 scripts/no_solver_guard.py "$ROOT"
blocked=0
for pair in "gcc:g++" "clang:clang++"; do
  name=${pair%%:*}; compiler=${pair##*:}
  if ! command -v "$compiler" >/dev/null 2>&1; then
    log "BLOCKED: $compiler not available"
    blocked=1
    continue
  fi
  rm -rf "$ROOT/_verify-$name"; mkdir -p "$ROOT/_verify-$name"
  if command -v cmake >/dev/null 2>&1; then
    run cmake -S "$ROOT" -B "$ROOT/_verify-$name" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$compiler"
    run cmake --build "$ROOT/_verify-$name" --parallel 2
    if command -v timeout >/dev/null 2>&1; then run timeout 120 ctest --test-dir "$ROOT/_verify-$name" --output-on-failure; else run ctest --test-dir "$ROOT/_verify-$name" --output-on-failure; fi
  else
    log "BLOCKED: cmake/ctest not available; using direct compiler smoke fallback"
    blocked=1
    objects=""
    for source in src/foundation/build_info.cpp src/model/model.cpp src/io/mps.cpp src/verify/primal_verifier.cpp src/linalg/dense_lu.cpp src/linalg/sparse_basis.cpp src/transform/canonicalize.cpp src/lp/reference/revised_simplex.cpp src/verify/reference_lp_verifier.cpp src/lp/dual/dual_simplex.cpp; do
      object="$ROOT/_verify-$name/$(basename "${source%.cpp}").o"
      run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude -c "$source" -o "$object"
      objects="$objects $object"
    done
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/foundation_test.cpp $objects -o "$ROOT/_verify-$name/foundation_test"
    run "$ROOT/_verify-$name/foundation_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/m1_test.cpp $objects -o "$ROOT/_verify-$name/m1_test"
    run "$ROOT/_verify-$name/m1_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/m1_edge_test.cpp $objects -o "$ROOT/_verify-$name/m1_edge_test"
    run "$ROOT/_verify-$name/m1_edge_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/fuzz/mps_fuzz_smoke.cpp $objects -o "$ROOT/_verify-$name/mps_fuzz_smoke"
    run "$ROOT/_verify-$name/mps_fuzz_smoke"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/m1_property_test.cpp $objects -o "$ROOT/_verify-$name/m1_property_test"
    run "$ROOT/_verify-$name/m1_property_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/m2_test.cpp $objects -o "$ROOT/_verify-$name/m2_test"
    run "$ROOT/_verify-$name/m2_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/m3_test.cpp $objects -o "$ROOT/_verify-$name/m3_test"
    run "$ROOT/_verify-$name/m3_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/m3_property_test.cpp $objects -o "$ROOT/_verify-$name/m3_property_test"
    run "$ROOT/_verify-$name/m3_property_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/m4_test.cpp $objects -o "$ROOT/_verify-$name/m4_test"
    run "$ROOT/_verify-$name/m4_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/m4_property_test.cpp $objects -o "$ROOT/_verify-$name/m4_property_test"
    run "$ROOT/_verify-$name/m4_property_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/m5_test.cpp $objects -o "$ROOT/_verify-$name/m5_test"
    run "$ROOT/_verify-$name/m5_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/m5_property_test.cpp $objects -o "$ROOT/_verify-$name/m5_property_test"
    run "$ROOT/_verify-$name/m5_property_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/m5_audit_regression_test.cpp $objects -o "$ROOT/_verify-$name/m5_audit_regression_test"
    run "$ROOT/_verify-$name/m5_audit_regression_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude apps/markov_cero_mps_inspect.cpp $objects -o "$ROOT/_verify-$name/markov-cero-mps-inspect"
    run "$ROOT/_verify-$name/markov-cero-mps-inspect" examples/blend.mps
  fi
done
if [[ $blocked -eq 0 ]]; then
  log "PASS: full M0 integrity/build/test checks completed"
else
  log "PASS WITH BLOCKERS: integrity and available-toolchain checks completed"
fi
