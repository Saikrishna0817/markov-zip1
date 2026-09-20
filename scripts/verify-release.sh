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
run python3 scripts/check-sovereignty.py "$ROOT"
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
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/build_info_test.cpp $objects -o "$ROOT/_verify-$name/build_info_test"
    run "$ROOT/_verify-$name/build_info_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/model_test.cpp $objects -o "$ROOT/_verify-$name/model_test"
    run "$ROOT/_verify-$name/model_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/mps_parser_test.cpp $objects -o "$ROOT/_verify-$name/mps_parser_test"
    run "$ROOT/_verify-$name/mps_parser_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/fuzz/mps_fuzz_smoke.cpp $objects -o "$ROOT/_verify-$name/mps_fuzz_smoke"
    run "$ROOT/_verify-$name/mps_fuzz_smoke"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/model_property_test.cpp $objects -o "$ROOT/_verify-$name/model_property_test"
    run "$ROOT/_verify-$name/model_property_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/dense_lu_test.cpp $objects -o "$ROOT/_verify-$name/dense_lu_test"
    run "$ROOT/_verify-$name/dense_lu_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/primal_simplex_test.cpp $objects -o "$ROOT/_verify-$name/primal_simplex_test"
    run "$ROOT/_verify-$name/primal_simplex_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/primal_simplex_property_test.cpp $objects -o "$ROOT/_verify-$name/primal_simplex_property_test"
    run "$ROOT/_verify-$name/primal_simplex_property_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/dual_simplex_test.cpp $objects -o "$ROOT/_verify-$name/dual_simplex_test"
    run "$ROOT/_verify-$name/dual_simplex_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/warm_start_property_test.cpp $objects -o "$ROOT/_verify-$name/warm_start_property_test"
    run "$ROOT/_verify-$name/warm_start_property_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/sparse_basis_test.cpp $objects -o "$ROOT/_verify-$name/sparse_basis_test"
    run "$ROOT/_verify-$name/sparse_basis_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/sparse_update_property_test.cpp $objects -o "$ROOT/_verify-$name/sparse_update_property_test"
    run "$ROOT/_verify-$name/sparse_update_property_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/regression_test.cpp $objects -o "$ROOT/_verify-$name/regression_test"
    run "$ROOT/_verify-$name/regression_test"
    run "$compiler" -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude apps/markov_cero_mps_inspect.cpp $objects -o "$ROOT/_verify-$name/markov-cero-mps-inspect"
    run "$ROOT/_verify-$name/markov-cero-mps-inspect" examples/blend.mps
  fi
done
if [[ $blocked -eq 0 ]]; then
  log "PASS: full M0 integrity/build/test checks completed"
else
  log "PASS WITH BLOCKERS: integrity and available-toolchain checks completed"
fi
