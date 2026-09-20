#!/usr/bin/env python3
"""
markov-cero Netlib Benchmark Runner & Test Harness
Evaluates markov-cero solver against standard Netlib Linear Programming problems.
Verifies optimality, primal feasibility, dual feasibility, and objective accuracy.
"""

import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
import time
import urllib.request
from typing import Dict, Any, List, Optional

# Canonical Netlib reference optimal values (from netlib.org/lp/data/readme)
NETLIB_BENCHMARKS = {
    "afiro": {
        "rows": 28,
        "cols": 32,
        "optimal": -464.753142857143,
        "description": "Michael Saunders, Stanford Systems Optimization Lab",
    },
    "adlittle": {
        "rows": 57,
        "cols": 97,
        "optimal": 225494.96316238,
        "description": "Arthur D. Little refinery model",
    },
    "sc50a": {
        "rows": 51,
        "cols": 48,
        "optimal": -64.575077058564,
        "description": "Staircase structure model A (50 stages)",
    },
    "sc50b": {
        "rows": 51,
        "cols": 48,
        "optimal": -70.000000000000,
        "description": "Staircase structure model B (50 stages)",
    },
    "sc105": {
        "rows": 106,
        "cols": 103,
        "optimal": -52.202061211707,
        "description": "Staircase structure model (105 stages)",
    },
    "sc205": {
        "rows": 206,
        "cols": 203,
        "optimal": -52.202061211707,
        "description": "Staircase structure model (205 stages)",
    },
    "share2b": {
        "rows": 97,
        "cols": 79,
        "optimal": -415.732240741418,
        "description": "Share model 2B",
    },
    "share1b": {
        "rows": 118,
        "cols": 225,
        "optimal": -76589.3185791855,
        "description": "Share model 1B",
    },
    "recipe": {
        "rows": 92,
        "cols": 180,
        "optimal": -266.616000000000,
        "description": "Food recipe formulation model",
    },
    "scagr7": {
        "rows": 130,
        "cols": 140,
        "optimal": -2331389.82433098,
        "description": "Agricultural multi-period model (7 periods)",
    },
    "beaconfd": {
        "rows": 174,
        "cols": 262,
        "optimal": 33592.4858072000,
        "description": "Beacon food distribution problem",
    },
    "scorpion": {
        "rows": 389,
        "cols": 358,
        "optimal": 1878.1248227381,
        "description": "Scorpion energy/flow model",
    },
}

BASE_URL = "https://raw.githubusercontent.com/coin-or-tools/Data-Netlib/master"

OFFLINE_INSTANCES = [
    "afiro", "adlittle", "sc50a", "sc50b", "sc105", "share2b", "recipe"
]
EXTENDED_INSTANCES = [
    "sc205", "share1b", "scagr7", "beaconfd", "scorpion"
]


def find_solver_binary() -> Optional[str]:
    candidates = [
        "_deployment-phase2-build/markov-cero-solve",
        "build/markov-cero-solve",
        "_build/markov-cero-solve",
        "markov-cero-solve",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def ensure_instance(name: str, data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    target = os.path.join(data_dir, f"{name}.mps")
    if os.path.isfile(target):
        return target

    url = f"{BASE_URL}/{name}.mps.gz"
    print(f"[*] Downloading {name} from {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "markov-cero-Netlib-Harness"})
        with urllib.request.urlopen(req) as resp:
            gz_data = resp.read()
        decompressed = gzip.decompress(gz_data)
        with open(target, "wb") as f:
            f.write(decompressed)
        print(f"[+] Cached {target} ({len(decompressed)} bytes)")
        return target
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {name}.mps.gz: {e}")


def run_instance(
    solver_bin: str,
    mps_path: str,
    engine: str = "primal",
    timeout_sec: int = 120,
) -> Dict[str, Any]:
    cmd = [solver_bin, mps_path, "--engine", engine]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        wall_ms = (time.perf_counter() - t0) * 1000.0
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "status": "Timeout",
            "verified": False,
            "error": f"Execution timed out after {timeout_sec}s",
            "wall_ms": timeout_sec * 1000.0,
        }

    parsed_json = None
    for line in proc.stdout.splitlines():
        line_str = line.strip()
        if line_str.startswith("{") and line_str.endswith("}"):
            try:
                parsed_json = json.loads(line_str)
                break
            except Exception:
                pass

    if parsed_json is None:
        return {
            "exit_code": proc.returncode,
            "status": "ParseError",
            "verified": False,
            "error": f"No valid JSON output. Stderr: {proc.stderr[:200]}",
            "stdout": proc.stdout[:200],
            "wall_ms": wall_ms,
        }

    parsed_json["exit_code"] = proc.returncode
    parsed_json["wall_ms"] = wall_ms
    return parsed_json


def main():
    parser = argparse.ArgumentParser(description="markov-cero Netlib Benchmark Runner")
    parser.add_argument(
        "--solver",
        default=None,
        help="Path to markov-cero-solve executable (default: auto-detected)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/netlib",
        help="Directory to store/read MPS files (default: data/netlib)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: netlib_results.csv or netlib_extended.csv)",
    )
    parser.add_argument(
        "--engine",
        choices=["primal", "dual"],
        default="primal",
        help="Simplex engine to test (default: primal)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Relative objective tolerance for verification (default: 1e-5)",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Run extended benchmark set (requires network/large instances)",
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        default=None,
        help="Subset of instances to run",
    )
    args = parser.parse_args()

    if args.instances is not None:
        target_instances = args.instances
        default_output = (
            "evidence/netlib_extended.csv" if args.extended else "evidence/netlib_results.csv"
        )
    elif args.extended:
        target_instances = EXTENDED_INSTANCES
        default_output = "evidence/netlib_extended.csv"
    else:
        target_instances = OFFLINE_INSTANCES
        default_output = "evidence/netlib_results.csv"

    output_path = args.output or default_output

    solver = args.solver or find_solver_binary()
    if not solver:
        print(
            "[-] Error: markov-cero-solve executable not found. Build the project first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"=== markov-cero Netlib Benchmark Suite ===")
    print(f"Solver:    {solver}")
    print(f"Engine:    {args.engine}")
    print(f"Data dir:  {args.data_dir}")
    print(f"Output:    {output_path}")
    print(f"Instances: {len(target_instances)}")
    print("=" * 105)

    results = []
    all_passed = True

    header_fmt = "{:<10} {:>8} {:>8} {:>18} {:>18} {:>10} {:>10} {:>10} {:>10}"
    row_fmt = "{:<10} {:>8} {:>8} {:>18.6f} {:>18.6f} {:>10.2e} {:>10} {:>10.1f} {:>10}"

    print(
        header_fmt.format(
            "Instance", "Rows", "Cols", "Ref Objective", "markov-cero Obj", "Rel Error", "Iters", "Time(ms)", "Status"
        )
    )
    print("-" * 105)

    for name in target_instances:
        if name not in NETLIB_BENCHMARKS:
            print(f"[!] Warning: Unknown instance {name}, skipping.")
            continue

        meta = NETLIB_BENCHMARKS[name]
        try:
            mps_file = ensure_instance(name, args.data_dir)
        except Exception as e:
            print(f"[-] {name}: Download failed: {e}")
            all_passed = False
            continue

        res = run_instance(solver, mps_file, engine=args.engine)
        status = res.get("status", "Unknown")
        verified = res.get("verified", False)
        computed_obj = res.get("objective", float("nan"))
        ref_obj = meta["optimal"]

        rel_error = float("nan")
        if isinstance(computed_obj, (int, float)) and status == "Optimal":
            rel_error = abs(computed_obj - ref_obj) / max(1.0, abs(ref_obj))

        iters = res.get("phase_one_iterations", 0) + res.get("phase_two_iterations", 0)
        time_ms = res.get("runtime_ms", res.get("wall_ms", 0.0))

        is_passed = (
            res.get("exit_code") == 0
            and status == "Optimal"
            and verified
            and (rel_error < args.tolerance)
        )

        if not is_passed:
            all_passed = False
            verdict = "FAIL"
        else:
            verdict = "PASS"

        print(
            row_fmt.format(
                name.upper(),
                meta["rows"],
                meta["cols"],
                ref_obj,
                computed_obj,
                rel_error,
                iters,
                time_ms,
                verdict,
            )
        )

        record = {
            "instance": name.upper(),
            "rows": meta["rows"],
            "cols": meta["cols"],
            "reference_objective": ref_obj,
            "computed_objective": computed_obj,
            "relative_error": rel_error,
            "status": status,
            "verified": verified,
            "canonical_verified": res.get("canonical_verified", False),
            "original_verified": res.get("original_verified", False),
            "phase_one_iterations": res.get("phase_one_iterations", 0),
            "phase_two_iterations": res.get("phase_two_iterations", 0),
            "total_iterations": iters,
            "runtime_ms": time_ms,
            "max_primal_violation": res.get("maximum_primal_violation", 0.0),
            "max_dual_violation": res.get("maximum_canonical_dual_violation", 0.0),
            "pass": verdict == "PASS",
        }
        if args.extended or "extended" in output_path:
            record["reproducibility"] = "requires network"
        results.append(record)

    print("-" * 105)

    # Save CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[+] Results written to {output_path}")
    passed_count = sum(1 for r in results if r["pass"])
    print(f"[+] Summary: {passed_count}/{len(results)} Netlib benchmark problems passed.")

    if not all_passed:
        print("[-] Some benchmark problems failed.", file=sys.stderr)
        sys.exit(1)
    else:
        print("[+] 100% of benchmark problems PASSED with mathematical optimality verification!")
        sys.exit(0)


if __name__ == "__main__":
    main()
