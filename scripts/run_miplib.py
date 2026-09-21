#!/usr/bin/env python3
"""
markov-cero MIPLIB Benchmark Runner & Test Harness
Evaluates markov-cero sovereign Mixed-Integer Linear Programming (MILP) branch-and-cut solver
against standard MIPLIB problems. Verifies optimality, primal feasibility, and integrality.
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

# Canonical MIPLIB reference benchmark instances and verified integer optima
MIPLIB_BENCHMARKS = {
    "stein9": {
        "rows": 13,
        "cols": 9,
        "integers": 9,
        "optimal": 5.0,
        "description": "Classic MIPLIB set covering problem (9 variables)",
        "url": "https://raw.githubusercontent.com/ruppinlab/MORSE/master/mps_files/stein9.mps.gz",
    },
    "stein15": {
        "rows": 36,
        "cols": 15,
        "integers": 15,
        "optimal": 9.0,
        "description": "Classic MIPLIB set covering problem (15 variables)",
        "url": "https://raw.githubusercontent.com/ruppinlab/MORSE/master/mps_files/stein15.mps.gz",
    },
    "flugpl": {
        "rows": 18,
        "cols": 18,
        "integers": 11,
        "optimal": 1201500.0,
        "description": "Harvey M. Wagner airline capacity assignment model",
        "url": "https://raw.githubusercontent.com/coin-or-tools/Data-miplib3/master/flugpl.gz",
    },
}


def find_solver_binary() -> Optional[str]:
    candidates = [
        "_deployment-phase2-build/markov-cero-solve",
        "build/markov-cero-solve",
        "bin/markov-cero-solve",
        "./markov-cero-solve",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    return None


def ensure_instance(name: str, data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    mps_path = os.path.join(data_dir, f"{name}.mps")
    if os.path.isfile(mps_path) and os.path.getsize(mps_path) > 0:
        return mps_path

    meta = MIPLIB_BENCHMARKS.get(name)
    if not meta or "url" not in meta:
        raise ValueError(f"Unknown instance: {name}")

    url = meta["url"]
    print(f"[*] Downloading {name} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "markov-cero-runner/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()

    # Decompress if gzipped
    if raw[:2] == b"\x1f\x8b":
        content = gzip.decompress(raw).decode("utf-8")
    else:
        content = raw.decode("utf-8")

    with open(mps_path, "w", encoding="utf-8") as f:
        f.write(content)

    return mps_path


def run_instance(
    solver_bin: str,
    mps_path: str,
    time_limit_sec: float = 60.0,
    timeout_sec: int = 120,
) -> Dict[str, Any]:
    cmd = [solver_bin, mps_path, "--engine", "milp", "--time-limit", str(time_limit_sec)]
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
    parser = argparse.ArgumentParser(description="markov-cero MIPLIB Benchmark Runner")
    parser.add_argument(
        "--solver",
        default=None,
        help="Path to markov-cero-solve executable (default: auto-detected)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/miplib",
        help="Directory to store/read MPS files (default: data/miplib)",
    )
    parser.add_argument(
        "--output",
        default="evidence/miplib_results.csv",
        help="Output CSV path (default: evidence/miplib_results.csv)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Relative objective tolerance for verification (default: 1e-4)",
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        default=["stein9", "stein15", "flugpl"],
        help="Subset of instances to run (default: stein9 stein15 flugpl)",
    )
    args = parser.parse_args()

    solver = args.solver or find_solver_binary()
    if not solver:
        print("[-] Error: markov-cero-solve executable not found. Build the project first.", file=sys.stderr)
        sys.exit(1)

    print(f"=== markov-cero MIPLIB Benchmark Suite ===")
    print(f"Solver:    {solver}")
    print(f"Engine:    milp")
    print(f"Data dir:  {args.data_dir}")
    print(f"Output:    {args.output}")
    print(f"Instances: {len(args.instances)}")
    print("=" * 115)

    results = []
    all_passed = True

    header_fmt = "{:<10} {:>6} {:>6} {:>6} {:>18} {:>18} {:>10} {:>8} {:>10} {:>10}"
    row_fmt = "{:<10} {:>6} {:>6} {:>6} {:>18.6f} {:>18.6f} {:>10.2e} {:>8} {:>10.1f} {:>10}"

    print(
        header_fmt.format(
            "Instance", "Rows", "Cols", "Ints", "Ref Objective", "markov-cero Obj", "Rel Error", "Nodes", "Time(ms)", "Status"
        )
    )
    print("-" * 115)

    for name in args.instances:
        if name not in MIPLIB_BENCHMARKS:
            print(f"[!] Warning: Unknown instance {name}, skipping.")
            continue

        meta = MIPLIB_BENCHMARKS[name]
        try:
            mps_file = ensure_instance(name, args.data_dir)
        except Exception as e:
            print(f"[-] {name}: Failed to access/download: {e}")
            all_passed = False
            continue

        res = run_instance(solver, mps_file)
        status = res.get("status", "Unknown")
        verified = res.get("verified", False)
        computed_obj = res.get("objective", float("nan"))
        ref_obj = meta["optimal"]

        rel_error = float("nan")
        if isinstance(computed_obj, (int, float)) and status == "Optimal":
            rel_error = abs(computed_obj - ref_obj) / max(1.0, abs(ref_obj))

        nodes = res.get("nodes_explored", 0)
        time_ms = res.get("runtime_ms", res.get("wall_ms", 0.0))

        is_passed = (
            res.get("exit_code") == 0
            and status == "Optimal"
            and verified
            and (rel_error <= args.tolerance)
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
                meta["integers"],
                ref_obj,
                computed_obj,
                rel_error,
                nodes,
                time_ms,
                verdict,
            )
        )

        record = {
            "instance": name.upper(),
            "rows": meta["rows"],
            "cols": meta["cols"],
            "integers": meta["integers"],
            "reference_objective": ref_obj,
            "computed_objective": computed_obj,
            "relative_error": rel_error,
            "status": status,
            "verified": verified,
            "original_verified": res.get("original_verified", False),
            "nodes_explored": nodes,
            "lp_iterations": res.get("lp_iterations", 0),
            "cuts_generated": res.get("cuts_generated", 0),
            "heuristics_found": res.get("heuristics_found", 0),
            "runtime_ms": time_ms,
            "max_primal_violation": res.get("maximum_primal_violation", 0.0),
            "max_integrality_violation": res.get("maximum_integrality_violation", 0.0),
            "pass": verdict == "PASS",
        }
        results.append(record)

    print("-" * 115)

    # Save CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[+] Results written to {args.output}")
    passed_count = sum(1 for r in results if r["pass"])
    print(f"[+] Summary: {passed_count}/{len(results)} MIPLIB benchmark problems passed.")

    if not all_passed:
        print("[-] Some benchmark problems failed.", file=sys.stderr)
        sys.exit(1)
    else:
        print("[+] 100% of benchmark problems PASSED with sovereign MILP optimality verification!")
        sys.exit(0)


if __name__ == "__main__":
    main()
