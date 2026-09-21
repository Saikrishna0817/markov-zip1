#!/usr/bin/env python3
"""
markov-cero GPU Benchmark Runner
Three-way algorithmic evaluation: CPU Simplex vs CPU PDLP vs GPU PDLP.
Generates a consolidated, reproducible CSV reporting four-part GPU timing (D-GPU-08).
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Dict, Any, List, Optional

# Canonical reference benchmarks with optimal objective values
BENCHMARKS = {
    "afiro": {"rows": 28, "cols": 32, "optimal": -464.753142857143},
    "blend": {"rows": 75, "cols": 84, "optimal": -30.8121498457},
    "adlittle": {"rows": 57, "cols": 97, "optimal": 225494.96316238},
    "sc50a": {"rows": 51, "cols": 48, "optimal": -64.575077058564},
    "sc50b": {"rows": 51, "cols": 48, "optimal": -70.000000000000},
    "sc105": {"rows": 106, "cols": 103, "optimal": -52.202061211707},
    "sc205": {"rows": 206, "cols": 203, "optimal": -52.202061211707},
    "share2b": {"rows": 97, "cols": 79, "optimal": -415.732240741418},
    "share1b": {"rows": 118, "cols": 225, "optimal": -76589.3185791855},
    "recipe": {"rows": 92, "cols": 180, "optimal": -266.616000000000},
    "scsd1": {"rows": 78, "cols": 760, "optimal": 8.66666667},
    "scsd6": {"rows": 148, "cols": 1350, "optimal": 50.5},
    "beaconfd": {"rows": 174, "cols": 262, "optimal": 33592.4858072000},
    "scorpion": {"rows": 389, "cols": 358, "optimal": 1878.1248227381},
}

DEFAULT_INSTANCES = ["afiro", "blend", "sc50a", "sc50b", "adlittle", "scsd1"]


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


def run_configuration(
    solver: str,
    mps_path: str,
    engine: str,
    backend: Optional[str] = None,
    tolerance: float = 1e-4,
    timeout: int = 120,
) -> Dict[str, Any]:
    cmd = [solver, mps_path, "--engine", engine]
    if backend:
        cmd.extend(["--backend", backend])
    if engine == "pdlp":
        cmd.extend(["--tolerance", str(tolerance)])

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        wall_ms = (time.perf_counter() - t0) * 1000.0
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "status": "Timeout",
            "verified": False,
            "objective": float("nan"),
            "iterations": 0,
            "time_ms": timeout * 1000.0,
            "h2d_ms": 0.0,
            "kernel_ms": 0.0,
            "d2h_ms": 0.0,
            "total_ms": timeout * 1000.0,
            "error": f"Timed out after {timeout}s",
        }

    parsed = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                break
            except Exception:
                pass

    if parsed is None:
        return {
            "exit_code": proc.returncode,
            "status": "ParseError",
            "verified": False,
            "objective": float("nan"),
            "iterations": 0,
            "time_ms": wall_ms,
            "h2d_ms": 0.0,
            "kernel_ms": 0.0,
            "d2h_ms": 0.0,
            "total_ms": wall_ms,
            "error": proc.stderr[:200],
        }

    iters = parsed.get("lp_iterations", parsed.get("total_iterations", 0))
    time_ms = parsed.get("total_ms", parsed.get("runtime_ms", wall_ms))
    return {
        "exit_code": proc.returncode,
        "status": parsed.get("status", "Unknown"),
        "verified": parsed.get("verified", False),
        "objective": parsed.get("objective", float("nan")),
        "iterations": iters,
        "time_ms": time_ms,
        "h2d_ms": parsed.get("h2d_ms", 0.0),
        "kernel_ms": parsed.get("kernel_ms", 0.0),
        "d2h_ms": parsed.get("d2h_ms", 0.0),
        "total_ms": parsed.get("total_ms", time_ms),
        "error": parsed.get("error", ""),
    }


def main():
    parser = argparse.ArgumentParser(
        description="markov-cero GPU Benchmark Runner (Simplex vs CPU PDLP vs GPU PDLP)"
    )
    parser.add_argument("--solver", help="Path to markov-cero-solve executable")
    parser.add_argument(
        "--instances", nargs="+", default=DEFAULT_INSTANCES, help="Netlib instances to benchmark"
    )
    parser.add_argument("--data-dir", default="data/netlib", help="Directory containing MPS files")
    parser.add_argument(
        "--tolerance", type=float, default=1e-4, help="Relative KKT tolerance for PDLP"
    )
    parser.add_argument("--timeout", type=int, default=120, help="Per-run timeout in seconds")
    parser.add_argument(
        "--output", default="reports/gpu_benchmark.csv", help="Output path for benchmark CSV"
    )

    args = parser.parse_args()

    solver = args.solver or find_solver_binary()
    if not solver or not os.access(solver, os.X_OK):
        print(f"Error: Solver executable not found: {solver}", file=sys.stderr)
        sys.exit(1)

    print("=" * 105)
    print(" markov-cero GPU Acceleration Benchmark (D-GPU-08 / Master Document §18.3)")
    print(f" Solver:    {solver}")
    print(f" Tolerance: {args.tolerance:.1e}")
    print(f" Output:    {args.output}")
    print(f" Instances: {', '.join(args.instances)}")
    print("=" * 105)

    header = (
        "{:<8} {:>5} {:>5} | {:>9} {:>7} | {:>9} {:>7} | {:>7} {:>7} {:>7} {:>8} | {:>8}"
    )
    print(
        header.format(
            "Instance", "Rows", "Cols",
            "Smplx(ms)", "Iters",
            "CPUp(ms)", "Iters",
            "H2D(ms)", "Krn(ms)", "D2H(ms)", "GPUp(ms)",
            "KrnSpdup"
        )
    )
    print("-" * 105)

    records = []
    all_passed = True

    for inst in args.instances:
        meta = BENCHMARKS.get(inst.lower(), {"rows": 0, "cols": 0, "optimal": 0.0})
        mps_file = os.path.join(args.data_dir, f"{inst.lower()}.mps")
        if not os.path.isfile(mps_file):
            alt = os.path.join("examples", f"{inst.lower()}.mps")
            if os.path.isfile(alt):
                mps_file = alt
            else:
                print(f"[!] Warning: Cannot find {mps_file}, skipping.")
                continue

        # 1. CPU Simplex baseline
        res_simplex = run_configuration(
            solver, mps_file, engine="dual", tolerance=args.tolerance, timeout=args.timeout
        )

        # 2. CPU PDLP
        res_cpu = run_configuration(
            solver, mps_file, engine="pdlp", backend="cpu",
            tolerance=args.tolerance, timeout=args.timeout
        )

        # 3. GPU PDLP
        res_gpu = run_configuration(
            solver, mps_file, engine="pdlp", backend="gpu",
            tolerance=args.tolerance, timeout=args.timeout
        )

        speedup_kernel = (
            (res_cpu["time_ms"] / res_gpu["kernel_ms"]) if res_gpu["kernel_ms"] > 0 else 0.0
        )
        speedup_end_to_end = (
            (res_cpu["time_ms"] / res_gpu["total_ms"]) if res_gpu["total_ms"] > 0 else 0.0
        )
        speedup_vs_simplex = (
            (res_simplex["time_ms"] / res_gpu["total_ms"]) if res_gpu["total_ms"] > 0 else 0.0
        )

        inst_passed = (
            res_gpu["status"] == "Optimal"
            and res_gpu["verified"]
            and res_cpu["status"] == "Optimal"
        )
        if not inst_passed:
            all_passed = False

        row = (
            "{:<8} {:>5} {:>5} | {:>9.2f} {:>7} | {:>9.2f} {:>7} | "
            "{:>7.3f} {:>7.3f} {:>7.3f} {:>8.2f} | {:>7.2f}x"
        ).format(
            inst.upper(), meta["rows"], meta["cols"],
            res_simplex["time_ms"], res_simplex["iterations"],
            res_cpu["time_ms"], res_cpu["iterations"],
            res_gpu["h2d_ms"], res_gpu["kernel_ms"], res_gpu["d2h_ms"], res_gpu["total_ms"],
            speedup_kernel
        )
        print(row)

        record = {
            "instance": inst.upper(),
            "rows": meta["rows"],
            "cols": meta["cols"],
            "reference_objective": meta["optimal"],
            "simplex_status": res_simplex["status"],
            "simplex_objective": res_simplex["objective"],
            "simplex_iterations": res_simplex["iterations"],
            "simplex_time_ms": res_simplex["time_ms"],
            "cpu_pdlp_status": res_cpu["status"],
            "cpu_pdlp_objective": res_cpu["objective"],
            "cpu_pdlp_iterations": res_cpu["iterations"],
            "cpu_pdlp_time_ms": res_cpu["time_ms"],
            "gpu_pdlp_status": res_gpu["status"],
            "gpu_pdlp_objective": res_gpu["objective"],
            "gpu_pdlp_iterations": res_gpu["iterations"],
            "gpu_h2d_ms": res_gpu["h2d_ms"],
            "gpu_kernel_ms": res_gpu["kernel_ms"],
            "gpu_d2h_ms": res_gpu["d2h_ms"],
            "gpu_total_ms": res_gpu["total_ms"],
            "speedup_kernel": speedup_kernel,
            "speedup_end_to_end": speedup_end_to_end,
            "speedup_vs_simplex": speedup_vs_simplex,
            "gpu_verified": res_gpu["verified"],
            "pass": inst_passed,
        }
        records.append(record)

    print("-" * 105)

    if records:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"\n[+] Results successfully written to {args.output}")

    if all_passed:
        print("[+] All GPU benchmark problems converged and verified.")
        sys.exit(0)
    else:
        print("[-] One or more benchmarks failed verification.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
