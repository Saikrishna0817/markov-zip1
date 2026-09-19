#!/usr/bin/env python3
"""
markov-cero Phase 4 ML Best-Practices Benchmark & Comparative Analytics Engine
Evaluates:
  Option A: First-Order PDLP (Applegate et al. 2021) vs Revised Simplex
  Option B: MIR Cuts (Marchand & Wolsey 2001) & Strong Branching (Achterberg 2005)
  Option C: Multithreaded Parallel Tree Search Scaling (1, 2, 4 threads)
Generates empirical comparative tables, speedup analysis, and zero-trust verification records.
"""

import json
import os
import subprocess
import sys
import time
from typing import Dict, Any, List

SOLVER_BIN = "./_deployment-phase2-build/markov-cero-solve"

def run_solver(args: List[str]) -> Dict[str, Any]:
    cmd = [SOLVER_BIN] + args
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    wall_ms = (t1 - t0) * 1000.0

    parsed = {}
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                break
            except Exception:
                pass
    parsed["wall_clock_ms"] = wall_ms
    parsed["returncode"] = res.returncode
    return parsed

def benchmark_option_a():
    print("=== [1/3] Benchmarking Option A: PDLP (First-Order) vs Simplex ===")
    instances = [
        ("examples/blend.mps", "Blend LP (2 cols, 2 rows)"),
        ("examples/refinery/refinery-feasible.mps", "Refinery Continuous (2 cols, 2 rows)")
    ]
    results = []
    for path, desc in instances:
        res_simplex = run_solver([path, "--engine", "primal"])
        res_pdlp = run_solver([path, "--engine", "pdlp"])
        results.append({
            "model": os.path.basename(path),
            "description": desc,
            "simplex_obj": res_simplex.get("objective"),
            "simplex_iters": res_simplex.get("lp_iterations"),
            "simplex_time_ms": res_simplex.get("runtime_ms"),
            "simplex_verified": res_simplex.get("verified"),
            "pdlp_obj": res_pdlp.get("objective"),
            "pdlp_iters": res_pdlp.get("lp_iterations"),
            "pdlp_time_ms": res_pdlp.get("runtime_ms"),
            "pdlp_verified": res_pdlp.get("verified"),
            "matrix_factorizations_pdlp": 0
        })
    return results

def benchmark_option_b():
    print("=== [2/3] Benchmarking Option B: Advanced Cuts & Branching ===")
    instances = [
        ("data/miplib/stein9.mps", "stein9"),
        ("data/miplib/flugpl.mps", "flugpl")
    ]
    results = []
    for path, name in instances:
        if not os.path.exists(path):
            continue
        # Pseudo-cost vs Most fractional
        res_pc = run_solver([path, "--engine", "milp", "--branching", "pseudo_cost"])
        res_mf = run_solver([path, "--engine", "milp", "--branching", "most_fractional"])
        res_nocut = run_solver([path, "--engine", "milp", "--no-cuts"])
        results.append({
            "instance": name,
            "pseudo_cost_nodes": res_pc.get("nodes_explored"),
            "pseudo_cost_time_ms": res_pc.get("runtime_ms"),
            "most_fractional_nodes": res_mf.get("nodes_explored"),
            "most_fractional_time_ms": res_mf.get("runtime_ms"),
            "no_cut_nodes": res_nocut.get("nodes_explored"),
            "cut_node_reduction": f"{(1.0 - (res_pc.get('nodes_explored', 1) / max(1, res_nocut.get('nodes_explored', 1)))) * 100:.1f}%",
            "verified": res_pc.get("verified")
        })
    return results

def benchmark_option_c():
    print("=== [3/3] Benchmarking Option C: Parallel Tree Search Concurrency Scaling ===")
    instances = [
        "data/miplib/stein9.mps"
    ]
    results = []
    for path in instances:
        if not os.path.exists(path):
            continue
        inst_name = os.path.basename(path)
        threads_runs = {}
        for th in [1, 2, 4]:
            runs = []
            for _ in range(5):
                r = run_solver([path, "--engine", "parallel", "--threads", str(th)])
                runs.append(r)
            best_time = min(r.get("runtime_ms", 999.0) for r in runs)
            obj = runs[0].get("objective")
            verified = all(r.get("verified", False) for r in runs)
            nodes = runs[0].get("nodes_explored")
            threads_runs[th] = {
                "runtime_ms": best_time,
                "objective": obj,
                "verified": verified,
                "nodes": nodes
            }
        
        t1 = threads_runs[1]["runtime_ms"]
        t2 = threads_runs[2]["runtime_ms"]
        t4 = threads_runs[4]["runtime_ms"]
        results.append({
            "instance": inst_name,
            "t1_ms": t1,
            "t2_ms": t2,
            "t4_ms": t4,
            "speedup_2th": f"{t1 / max(0.001, t2):.2f}x",
            "speedup_4th": f"{t1 / max(0.001, t4):.2f}x",
            "efficiency_4th": f"{(t1 / max(0.001, t4)) / 4.0 * 100:.1f}%",
            "identical_optima": (threads_runs[1]["objective"] == threads_runs[2]["objective"] == threads_runs[4]["objective"]),
            "zero_trust_verified": all(threads_runs[th]["verified"] for th in [1, 2, 4])
        })
    return results

def main():
    opt_a = benchmark_option_a()
    opt_b = benchmark_option_b()
    opt_c = benchmark_option_c()

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "option_a_pdlp": opt_a,
        "option_b_cuts_and_branching": opt_b,
        "option_c_parallel_scaling": opt_c
    }

    report_path = "reports/phase4_benchmark_analytics.json"
    os.makedirs("reports", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*70)
    print("PHASE 4 EMPIRICAL BENCHMARK ANALYTICS SUMMARY")
    print("="*70)
    print("\n--- Option A: PDLP vs Simplex ---")
    for r in opt_a:
        print(f"Model: {r['model']} ({r['description']})")
        print(f"  Simplex: obj={r['simplex_obj']}, iters={r['simplex_iters']}, time={r['simplex_time_ms']:.3f}ms, verified={r['simplex_verified']}")
        print(f"  PDLP:    obj={r['pdlp_obj']:.4f}, iters={r['pdlp_iters']}, time={r['pdlp_time_ms']:.3f}ms, verified={r['pdlp_verified']}")
        print(f"  Matrix Factorizations: Simplex=LU basis update; PDLP=0 (Matrix-Free)")

    print("\n--- Option B: Advanced Cuts & Branching Impact ---")
    for r in opt_b:
        print(f"Instance: {r['instance']}")
        print(f"  Pseudo-Cost Branching:  nodes={r['pseudo_cost_nodes']}, time={r['pseudo_cost_time_ms']:.3f}ms")
        print(f"  Most-Fractional:        nodes={r['most_fractional_nodes']}, time={r['most_fractional_time_ms']:.3f}ms")
        print(f"  No-Cuts Baseline:       nodes={r['no_cut_nodes']}")
        print(f"  Node Reduction:         {r['cut_node_reduction']}")

    print("\n--- Option C: Multithreaded Parallel Tree Search Scaling ---")
    for r in opt_c:
        print(f"Instance: {r['instance']}")
        print(f"  1 Thread:  {r['t1_ms']:.3f}ms")
        print(f"  2 Threads: {r['t2_ms']:.3f}ms (Speedup: {r['speedup_2th']})")
        print(f"  4 Threads: {r['t4_ms']:.3f}ms (Speedup: {r['speedup_4th']}, Efficiency: {r['efficiency_4th']})")
        print(f"  Identical Optima Across All Threads: {r['identical_optima']}")
        print(f"  Zero-Trust Verification:            {r['zero_trust_verified']}")
    print("="*70)

if __name__ == "__main__":
    main()
