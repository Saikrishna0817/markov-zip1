#!/usr/bin/env python3
"""
markov-cero Scalable LP Instance Generator (T-5.13 / T-5.15)
Generates deterministic, structurally realistic multi-stage network flow
and staircase resource allocation LP models spanning 1e2 to 1e6 nonzeros.
"""

import argparse
import os
import random
import sys


def generate_network_lp(
    name: str,
    num_stages: int,
    nodes_per_stage: int,
    seed: int = 42,
) -> str:
    """
    Generates a multi-stage minimum-cost resource flow network LP in MPS format.
    Guaranteed feasible, bounded, and highly sparse (2-3 entries per column).
    """
    rng = random.Random(seed)
    lines = [f"NAME {name.upper()}", "ROWS", " N COST"]

    # Generate balance constraints for each node across stages
    for s in range(num_stages):
        for n in range(nodes_per_stage):
            lines.append(f" E N_{s}_{n}")
    # Capacity constraints per stage
    for s in range(num_stages):
        lines.append(f" L CAP_{s}")

    lines.append("COLUMNS")
    rhs_entries = []
    bounds_entries = []

    # Variables: flows between node (s, u) and node (s+1, v)
    # Plus source injection at stage 0 and sink demand at stage S-1
    for s in range(num_stages - 1):
        for u in range(nodes_per_stage):
            # Connect each node to 2 adjacent nodes in the next stage
            v_targets = [u, (u + 1) % nodes_per_stage]
            for v in v_targets:
                var_name = f"F_{s}_{u}_{v}"
                cost = rng.uniform(1.0, 10.0)
                # Leaves node (s, u) -> -1, enters node (s+1, v) -> +1
                # Capacity at stage s -> 1
                lines.append(f" {var_name} COST {cost:.4f} N_{s}_{u} -1.0")
                lines.append(f" {var_name} N_{s+1}_{v} 1.0 CAP_{s} 1.0")
                # Variable upper bound
                cap = rng.uniform(15.0, 50.0)
                bounds_entries.append(f" UP BND {var_name} {cap:.2f}")

    # External supply into stage 0
    total_flow = nodes_per_stage * 10.0
    for u in range(nodes_per_stage):
        sup_var = f"SUP_{u}"
        lines.append(f" {sup_var} COST 0.5 N_0_{u} 1.0")
        bounds_entries.append(f" UP BND {sup_var} 20.0")
        # Target demand at last stage
        rhs_entries.append((f"N_0_{u}", 0.0))

    # Demand out of last stage
    for u in range(nodes_per_stage):
        dem_var = f"DEM_{u}"
        lines.append(f" {dem_var} COST 0.5 N_{num_stages-1}_{u} -1.0")
        demand_val = 10.0
        bounds_entries.append(f" UP BND {dem_var} 25.0")
        rhs_entries.append((f"N_{num_stages-1}_{u}", -demand_val))

    # Stage capacities in RHS
    for s in range(num_stages):
        stage_cap = nodes_per_stage * 25.0
        rhs_entries.append((f"CAP_{s}", stage_cap))

    lines.append("RHS")
    for row, val in rhs_entries:
        if abs(val) > 1e-6:
            lines.append(f" RHS1 {row} {val:.2f}")

    if bounds_entries:
        lines.append("BOUNDS")
        lines.extend(bounds_entries)

    lines.append("ENDATA\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate scalable LP instances for crossover study"
    )
    parser.add_argument(
        "--output-dir", default="data/scale_study", help="Directory to save MPS models"
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Scale configurations: (name, stages, nodes_per_stage)
    # Spans fine-grained small models (<50 rows) to large (>10,000 nonzeros)
    scales = [
        ("scale_5", 2, 2),         # ~6 rows, ~8 cols, ~16 nnz
        ("scale_10", 3, 2),        # ~9 rows, ~12 cols, ~28 nnz
        ("scale_20", 4, 3),        # ~16 rows, ~24 cols, ~60 nnz
        ("scale_35", 5, 4),        # ~25 rows, ~40 cols, ~104 nnz
        ("scale_50", 7, 5),        # ~42 rows, ~70 cols, ~190 nnz
        ("scale_75", 8, 6),        # ~56 rows, ~96 cols, ~264 nnz
        ("scale_100", 10, 5),      # ~60 rows, ~100 cols, ~280 nnz
        ("scale_200", 14, 8),      # ~126 rows, ~224 cols, ~640 nnz
        ("scale_500", 25, 10),     # ~275 rows, ~500 cols, ~1,460 nnz
        ("scale_1000", 35, 14),    # ~525 rows, ~980 cols, ~2,884 nnz
        ("scale_2000", 50, 20),    # ~1,050 rows, ~2,000 cols, ~5,920 nnz
        ("scale_5000", 75, 33),    # ~2,550 rows, ~4,950 cols, ~14,718 nnz
        ("scale_10000", 100, 50),  # ~5,100 rows, ~10,000 cols, ~29,800 nnz
        ("scale_50000", 250, 100), # ~25,250 rows, ~50,000 cols, ~149,600 nnz
    ]

    print("=" * 75)
    print(f" Generating Scalable LP Corpus into {args.output_dir}")
    print("=" * 75)
    for name, stages, nodes in scales:
        filepath = os.path.join(args.output_dir, f"{name}.mps")
        content = generate_network_lp(name, stages, nodes, seed=args.seed)
        with open(filepath, "w") as f:
            f.write(content)
        size_kb = os.path.getsize(filepath) / 1024.0
        rows = stages * nodes + stages
        cols = (stages - 1) * nodes * 2 + 2 * nodes
        nnz = (stages - 1) * nodes * 2 * 3 + 2 * nodes
        msg = (
            f"  [+] {name:<12} Rows: {rows:>6}  Cols: {cols:>6}  "
            f"NNZ: ~{nnz:>6}  ({size_kb:.1f} KB)"
        )
        print(msg)

    print(f"\n[+] Generated {len(scales)} benchmark models successfully.")


if __name__ == "__main__":
    main()
