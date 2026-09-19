#!/usr/bin/env python3
"""Deterministically generate the M5 libFuzzer seed corpora.

The corpus must be reproducible without network access so that evidence
records can state exactly which seeds were used. Re-running this script
always produces byte-identical files.

sparse_basis seed layout (consumed by tests/fuzz/sparse_basis_fuzz.cpp):
  byte 0              : n - 1 (dimension minus one)
  next n*n bytes      : column-major raw int8 values; kept iff (raw & 3) != 0,
                        value = raw / 32.0; the harness then adds n + 1 to
                        every diagonal entry
  next n bytes        : raw int8 right-hand-side values, value = raw / 16.0
  remaining bytes     : replacement column index and perturbation position
"""
from __future__ import annotations

import sys
from pathlib import Path

MPS_SEEDS: dict[str, str] = {
    # Minimal free-format LP: min -x - y s.t. x + y <= 4, x,y >= 0.
    "tiny-lp.mps": "\n".join(
        [
            "NAME          TINYLP",
            "ROWS",
            " N  COST",
            " L  R1",
            "COLUMNS",
            "    X         COST       -1.0        R1         1.0",
            "    Y         COST       -1.0        R1         1.0",
            "RHS",
            "    RHS1      R1         4.0",
            "ENDATA",
            "",
        ]
    ),
    # Equality row, RANGES on an L row, FX bound, and an integer marker block.
    "range-fx-int.mps": "\n".join(
        [
            "NAME          RANGEFX",
            "ROWS",
            " N  OBJ",
            " E  E1",
            " L  CAP",
            "COLUMNS",
            "    MARKER                 'MARKER'                 'INTORG'",
            "    XI        OBJ         2.0        E1         1.0",
            "    XI        CAP         1.0",
            "    MARKER                 'MARKER'                 'INTEND'",
            "    C         OBJ         3.0        E1         1.0",
            "RHS",
            "    RHS1      E1         2.5        CAP        6.0",
            "RANGES",
            "    RNG       CAP        2.0",
            "BOUNDS",
            " FX BND       C          1.5",
            "ENDATA",
            "",
        ]
    ),
    # Maximization sense with a D-exponent numeric token.
    "maximize-dexp.mps": "\n".join(
        [
            "NAME          MAXDEXP",
            "OBJSENSE",
            "    MAX",
            "ROWS",
            " N  PROFIT",
            " G  DEM",
            "COLUMNS",
            "    A         PROFIT     1.5D+00     DEM        1.0",
            "    B         PROFIT     -2.0        DEM        1.0",
            "RHS",
            "    RHS1      DEM        3.5",
            "ENDATA",
            "",
        ]
    ),
}


def entry(value: float) -> int:
    """Encode a matrix value as a kept int8 raw byte (value = raw / 32.0)."""
    raw = int(value * 32.0)
    if raw & 3 == 0:
        raw += 1  # low two bits must be nonzero for the value to be kept
    return raw & 0xFF


def absent() -> int:
    return 0


def sparse_seed(n: int, columns: list[list[float]], rhs: list[float], tail: list[int]) -> bytes:
    assert len(columns) == n and all(len(col) == n for col in columns)
    assert len(rhs) == n
    payload = bytearray([n - 1])
    for j in range(n):
        for i in range(n):
            value = columns[j][i]
            payload.append(entry(value) if value != 0.0 else absent())
    for value in rhs:
        payload += (int(value * 16.0) & 0xFF).to_bytes(1, "little")
    payload += bytes(index & 0xFF for index in tail)
    return bytes(payload)


def build_sparse_seeds() -> dict[str, bytes]:
    seeds: dict[str, bytes] = {}
    # 3x3 unit-lower-triangular pattern with a clean replacement tail.
    seeds["lower3x3.bin"] = sparse_seed(
        3,
        [[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.25, 0.0, 1.0]],
        [1.0, -2.0, 3.0],
        [1, 2],  # replace column 1, perturb row 2
    )
    # 1x1 boundary size.
    seeds["tiny1x1.bin"] = sparse_seed(1, [[2.0]], [7.0], [0, 0])
    # 24x24 (harness maximum) sparse diagonal-dominant matrix.
    n = 24
    columns = [[0.0] * n for _ in range(n)]
    for j in range(n):
        columns[j][j] = 1.0
        for i in range(n):
            if i != j and (i + j) % 7 == 0:
                columns[j][i] = 0.25
    seeds["wide24x24.bin"] = sparse_seed(n, columns, [(i % 9) - 4 for i in range(n)], [5, 11])
    return seeds


def main() -> None:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    mps_dir = root / "tests/fuzz/corpus/mps"
    sparse_dir = root / "tests/fuzz/corpus/sparse_basis"
    mps_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    for name, text in sorted(MPS_SEEDS.items()):
        (mps_dir / name).write_bytes(text.encode("ascii"))
    sparse_seeds = build_sparse_seeds()
    for name, blob in sorted(sparse_seeds.items()):
        (sparse_dir / name).write_bytes(blob)
    print(f"wrote {len(MPS_SEEDS)} MPS seeds and {len(sparse_seeds)} sparse-basis seeds")


if __name__ == "__main__":
    main()
