#!/usr/bin/env python3
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
src = root / "src"

forbidden_paths = [
    "src/lp/simplex",
    "src/lp/pdlp",
    "src/mip",
    "src/qp",
    "src/gpu",
]
for rel in forbidden_paths:
    if (root / rel).exists():
        raise SystemExit("forbidden future-solver path exists: " + rel)

allowed_lp = {src / "lp" / "reference", src / "lp" / "dual"}
if (src / "lp").is_dir():
    for child in (src / "lp").iterdir():
        if child.is_dir() and child not in allowed_lp:
            raise SystemExit("unexpected solver directory: " + str(child.relative_to(root)))

tokens = ("branch_and_bound", "pdhg", "cudaKernel")
for path in src.rglob("*"):
    if not path.is_file():
        continue
    text = path.read_text(errors="ignore")
    for token in tokens:
        if token in text:
            raise SystemExit(f"possible solver implementation token {token} in {path.relative_to(root)}")

print("scope guard passed")
