#!/usr/bin/env python3
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
for forbidden in ["src/lp/simplex", "src/lp/pdlp", "src/mip", "src/qp", "src/gpu"]:
    if (root / forbidden).exists():
        raise SystemExit("M4 forbidden future-solver path exists: " + forbidden)
text = "\n".join(
    path.read_text(errors="ignore")
    for path in (root / "src").rglob("*")
    if path.is_file()
)
for token in ["branch_and_bound", "pdhg", "cudaKernel"]:
    if token in text:
        raise SystemExit("possible solver implementation token: " + token)
print("M4 scope guard passed")
