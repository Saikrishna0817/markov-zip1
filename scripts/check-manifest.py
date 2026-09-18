#!/usr/bin/env python3
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = root / "provenance/source-manifest.sha256"
errors = []
for line in manifest.read_text(encoding="utf-8").splitlines():
    digest, rel = line.split("  ", 1)
    path = root / rel
    if not path.is_file():
        errors.append("missing " + rel)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        errors.append("mismatch " + rel)
if errors:
    raise SystemExit("\n".join(errors))
print("source manifest passed")
