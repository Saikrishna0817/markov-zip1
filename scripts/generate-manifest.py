#!/usr/bin/env python3
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out = root / "provenance/source-manifest.sha256"
rows = []
for path in sorted(root.rglob("*")):
    rel = path.relative_to(root).as_posix()
    if not path.is_file() or rel == "provenance/source-manifest.sha256":
        continue
    if ".git" in path.parts or "__pycache__" in path.parts:
        continue
    parts = path.relative_to(root).parts
    if rel.startswith("build/") or any(part.startswith("_verify-") or part.startswith("_m5-") for part in parts):
        continue
    if parts[0] in {"cmake-build-debug", "cmake-build-release"}:
        continue
    if rel in {"evidence/local-verification-report.txt", "evidence/environment-local.json"}:
        continue
    rows.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + rel)
out.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"wrote {len(rows)} checksums")
