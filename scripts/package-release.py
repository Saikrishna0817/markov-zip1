#!/usr/bin/env python3
import hashlib
import os
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
exclude = {".git", "build", "_verify-gcc", "_verify-clang", "__pycache__"}
files = []
for path in root.rglob("*"):
    rel = path.relative_to(root)
    if not path.is_file() or any(part in exclude or part.startswith("_verify-") for part in rel.parts):
        continue
    if rel.as_posix() in {"evidence/local-verification-report.txt", "evidence/environment-local.json"}:
        continue
    files.append((path, Path("sihopt") / rel))
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path, arcname in sorted(files, key=lambda item: item[1].as_posix()):
        info = zipfile.ZipInfo(arcname.as_posix(), (2026, 9, 13, 0, 0, 0))
        info.create_system = 3
        info.external_attr = (0o755 if os.access(path, os.X_OK) else 0o644) << 16
        archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
digest = hashlib.sha256(out.read_bytes()).hexdigest()
out.with_suffix(out.suffix + ".sha256").write_text(
    f"{digest}  {out.name}\n", encoding="utf-8"
)
print(digest)
