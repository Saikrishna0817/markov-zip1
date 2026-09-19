"""Thin M1 Python interface; authoritative parsing remains in C++."""
from __future__ import annotations
import json, subprocess
from pathlib import Path

def inspect_mps(path: str | Path, executable: str = "markov-cero-mps-inspect") -> dict[str, object]:
    completed = subprocess.run([executable, str(path)], check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)
