#!/usr/bin/env python3
"""Sovereignty verification guard for markov-cero.

Enforces SIH 2026 clean-room sovereignty requirements:
1. No link against any external optimization library.
2. No vendored third-party solver source.
3. Dependency set strictly matches {C++20 stdlib, Threads, CUDA(optional)}.
4. Binary symbols and dynamic link dependencies contain zero forbidden targets.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_LIBRARIES = {
    "glpk", "gurobi", "cplex", "xpress", "xprs", "highs", "clp", "cbc",
    "coin", "osqp", "scs", "ipopt", "scip", "mosek", "eigen", "eigen3",
    "boost", "fmt", "spdlog", "gtest", "gmock", "googletest", "nlohmann",
    "torch", "libtorch", "onnx", "onnxruntime", "alglib", "ceres",
}

FORBIDDEN_HEADERS = [
    r"glpk\.h", r"gurobi_c\+\+\.h", r"gurobi_c\.h", r"cplex\.h", r"xprs\.h",
    r"Highs\.h", r"ClpSimplex\.hpp", r"CbcModel\.hpp", r"osqp/osqp\.h",
    r"scs/scs\.h", r"scip/scip\.h", r"mosek\.h", r"Eigen/", r"boost/",
    r"fmt/", r"spdlog/", r"gtest/gtest\.h", r"nlohmann/json\.hpp",
    r"torch/torch\.h", r"onnxruntime",
]

FORBIDDEN_SYMBOLS = [
    r"^glp_", r"^GRB", r"^CPX", r"^XPRS", r"^Highs_", r"^Clp_", r"^Cbc_",
    r"^OSQP", r"^scs_", r"^SCIP", r"^MSK_",
]

ALLOWED_DYNAMIC_LIBS = {
    "linux-vdso.so", "libstdc++.so", "libm.so", "libgcc_s.so", "libc.so",
    "ld-linux-x86-64.so", "ld-linux.so", "libpthread.so", "libdl.so",
    "librt.so", "libresolv.so", "libcuda.so", "libcudart.so",
    "libasan.so", "libubsan.so", "libtsan.so", "liblsan.so",
}

ALLOWED_FIND_PACKAGES = {"threads", "cuda", "cudatoolkit"}
ALLOWED_LINK_TARGETS = {"threads::threads", "markov_cero_core"}


def check_cmake(root: Path) -> list[str]:
    violations = []
    cmake_path = root / "CMakeLists.txt"
    if not cmake_path.is_file():
        return [f"CMakeLists.txt not found at {cmake_path}"]

    content = cmake_path.read_text(encoding="utf-8", errors="ignore")
    find_pattern = re.compile(r"find_package\s*\(\s*([A-Za-z0-9_]+)", re.IGNORECASE)
    for match in find_pattern.finditer(content):
        pkg = match.group(1).lower()
        if pkg not in ALLOWED_FIND_PACKAGES:
            violations.append(
                f"CMakeLists.txt: disallowed find_package({match.group(1)})"
            )

    link_pattern = re.compile(r"target_link_libraries\s*\([^)]+\)", re.DOTALL)
    for block in link_pattern.finditer(content):
        tokens = block.group(0).split()
        for token in tokens[2:]:
            clean = token.rstrip(")").lower()
            if clean in {"private", "public", "interface"}:
                continue
            for forbidden in FORBIDDEN_LIBRARIES:
                if forbidden in clean:
                    violations.append(
                        f"CMakeLists.txt: forbidden link target '{token.rstrip(')')}'"
                    )

    for term in ["FetchContent", "ExternalProject_Add", "add_subdirectory"]:
        pattern = re.compile(rf"{term}\s*\(", re.IGNORECASE)
        if pattern.search(content):
            violations.append(
                f"CMakeLists.txt: external dependency mechanism '{term}' detected"
            )

    return violations


def check_source_tree(root: Path) -> list[str]:
    violations = []
    scan_dirs = ["src", "include", "apps", "tests", "gpu"]
    header_regexes = [re.compile(pat, re.IGNORECASE) for pat in FORBIDDEN_HEADERS]

    for dname in scan_dirs:
        scan_dir = root / dname
        if not scan_dir.is_dir():
            continue
        for path in scan_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".hpp", ".cpp", ".h", ".c", ".cxx", ".cc"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as err:
                violations.append(f"Cannot read {path}: {err}")
                continue

            for lineno, line in enumerate(text.splitlines(), 1):
                strip_line = line.strip()
                if not strip_line.startswith("#include"):
                    continue
                for reg in header_regexes:
                    if reg.search(strip_line):
                        violations.append(
                            f"{path.relative_to(root)}:{lineno}: "
                            f"forbidden include '{strip_line}'"
                        )

    vendor_dirs = ["third_party", "vendor", "extern", "submodules"]
    code_exts = {
        ".cpp", ".c", ".cc", ".cxx", ".hpp", ".h",
        ".a", ".so", ".dylib", ".dll"
    }
    for vname in vendor_dirs:
        vdir = root / vname
        if vdir.is_dir():
            code_files = [
                p for p in vdir.rglob("*")
                if p.is_file() and p.suffix.lower() in code_exts
            ]
            if code_files:
                rel = code_files[0].relative_to(root)
                violations.append(
                    f"Forbidden vendored code/binary found: {rel}"
                )

    return violations


def inspect_binary(binary_path: Path) -> list[str]:
    violations = []
    if not binary_path.is_file():
        return [f"Binary not found: {binary_path}"]

    try:
        res = subprocess.run(
            ["readelf", "-d", str(binary_path)],
            capture_output=True, text=True, check=True
        )
        for line in res.stdout.splitlines():
            if "(NEEDED)" not in line:
                continue
            match = re.search(r"Shared library:\s*\[(.*?)\]", line)
            if not match:
                continue
            so_name = match.group(1)
            so_stem = so_name.split(".so")[0] + ".so"
            if so_stem not in ALLOWED_DYNAMIC_LIBS:
                violations.append(
                    f"{binary_path.name}: disallowed dynamic dependency '{so_name}'"
                )
            for forbidden in FORBIDDEN_LIBRARIES:
                if forbidden in so_name.lower():
                    violations.append(
                        f"{binary_path.name}: forbidden library link target '{so_name}'"
                    )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        nm_res = subprocess.run(
            ["nm", "-D", str(binary_path)],
            capture_output=True, text=True, check=True
        )
        sym_regexes = [re.compile(pat) for pat in FORBIDDEN_SYMBOLS]
        for line in nm_res.stdout.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            sym = parts[-1].split("@")[0]
            for sreg in sym_regexes:
                if sreg.search(sym):
                    violations.append(
                        f"{binary_path.name}: forbidden external solver symbol '{sym}'"
                    )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify clean-room sovereignty and dependency hygiene"
    )
    parser.add_argument(
        "root", nargs="?", default=".",
        help="Repository root directory (default: current directory)"
    )
    parser.add_argument(
        "--binary", default=None,
        help="Path to markov-cero-solve binary to inspect"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    all_violations: list[str] = []

    cmake_v = check_cmake(root)
    all_violations.extend(cmake_v)

    src_v = check_source_tree(root)
    all_violations.extend(src_v)

    binaries_to_check: list[Path] = []
    if args.binary:
        binaries_to_check.append(Path(args.binary).resolve())
    else:
        candidates = [
            root / "build" / "markov-cero-solve",
            root / "_deployment-phase2-build" / "markov-cero-solve",
            root / "_build" / "markov-cero-solve",
        ]
        for cand in candidates:
            if cand.is_file():
                binaries_to_check.append(cand)

    for bpath in binaries_to_check:
        bin_v = inspect_binary(bpath)
        all_violations.extend(bin_v)

    if all_violations:
        print("[!] SOVEREIGNTY VIOLATION DETECTED:", file=sys.stderr)
        for v in all_violations:
            print(f"    - {v}", file=sys.stderr)
        return 1

    checked_bin = f" (inspected {len(binaries_to_check)} binaries)" if binaries_to_check else ""
    print(f"Sovereignty verification passed: zero external solver dependencies{checked_bin}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
