"""
Minimal setup.py — only responsible for the sdm_cpp C++ extension.
All project metadata lives in pyproject.toml.

OpenMP detection
----------------
macOS  : looks for libomp via `brew --prefix libomp`; silently omits OpenMP
         flags when libomp is not found (build still succeeds, single-threaded).
         Install libomp optionally with:  brew install libomp
Linux  : uses -fopenmp (GCC/Clang link flag).
Windows: uses /openmp (MSVC).
"""
from __future__ import annotations

import subprocess
import sys
from setuptools import setup, Extension

try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError as exc:
    raise RuntimeError(
        "pybind11 is required to build sdm_cpp. "
        "Install it with:  pip install pybind11"
    ) from exc


# ---------------------------------------------------------------------------
# OpenMP flag detection
# ---------------------------------------------------------------------------

def _find_openmp() -> tuple[list[str], list[str]]:
    """Return (extra_compile_args, extra_link_args) for OpenMP, or ([], [])."""

    if sys.platform == "darwin":
        import os
        # universal2 / cross-compile: ARCHFLAGS contains x86_64, but Homebrew on
        # Apple Silicon only provides arm64 libomp → skip OpenMP to avoid a
        # broken binary.  The build succeeds in single-threaded mode.
        archflags = os.environ.get("ARCHFLAGS", "")
        if "x86_64" in archflags:
            print("[sdm_cpp] Cross-compilation/universal2 detected (ARCHFLAGS contains x86_64) — skipping OpenMP.")
            return [], []

        try:
            result = subprocess.run(
                ["brew", "--prefix", "libomp"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                prefix = result.stdout.strip()
                omp_header = os.path.join(prefix, "include", "omp.h")
                omp_lib = os.path.join(prefix, "lib", "libomp.dylib")
                if os.path.isfile(omp_header) and os.path.isfile(omp_lib):
                    compile_flags = [
                        "-Xpreprocessor", "-fopenmp",
                        f"-I{prefix}/include",
                    ]
                    link_flags = [
                        f"-L{prefix}/lib",
                        "-lomp",
                    ]
                    print(f"[sdm_cpp] OpenMP found via Homebrew libomp: {prefix}")
                    return compile_flags, link_flags
                else:
                    print(f"[sdm_cpp] brew prefix {prefix!r} exists but omp.h / libomp.dylib missing — skipping OpenMP.")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        print("[sdm_cpp] libomp not found — building without OpenMP (single-threaded).")
        return [], []

    if sys.platform.startswith("linux"):
        return ["-fopenmp"], ["-fopenmp"]

    if sys.platform == "win32":
        return ["/openmp"], []

    return [], []


omp_compile, omp_link = _find_openmp()

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    base_compile = ["/O2", "/std:c++17"] + omp_compile
else:
    base_compile = ["-O3", "-std=c++17"] + omp_compile

# ---------------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------------

sdm_cpp_ext = Extension(
    name="sdm_cpp",
    sources=["src/sdm_cpp/sdm_cpp.cpp"],
    include_dirs=[pybind11_include],
    extra_compile_args=base_compile,
    extra_link_args=omp_link,
    language="c++",
)

setup(ext_modules=[sdm_cpp_ext])

