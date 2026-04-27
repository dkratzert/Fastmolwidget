"""
Headless profiling of the 'grow' path for 1548072_many_atoms.cif.
No QApplication / OpenGL needed – only CifReader + SDM are measured.

Run:
    uv run python profile_grow.py
    uv run python -m pstats grow.prof   (interactive)
"""

from __future__ import annotations

import cProfile
import pstats
import io
import time
from pathlib import Path

# ── locate test data relative to this script ──────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
CIF_PATH  = REPO_ROOT / "tests" / "test-data" / "1548072_many_atoms.cif"


def run_grow() -> None:
    """Full grow cycle: parse CIF → build SDM → pack atoms."""
    from fastmolwidget.cif.cif_file_io import CifReader
    from fastmolwidget.sdm import SDM, HAS_CPP

    t0 = time.perf_counter()
    cif = CifReader(CIF_PATH)
    t1 = time.perf_counter()
    print(f"  cif read:          {t1-t0:.4f} s")

    print(f"sdm_cpp C++ extension loaded: {HAS_CPP}")

    # ── 1. CIF read + atoms_fract ──────────────────────────────────────────────
    t0 = time.perf_counter()
    fract_atoms = list(cif.atoms_fract)
    t1 = time.perf_counter()
    print(f"  atoms_fract ({len(fract_atoms)} atoms):          {t1-t0:.4f} s")

    # ── 2. SDM construction ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    sdm = SDM(fract_atoms, cif.symmops, cif.cell, centric=cif.is_centrosymm)
    t1 = time.perf_counter()
    print(f"  SDM.__init__:                          {t1-t0:.4f} s")

    # ── 3. calc_sdm (the O(N²·S) kernel) ──────────────────────────────────────
    t0 = time.perf_counter()
    need_symm = sdm.calc_sdm()
    t1 = time.perf_counter()
    print(f"  calc_sdm (incl molindex+collect_symm): {t1-t0:.4f} s  →  {len(need_symm)} needed symm entries")

    # ── 4. packer ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    grown_atoms = sdm.packer(sdm, need_symm)
    t1 = time.perf_counter()
    print(f"  packer:                                {t1-t0:.4f} s  →  {len(grown_atoms)} Cartesian atoms")


# ── cProfile pass ─────────────────────────────────────────────────────────────
PROF_FILE = REPO_ROOT / "grow.prof"

print("=" * 60)
print(f"Profiling grow on: {CIF_PATH.name}")
print("=" * 60)

profiler = cProfile.Profile()
profiler.enable()
run_grow()
profiler.disable()

profiler.dump_stats(str(PROF_FILE))
print(f"\nProfile saved → {PROF_FILE}")

# ── pretty-print top 30 by cumulative time ─────────────────────────────────────
print("\n── Top 30 functions by cumulative time ──────────────────────")
buf = io.StringIO()
stats = pstats.Stats(profiler, stream=buf)
stats.strip_dirs()
stats.sort_stats("cumulative")
stats.print_stats(30)
print(buf.getvalue())

# ── pretty-print top 30 by total (self) time ──────────────────────────────────
print("── Top 30 functions by self (tottime) ───────────────────────")
buf2 = io.StringIO()
stats2 = pstats.Stats(profiler, stream=buf2)
stats2.strip_dirs()
stats2.sort_stats("tottime")
stats2.print_stats(30)
print(buf2.getvalue())

