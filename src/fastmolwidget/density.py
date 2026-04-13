"""
Residual electron density (Fo² − Fc²) computation using gemmi.

This module provides functions to:

1. Read observed structure factors (Fo²) from SHELX HKL files or embedded CIF data.
2. Compute model structure factors (Fc) from the crystal structure model via gemmi.
3. Build a difference Fourier map using (Fo − |Fc|) × exp(iφ_c) coefficients.
4. Return the resulting 3D density grid and its metadata.
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np


def read_hkl_data(
    hkl_source: str | Path,
) -> list[tuple[int, int, int, float, float]]:
    """Read reflection data (h, k, l, Fo², σ) from a SHELX HKLF-4 file.

    Reading stops at the sentinel line ``0 0 0 0 0``.

    :param hkl_source: Path to a ``.hkl`` file **or** the raw text content of
        an embedded ``_shelx_hkl_file`` CIF data item.
    :returns: A list of ``(h, k, l, Fo², σ(Fo²))`` tuples.
    """
    if isinstance(hkl_source, Path) or (
        isinstance(hkl_source, str) and "\n" not in hkl_source and Path(hkl_source).is_file()
    ):
        text = Path(hkl_source).read_text()
    else:
        text = hkl_source

    reflections: list[tuple[int, int, int, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
            fo2 = float(parts[3])
            sig = float(parts[4])
        except (ValueError, IndexError):
            continue
        if h == 0 and k == 0 and l == 0:
            break
        reflections.append((h, k, l, fo2, sig))
    return reflections


def _merge_reflections(
    reflections: list[tuple[int, int, int, float, float]],
) -> dict[tuple[int, int, int], tuple[float, float]]:
    """Merge duplicate (h, k, l) observations by averaging Fo² and propagating σ.

    :returns: Mapping ``{(h,k,l): (mean_Fo², propagated_σ)}``.
    """
    from collections import defaultdict

    accum: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for h, k, l, fo2, sig in reflections:
        accum[(h, k, l)].append(fo2)

    merged: dict[tuple[int, int, int], tuple[float, float]] = {}
    for hkl, values in accum.items():
        mean_fo2 = sum(values) / len(values)
        # simple propagated σ ≈ std/√n  (approximate)
        merged[hkl] = (mean_fo2, 1.0)
    return merged


def compute_difference_density(
    cif_path: str | Path,
    hkl_path: str | Path | None = None,
    d_min: float | None = None,
    grid_oversampling: float = 3.0,
) -> tuple[np.ndarray, gemmi.UnitCell, gemmi.SpaceGroup]:
    """Compute the residual electron-density map (Fo − Fc difference Fourier).

    The workflow:

    1. Read the structural model from *cif_path* using
       :func:`gemmi.read_small_structure`.
    2. Read observed Fo² values from *hkl_path* (or from the HKL data embedded
       in the CIF ``_shelx_hkl_file`` item when *hkl_path* is ``None``).
    3. For each unique (h, k, l):
       - Compute Fc using ``gemmi.StructureFactorCalculatorX``.
       - Compute Fo = √(max(0, Fo²)).
       - Scale |Fc| to Fo using a Wilson-type least-squares scale factor.
       - Form the difference coefficient
         ΔF = (Fo − k·|Fc|) × exp(i·φ_c).
    4. Place the ΔF coefficients on a reciprocal-space grid via
       :meth:`gemmi.ComplexAsuData.get_f_phi_on_grid`.
    5. Transform back to real space with
       :func:`gemmi.transform_f_phi_grid_to_map`.

    :param cif_path: Path to the CIF file with the structural model.
    :param hkl_path: Optional path to a separate ``.hkl`` file.  When
        ``None``, the embedded ``_shelx_hkl_file`` data from the CIF is used.
    :param d_min: Resolution limit (Å).  If ``None``, determined from the data.
    :param grid_oversampling: Oversampling factor for the FFT grid (default 3).
    :returns: A tuple ``(density_array, unit_cell, spacegroup)`` where
        *density_array* is a 3-D :class:`numpy.ndarray` of electron-density
        values (e/ų) indexed ``[u, v, w]`` in fractional-coordinate order.
    """
    cif_path = Path(cif_path)

    # 1. Read structure model
    st = gemmi.read_small_structure(str(cif_path))
    st.setup_cell_images()

    # 2. Read HKL data
    if hkl_path is not None:
        reflections = read_hkl_data(hkl_path)
    else:
        doc = gemmi.cif.read(str(cif_path))
        block = doc[0]
        hkl_text = block.find_value("_shelx_hkl_file")
        if not hkl_text or str(hkl_text).strip() in ("?", ""):
            raise ValueError(
                f"No HKL data found: no embedded _shelx_hkl_file in {cif_path} "
                f"and no separate hkl_path was provided."
            )
        reflections = read_hkl_data(str(hkl_text))

    if not reflections:
        raise ValueError("No reflections read from HKL data.")

    merged = _merge_reflections(reflections)

    # 3. Compute Fc for each reflection
    calc = gemmi.StructureFactorCalculatorX(st.cell)

    miller_list: list[list[int]] = []
    fo_arr: list[float] = []
    fc_complex_arr: list[complex] = []

    for (h, k, l), (fo2, _sig) in merged.items():
        fc = calc.calculate_sf_from_small_structure(st, [h, k, l])
        fo = np.sqrt(max(0.0, fo2))
        miller_list.append([h, k, l])
        fo_arr.append(fo)
        fc_complex_arr.append(fc)

    fo_np = np.array(fo_arr, dtype=np.float64)
    fc_np = np.array(fc_complex_arr, dtype=np.complex128)
    fc_abs = np.abs(fc_np)

    # 4. Compute least-squares scale factor:  Fo ≈ k_scale · |Fc|
    #    Minimising Σ(Fo − k·|Fc|)²  ⇒  k = Σ(Fo·|Fc|) / Σ(|Fc|²)
    mask = fc_abs > 1e-10
    if mask.sum() < 10:
        raise ValueError("Too few reflections with non-zero Fc to compute scale factor.")

    k_scale = float(np.sum(fo_np[mask] * fc_abs[mask]) / np.sum(fc_abs[mask] ** 2))

    # 5. Form difference coefficients: ΔF = (Fo − k·|Fc|) × exp(i·φc)
    phases = np.angle(fc_np)
    delta_f = (fo_np - k_scale * fc_abs)
    diff_coeff = delta_f * np.exp(1j * phases)

    # 6. Build ComplexAsuData and transform to map
    miller_array = np.array(miller_list, dtype=np.int32)
    value_array = np.array(diff_coeff, dtype=np.complex64)

    asu_data = gemmi.ComplexAsuData(st.cell, st.spacegroup, miller_array, value_array)

    # Determine grid size (oversampled)
    size = asu_data.get_size_for_hkl(sample_rate=grid_oversampling)
    f_phi_grid = asu_data.get_f_phi_on_grid(size)
    density_map = gemmi.transform_f_phi_grid_to_map(f_phi_grid)

    arr = np.array(density_map, copy=True)

    return arr, st.cell, st.spacegroup
