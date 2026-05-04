from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

from fastmolwidget.atoms import get_radius_from_element


def to_float(st: str) -> float | None:
    try:
        return float(st.split('(')[0])
    except ValueError:
        return None


def to_float_list(st: Sequence[str]) -> list[float] | None:
    try:
        return [float(x) for x in st[-2:]]
    except ValueError:
        return None


def get_error_from_value(value: str) -> tuple[float, float]:
    """
    Returns the error value from a number string.
    """
    try:
        value = value.replace(" ", "")
    except AttributeError:
        return float(value), 0.0
    if "(" in value:
        vval, err = value.split("(")
        val = vval.split('.')
        err = err.split(")")[0]
        if not err:
            return float(vval), 0.0
        if len(val) > 1:
            return float(vval), int(err) * (10 ** (-1 * len(val[1])))
        else:
            return float(vval), float(err)
    else:
        try:
            return float(value), 0.0
        except ValueError:
            return 0.0, 0.0


def isnumeric(value: str) -> bool:
    """
    Determines if a string can be converted to a number.
    """
    value = value.split('(')[0]
    try:
        float(value)
    except ValueError:
        return False
    return True


def grouper(inputs, n, fillvalue=None):
    iters = [iter(inputs)] * n
    return itertools.zip_longest(*iters, fillvalue=fillvalue)


def build_conntable(
        coords: np.ndarray,
        types: list[str],
        parts: list[int],
        radii: np.ndarray | None = None,
        extra_param: float = 1.2,
        symmgen: list[bool] | np.ndarray | None = None,
) -> tuple[tuple[int, int], ...]:
    """Vectorised connectivity-table builder (shared by 2D and 3D widgets).

    Parameters
    ----------
    coords : ndarray of shape (N, 3)
        Cartesian atom positions.
    types : list[str]
        Element symbols (length N).
    parts : list[int]
        SHELX disorder-part numbers (length N).
    radii : ndarray of shape (N,), optional
        Pre-computed covalent radii.  If *None*, looked up from *types*.
    extra_param : float
        Multiplier applied to the sum of covalent radii for bond detection.
    symmgen : list[bool] or ndarray of shape (N,), optional
        Per-atom flag indicating whether the atom was symmetry-generated.
        Required for the negative-PART exclusion rule.

    Returns
    -------
    tuple of (i, j) pairs with i < j.
    """
    n = len(coords)
    if n == 0:
        return ()

    coords = np.asarray(coords, dtype=np.float64)

    # ── pairwise distance matrix ─────────────────────────────────────────
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 3)
    dists = np.linalg.norm(diff, axis=2)  # (N, N)

    # ── per-pair bond-distance thresholds ────────────────────────────────
    if radii is None:
        radii = np.array(
            [get_radius_from_element(t) for t in types], dtype=np.float64
        )
    else:
        radii = np.asarray(radii, dtype=np.float64)

    radii_sum = (radii[:, None] + radii[None, :]) * extra_param  # (N, N)

    # ── combined Boolean mask ────────────────────────────────────────────
    # Upper triangle (i < j), non-trivial distance, within 4 Å pre-filter
    triu = np.triu(np.ones((n, n), dtype=bool), k=1)
    bond_mask = triu & (dists > 0.01) & (dists <= 4.0) & (dists < radii_sum)

    if not np.any(bond_mask):
        return ()

    # Part filter: forbidden when both parts are non-zero and differ
    parts_arr = np.array(parts, dtype=np.int32)
    bond_mask &= ~(
            (parts_arr[:, None] != 0)
            & (parts_arr[None, :] != 0)
            & (parts_arr[:, None] != parts_arr[None, :])
    )

    # Negative-part filter: if an atom has a negative part number, bonds
    # crossing the asymmetric-unit / symmetry-copy boundary are excluded.
    # This prevents disordered fragments on special positions from bonding
    # to their own symmetry images while preserving intra-copy connectivity.
    if symmgen is not None:
        symmgen_arr = np.asarray(symmgen, dtype=bool)
        neg_part = parts_arr < 0
        # True when the two atoms are on different sides of the boundary
        cross_boundary = symmgen_arr[:, None] != symmgen_arr[None, :]
        # Exclude when at least one atom has negative part and they cross
        either_neg = neg_part[:, None] | neg_part[None, :]
        bond_mask &= ~(either_neg & cross_boundary)

    # H–H filter: skip bonds between two hydrogen atoms
    is_h = np.array([t in ("H", "D") for t in types], dtype=bool)
    bond_mask &= ~(is_h[:, None] & is_h[None, :])

    rows, cols = np.where(bond_mask)
    return tuple(zip(rows.tolist(), cols.tolist()))
