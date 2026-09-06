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
    """Return ``(value, esd)`` from a CIF-style numeric string."""
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
    """True if a CIF-style numeric string can be converted to ``float``."""
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
    """Build the bond table for Cartesian coordinates.

    Bonds are limited by distance, covalent radii, PART rules, the negative-
    PART symmetry-copy rule, and an H-H exclusion.
    """
    n = len(coords)
    if n == 0:
        return ()

    coords = np.asarray(coords, dtype=np.float64)

    # Pairwise distance matrix.
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 3)
    dists = np.linalg.norm(diff, axis=2)  # (N, N)

    # Per-pair bond-distance thresholds.
    if radii is None:
        radii = np.array(
            [get_radius_from_element(t) for t in types], dtype=np.float64
        )
    else:
        radii = np.asarray(radii, dtype=np.float64)

    radii_sum = (radii[:, None] + radii[None, :]) * extra_param  # (N, N)

    # Upper triangle, non-trivial distance, and 4 Å pre-filter.
    triu = np.triu(np.ones((n, n), dtype=bool), k=1)
    bond_mask = triu & (dists > 0.01) & (dists <= 4.0) & (dists < radii_sum)

    if not np.any(bond_mask):
        return ()

    # Different non-zero PART values cannot bond.
    parts_arr = np.array(parts, dtype=np.int32)
    bond_mask &= ~(
            (parts_arr[:, None] != 0)
            & (parts_arr[None, :] != 0)
            & (parts_arr[:, None] != parts_arr[None, :])
    )

    # Negative PART forbids bonds across the asymmetric-unit/symmetry-copy
    # boundary, preventing self-bonding on special positions.
    if symmgen is not None:
        symmgen_arr = np.asarray(symmgen, dtype=bool)
        neg_part = parts_arr < 0
        # True when atoms lie on opposite sides of that boundary.
        cross_boundary = symmgen_arr[:, None] != symmgen_arr[None, :]
        # Exclude cross-boundary bonds involving a negative PART.
        either_neg = neg_part[:, None] | neg_part[None, :]
        bond_mask &= ~(either_neg & cross_boundary)

    # Skip H-H bonds.
    is_h = np.array([t in ("H", "D") for t in types], dtype=bool)
    bond_mask &= ~(is_h[:, None] & is_h[None, :])

    rows, cols = np.where(bond_mask)
    return tuple(zip(rows.tolist(), cols.tolist()))
