"""Type stubs for the optional ``density_cpp`` pybind11 extension.

The compiled module carries no type information of its own, so every call into
it was opaque to type checkers.  This stub describes the two entry points
declared in ``density_cpp/density_cpp.cpp``; keep the two in step whenever a
binding is added or its signature changes.

The module is optional — import it through the ``HAS_DENSITY_CPP`` guard in
:mod:`fastmolwidget.density` rather than directly.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

def marching_cubes(
    grid: NDArray[Any],
    level: float,
    origin: Sequence[float] = ...,
    step: Sequence[float] = ...,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Extract a wireframe isosurface from a regular 3-D scalar density grid.

    :param grid: C-contiguous ``(nx, ny, nz)`` float32 or float64 array.
    :param level: Isosurface level.
    :param origin: Cartesian coordinates of ``grid[0, 0, 0]``; three floats.
    :param step: Cartesian spacing along the three grid axes; three floats.
    :returns: ``(vertices, edges)`` — an ``(M, 3)`` float64 array of Cartesian
        vertices and a ``(K, 2)`` int64 array of unique undirected edges.
    """

def structure_factors(
    hkl: ArrayLike,
    stol2: ArrayLike,
    rotations: ArrayLike,
    translations: ArrayLike,
    fract: ArrayLike,
    occupancies: ArrayLike,
    u_iso: ArrayLike,
    aniso: ArrayLike,
    form_index: ArrayLike,
    form_factors: ArrayLike,
    reciprocal: Sequence[float],
) -> NDArray[np.complex128]:
    """Sum calculated structure factors over a small-molecule model.

    :param hkl: ``(N, 3)`` int32 Miller indices.
    :param stol2: ``(N,)`` float64 array of ``(sin(theta) / lambda)**2``.
    :param rotations: ``(M, 3, 3)`` float64 symmetry rotations, identity
        included.
    :param translations: ``(M, 3)`` float64 symmetry translations.
    :param fract: ``(S, 3)`` float64 fractional coordinates of the sites.
    :param occupancies: ``(S,)`` float64 site occupancies.
    :param u_iso: ``(S,)`` float64 isotropic ADPs.
    :param aniso: ``(S, 6)`` float64 anisotropic ADPs as
        ``U11, U22, U33, U12, U13, U23`` in the small-molecule convention; an
        all-zero row means the site is isotropic.
    :param form_index: ``(S,)`` int32 row of *form_factors* per site.
    :param form_factors: ``(F, N)`` float64 scattering factors, addends
        included.
    :param reciprocal: Reciprocal cell lengths ``a*, b*, c*``.
    :returns: ``(N,)`` complex128 array of calculated structure factors.
    """