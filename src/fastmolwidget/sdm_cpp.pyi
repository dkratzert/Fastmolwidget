"""Type stubs for the optional ``sdm_cpp`` pybind11 extension.

Mirrors the bindings declared in ``sdm_cpp/sdm_cpp.cpp``; keep the two in step
whenever a binding is added or its signature changes.

The module is optional — import it through the ``HAS_CPP`` guard in
:mod:`fastmolwidget.sdm` rather than directly.
"""

from collections.abc import Sequence

#: ``True`` when the extension was compiled with OpenMP support.
has_openmp: bool

def calc_sdm_cpp(
    coords: Sequence[Sequence[float]],
    symm_m: Sequence[Sequence[Sequence[float]]],
    symm_t: Sequence[Sequence[float]],
    aga: float,
    bbe: float,
    cal: float,
    asq: float,
    bsq: float,
    csq: float,
    radii: Sequence[float],
    is_h: Sequence[bool],
    parts: Sequence[float],
) -> list[tuple[int, int, int, float, float, bool]]:
    """Calculate the Shortest Distance Matrix for all atom pairs.

    :param coords: ``N`` fractional ``[x, y, z]`` coordinates.
    :param symm_m: ``S`` 3x3 symmetry rotation matrices.
    :param symm_t: ``S`` 3-element translation vectors.
    :param aga: ``a * b * cos(gamma)``.
    :param bbe: ``a * c * cos(beta)``.
    :param cal: ``b * c * cos(alpha)``.
    :param asq: ``a**2``.
    :param bsq: ``b**2``.
    :param csq: ``c**2``.
    :param radii: Covalent radius of every atom, in Angstrom.
    :param is_h: Whether each atom is hydrogen or deuterium.
    :param parts: SHELX disorder-part number of every atom.
    :returns: One ``(i, j, best_n, mind, dddd, covalent)`` tuple per pair —
        the two atom indices, the symmetry operation that gave the shortest
        distance, that distance, the covalent-bond cutoff (``0.0`` when the
        pair cannot bond) and whether the pair is bonded.
    """
