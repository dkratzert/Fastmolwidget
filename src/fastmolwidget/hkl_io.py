"""
Readers for diffraction reflection data and SHELX refinement parameters.

Supported inputs
----------------
* **SHELX ``.hkl``** – the classic fixed-format ``HKLF 4`` file
  (``3I4,2F8.2``: *h k l* F² σ(F²), optionally followed by a batch number),
  terminated by a ``0 0 0`` record.  See :func:`read_shelx_hkl`.
* **fcf-style CIF reflection loops** – ``_refln_index_h/k/l`` together with
  ``_refln_F_squared_meas`` (or ``_refln_F_meas``) and, when present,
  ``_refln_F_squared_calc`` / ``_refln_F_calc``.  See :func:`read_cif_reflections`.
* **CIF-embedded SHELX reflections** – self-contained CIFs written by SHELXL
  keep the whole ``.hkl`` in ``_shelx_hkl_file``, so the CIF alone is enough.
  See :func:`embedded_shelx_hkl`.
* **SHELX refinement parameters** – the refined overall scale factor (OSF,
  i.e. the first ``FVAR``), the ``WGHT`` weighting scheme and the ``EXTI``
  extinction coefficient.  These are read from a standalone ``.res``/``.ins``
  file *or* from a SHELX ``.res`` block embedded inside a CIF
  (``_shelx_res_file`` / ``_iucr_refine_instructions_details``).
  See :func:`read_shelx_parameters`.

A leading ``global_`` block is ignored everywhere — it carries inherited
values, not a structure of its own.

Everything in this module is Qt-free and only depends on ``gemmi``,
``numpy`` and ``shelxfile``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import gemmi
import numpy as np

__all__ = [
    'ReflectionData',
    'ShelxParameters',
    'embedded_shelx_hkl',
    'embedded_shelx_res',
    'find_reflection_file',
    'has_reflections',
    'read_cif_reflections',
    'read_reflections',
    'read_shelx_hkl',
    'read_shelx_parameters',
]


def _data_blocks(doc) -> list:
    """Return the real data blocks of *doc*, skipping any ``global_`` block.

    A ``global_`` block holds values inherited by the blocks that follow it;
    it is not a structure of its own, so every reader here ignores it.
    ``gemmi`` represents it as a block whose name is empty or ``'global'``.
    """
    return [block for block in doc
            if block.name and block.name.lower() != 'global']


@dataclass
class ReflectionData:
    """A set of measured (and optionally calculated) structure factors.

    :param hkl: ``(N, 3)`` integer array of Miller indices.
    :param f_sq_meas: ``(N,)`` array of measured F² values, on the arbitrary
        scale of the raw data.
    :param sigma: ``(N,)`` array of σ(F²).
    :param f_calc: Optional ``(N,)`` complex array of calculated structure
        factors, present only when the source file already contained them
        (an fcf-style CIF).  ``None`` means *compute them yourself*.
    """

    hkl: np.ndarray
    f_sq_meas: np.ndarray
    sigma: np.ndarray
    f_calc: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.hkl)

    @property
    def has_f_calc(self) -> bool:
        """``True`` when calculated structure factors came with the data."""
        return self.f_calc is not None


@dataclass
class ShelxParameters:
    """Refined parameters taken from a SHELX ``.res``/``.ins`` model.

    :param osf: Refined overall scale factor (the first ``FVAR``).  SHELX
        scales such that ``|Fo| ≈ osf · |Fc|``.
    :param wght_a: ``a`` of the ``WGHT`` weighting scheme.
    :param wght_b: ``b`` of the ``WGHT`` weighting scheme.
    :param exti: ``EXTI`` extinction coefficient (``0.0`` when not refined).
    :param wavelength: Radiation wavelength in Å, from the ``CELL`` card.
    :param free_variables: The full ``FVAR`` list (``free_variables[0]`` is
        the OSF), used to decode SHELX occupancy codes.
    """

    osf: float = 1.0
    wght_a: float = 0.1
    wght_b: float = 0.0
    exti: float = 0.0
    wavelength: float = 0.71073
    free_variables: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SHELX .hkl
# ---------------------------------------------------------------------------

def read_shelx_hkl(path: str | Path) -> ReflectionData:
    """Read a SHELX ``HKLF 4`` reflection file.

    The format is column-based (``3I4,2F8.2`` plus an optional batch number).
    Free-format files (whitespace-separated) are also accepted as a fallback,
    because many programs write them that way.

    Reading stops at the terminating ``0 0 0`` record, as SHELX does.

    :param path: Path to the ``.hkl`` file.
    :returns: The measured reflections; ``f_calc`` is always ``None``.
    :raises ValueError: If no usable reflection could be parsed.
    """
    return parse_shelx_hkl(Path(path).read_text(errors='replace'), source=path)


def parse_shelx_hkl(text: str, *, source: str | Path = '<text>') -> ReflectionData:
    """Parse SHELX ``HKLF 4`` reflection *text*.

    Used both for standalone ``.hkl`` files and for the ``_shelx_hkl_file``
    block embedded in a CIF.

    :param text: The reflection records.
    :param source: Only used in the error message.
    :raises ValueError: If no usable reflection could be parsed.
    """
    hkl: list[tuple[int, int, int]] = []
    f_sq: list[float] = []
    sig: list[float] = []

    for raw in text.splitlines():
        line = raw.rstrip('\r')
        if not line.strip():
            continue
        parsed = _parse_hkl_line(line)
        if parsed is None:
            continue
        h, k, l, fsq, s = parsed
        if h == 0 and k == 0 and l == 0:
            break
        hkl.append((h, k, l))
        f_sq.append(fsq)
        sig.append(s)

    if not hkl:
        raise ValueError(f'No reflections found in {source}')

    return ReflectionData(
        hkl=np.array(hkl, dtype=np.int32),
        f_sq_meas=np.array(f_sq, dtype=float),
        sigma=np.array(sig, dtype=float),
    )


def _parse_hkl_line(line: str) -> tuple[int, int, int, float, float] | None:
    """Parse one SHELX ``.hkl`` record, fixed-format first, free-format after.

    :returns: ``(h, k, l, F², σ)`` or ``None`` when the line is not a
        reflection record.
    """
    if len(line) >= 28:
        try:
            return (
                int(line[0:4]), int(line[4:8]), int(line[8:12]),
                float(line[12:20]), float(line[20:28]),
            )
        except ValueError:
            pass
    fields = line.split()
    if len(fields) >= 5:
        try:
            return (
                int(fields[0]), int(fields[1]), int(fields[2]),
                float(fields[3]), float(fields[4]),
            )
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# CIF reflection loops (fcf style)
# ---------------------------------------------------------------------------

def read_cif_reflections(path: str | Path) -> ReflectionData | None:
    """Read an fcf-style reflection loop from a CIF.

    Recognises ``_refln_index_h/k/l`` together with either
    ``_refln_F_squared_meas`` or ``_refln_F_meas``.  When the loop also
    provides calculated structure factors (``_refln_F_squared_calc`` /
    ``_refln_F_calc``, optionally with ``_refln_phase_calc``) these are
    returned as well, so they do not have to be recomputed.

    :param path: Path to the CIF (or ``.fcf``) file.
    :returns: The reflections, or ``None`` when the file contains no
        reflection loop at all.
    """
    doc = gemmi.cif.read(str(path))
    for block in _data_blocks(doc):
        h_col = block.find_values('_refln_index_h')
        if not h_col:
            continue
        k_col = block.find_values('_refln_index_k')
        l_col = block.find_values('_refln_index_l')
        if not k_col or not l_col:
            continue

        meas_sq = block.find_values('_refln_F_squared_meas')
        meas_f = block.find_values('_refln_F_meas')
        if meas_sq:
            f_sq = np.array([_num(v) for v in meas_sq], dtype=float)
            sig_col = block.find_values('_refln_F_squared_sigma')
        elif meas_f:
            f_amp = np.array([_num(v) for v in meas_f], dtype=float)
            f_sq = f_amp ** 2
            sig_col = block.find_values('_refln_F_sigma')
        else:
            continue

        if sig_col:
            sigma = np.array([_num(v) for v in sig_col], dtype=float)
        else:
            sigma = np.ones_like(f_sq)

        hkl = np.array(
            [[int(_num(a)), int(_num(b)), int(_num(c))]
             for a, b, c in zip(h_col, k_col, l_col)],
            dtype=np.int32,
        )

        f_calc = _cif_f_calc(block, len(hkl))
        return ReflectionData(hkl=hkl, f_sq_meas=f_sq, sigma=sigma,
                              f_calc=f_calc)
    return None


def _cif_f_calc(block, n: int) -> np.ndarray | None:
    """Extract calculated structure factors from a CIF reflection loop.

    Amplitudes come from ``_refln_F_calc`` or ``sqrt(_refln_F_squared_calc)``
    and are combined with ``_refln_phase_calc`` (degrees) when available.
    Without phases the values are returned as real numbers, which is only
    useful for centrosymmetric structures — callers should check
    :attr:`ReflectionData.has_f_calc` together with their own needs.

    :returns: ``(n,)`` complex array, or ``None`` when no calculated
        amplitudes are present.
    """
    calc_f = block.find_values('_refln_F_calc')
    calc_sq = block.find_values('_refln_F_squared_calc')
    if calc_f:
        amp = np.array([_num(v) for v in calc_f], dtype=float)
    elif calc_sq:
        amp = np.sqrt(np.clip([_num(v) for v in calc_sq], 0.0, None))
    else:
        return None
    if len(amp) != n:
        return None

    phase = block.find_values('_refln_phase_calc')
    if phase and len(phase) == n:
        ang = np.radians(np.array([_num(v) for v in phase], dtype=float))
        return amp * np.exp(1j * ang)
    return amp.astype(complex)


def _num(value: str) -> float:
    """Convert a CIF numeric string to ``float``, ignoring an esd in braces.

    ``'?'`` and ``'.'`` (CIF null values) become ``0.0``.
    """
    text = str(value).strip().strip("'\"")
    if text in ('?', '.', ''):
        return 0.0
    text = text.split('(')[0]
    try:
        return float(text)
    except ValueError:
        return 0.0


def embedded_shelx_hkl(path: str | Path) -> str | None:
    """Return the SHELX ``.hkl`` text embedded in a CIF, if there is one.

    Self-contained CIFs written by SHELXL carry the complete reflection file
    in ``_shelx_hkl_file``, which makes the CIF alone sufficient to compute a
    residual-density map.

    :param path: Path to the CIF file.
    :returns: The embedded reflection records, or ``None`` when absent.
    """
    try:
        doc = gemmi.cif.read(str(path))
    except Exception:  # noqa: BLE001 - a non-CIF file simply has no HKL block
        return None
    for block in _data_blocks(doc):
        value = block.find_value('_shelx_hkl_file')
        if value:
            text = gemmi.cif.as_string(value)
            if text and text.strip():
                return text
    return None


def read_reflections(path: str | Path) -> ReflectionData:
    """Read reflections from a ``.hkl`` file or a CIF.

    The format is chosen from the file suffix: ``.hkl`` uses
    :func:`read_shelx_hkl`.  For anything else the file is treated as a CIF and
    tried in this order — an fcf-style ``_refln_*`` loop, then a ``.hkl``
    embedded in ``_shelx_hkl_file`` — before falling back to the SHELX reader.

    :param path: Path to the reflection file.
    :raises ValueError: If no reflections could be read.
    """
    path = Path(path)
    if path.suffix.lower() == '.hkl':
        return read_shelx_hkl(path)

    try:
        data = read_cif_reflections(path)
    except Exception:  # noqa: BLE001 - any parse failure just means "not a CIF"
        data = None
    if data is not None:
        return data

    text = embedded_shelx_hkl(path)
    if text is not None:
        return parse_shelx_hkl(text, source=f'{path} (_shelx_hkl_file)')

    return read_shelx_hkl(path)


#: Sibling extensions searched for reflection data, in order of preference.
_REFLECTION_SUFFIXES: tuple[str, ...] = ('.hkl', '.fcf', '.fco', '.cif')


def has_reflections(path: str | Path) -> bool:
    """Cheaply test whether *path* holds usable reflection data.

    Avoids parsing the whole file: a ``.hkl`` only has to exist, and a CIF is
    scanned for an fcf-style loop or an embedded ``_shelx_hkl_file``.

    :param path: The file to inspect.
    """
    path = Path(path)
    if not path.is_file():
        return False
    if path.suffix.lower() == '.hkl':
        return True
    try:
        doc = gemmi.cif.read(str(path))
    except Exception:  # noqa: BLE001 - unreadable or not a CIF
        return False
    for block in _data_blocks(doc):
        if block.find_values('_refln_index_h'):
            return True
        if block.find_value('_shelx_hkl_file'):
            return True
    return False


def find_reflection_file(model_path: str | Path) -> Path | None:
    """Locate the reflection data belonging to the model at *model_path*.

    Searched in order:

    1. the model file itself — self-contained SHELXL CIFs carry the whole
       ``.hkl`` in ``_shelx_hkl_file``, and fcf-style files carry a
       ``_refln_*`` loop;
    2. files of the same basename with a ``.hkl``, ``.fcf``, ``.fco`` or
       ``.cif`` extension.

    :param model_path: Path to the structure file.
    :returns: The reflection file, or ``None`` when nothing was found — the
        caller should then ask the user.
    """
    path = Path(model_path)
    if path.suffix.lower() != '.hkl' and has_reflections(path):
        return path
    for suffix in _REFLECTION_SUFFIXES:
        candidate = path.with_suffix(suffix)
        if candidate != path and has_reflections(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# SHELX refinement parameters (standalone .res or embedded in a CIF)
# ---------------------------------------------------------------------------

#: CIF tags that are known to carry a complete SHELX ``.res``/``.ins`` file.
_EMBEDDED_RES_TAGS: tuple[str, ...] = (
    '_shelx_res_file',
    '_iucr_refine_instructions_details',
    '_shelxl_version_number_res_file',
)


def embedded_shelx_res(path: str | Path) -> str | None:
    """Return the SHELX ``.res`` text embedded in a CIF, if there is one.

    Deposited CIFs frequently carry the complete final ``.res`` file in a
    semicolon-delimited text field, which lets us recover the exact refined
    ``FVAR`` / ``WGHT`` / ``EXTI`` values even when no separate ``.res`` file
    is available.

    :param path: Path to the CIF file.
    :returns: The embedded SHELX text, or ``None`` when absent.
    """
    try:
        doc = gemmi.cif.read(str(path))
    except Exception:  # noqa: BLE001 - a non-CIF file simply has no SHELX block
        return None
    for block in _data_blocks(doc):
        for tag in _EMBEDDED_RES_TAGS:
            value = block.find_value(tag)
            if not value:
                continue
            text = gemmi.cif.as_string(value)
            if text and re.search(r'^\s*FVAR', text, re.MULTILINE | re.IGNORECASE):
                return text
    return None


def read_shelx_parameters(path: str | Path) -> ShelxParameters | None:
    """Read refined SHELX parameters for the model at *path*.

    Sources are tried in this order:

    1. *path* itself when it is a ``.res``/``.ins`` file;
    2. a ``.res`` (then ``.ins``) file sitting next to a CIF with the same
       basename;
    3. a SHELX ``.res`` block embedded inside the CIF
       (see :func:`embedded_shelx_res`).

    :param path: Path to the model file (``.res``, ``.ins`` or ``.cif``).
    :returns: The refined parameters, or ``None`` when none could be found.
    """
    path = Path(path)
    if path.suffix.lower() in ('.res', '.ins'):
        return _parse_shelx_text(path.read_text(errors='replace'))

    for suffix in ('.res', '.ins'):
        sibling = path.with_suffix(suffix)
        if sibling.exists():
            return _parse_shelx_text(sibling.read_text(errors='replace'))

    text = embedded_shelx_res(path)
    if text is not None:
        return _parse_shelx_text(text)
    return None


def _parse_shelx_text(text: str) -> ShelxParameters:
    """Extract ``CELL``, ``FVAR``, ``WGHT`` and ``EXTI`` from SHELX text.

    Only the instruction cards are read, so this works equally well on a
    standalone ``.res`` file and on a ``.res`` block embedded in a CIF.
    Continuation lines (``=``) are irrelevant for these cards and ignored.

    :param text: The SHELX instruction text.
    :returns: The parsed parameters, with sensible SHELX defaults for any
        card that is missing.
    """
    params = ShelxParameters()
    fvars: list[float] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith('CELL'):
            values = _floats(line[4:])
            if values:
                params.wavelength = values[0]
        elif upper.startswith('FVAR'):
            fvars.extend(_floats(line[4:]))
        elif upper.startswith('WGHT'):
            values = _floats(line[4:])
            if values:
                params.wght_a = values[0]
            if len(values) > 1:
                params.wght_b = values[1]
        elif upper.startswith('EXTI'):
            values = _floats(line[4:])
            if values:
                params.exti = values[0]
        elif upper.startswith('HKLF'):
            break  # atom list and instructions are finished

    if fvars:
        params.free_variables = fvars
        params.osf = fvars[0]
    return params


def _floats(text: str) -> list[float]:
    """Return every whitespace-separated token in *text* that is a number."""
    out: list[float] = []
    for token in text.replace('=', ' ').split():
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out
