"""Reflection readers and SHELX refinement-parameter readers.

Supports SHELX ``.hkl`` files, fcf-style ``_refln_*`` CIF loops, embedded
``_shelx_hkl_file`` data, and embedded or standalone SHELX ``.res``/``.ins``
instructions. Leading ``global_`` blocks are ignored. All CIF readers accept a
path, :class:`gemmi.cif.Document`, or :class:`gemmi.cif.Block`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import gemmi
import numpy as np

__all__ = [
    'CifSource',
    'ReflectionData',
    'ReflectionSource',
    'ShelxParameters',
    'block_has_reflections',
    'block_shelx_hkl',
    'block_shelx_parameters',
    'block_shelx_res',
    'clear_cif_cache',
    'embedded_shelx_hkl',
    'embedded_shelx_res',
    'find_reflection_file',
    'has_reflections',
    'read_block_raw_reflections',
    'read_block_reflections',
    'read_cif_document',
    'read_cif_raw_reflections',
    'read_cif_reflections',
    'read_reflections',
    'read_shelx_hkl',
    'read_shelx_parameters',
]

#: CIF source accepted by the readers in this module.
CifSource = str | Path | gemmi.cif.Document | gemmi.cif.Block


@lru_cache(maxsize=2)
def _parse_cif_document(path: str, fingerprint: tuple[int, int]):
    """Parse a CIF, keyed on path plus ``(mtime_ns, size)``."""
    return gemmi.cif.read(path)


def read_cif_document(path: str | Path):
    """Read a CIF with a tiny shared cache.

    Cached by path, mtime and size. Treat the returned document as read-only.
    """
    name = str(path)
    try:
        stat = Path(name).stat()
    except OSError:  # not a real file - let gemmi produce the error
        return gemmi.cif.read(name)
    return _parse_cif_document(name, (stat.st_mtime_ns, stat.st_size))


def clear_cif_cache() -> None:
    """Forget every CIF document cached by :func:`read_cif_document`."""
    _parse_cif_document.cache_clear()


def _data_blocks(doc) -> list:
    """Return the real data blocks of *doc*, skipping any ``global_`` block."""
    return [block for block in doc
            if block.name and block.name.lower() != 'global']


def _cif_blocks(source: CifSource) -> list:
    """Return the data blocks of *source*.

    Paths are parsed through the document cache. Parse failures return ``[]``.
    """
    if isinstance(source, gemmi.cif.Block):
        return [source]
    if isinstance(source, gemmi.cif.Document):
        return _data_blocks(source)
    try:
        return _data_blocks(read_cif_document(source))
    except Exception:  # noqa: BLE001 - unreadable or not a CIF at all
        return []


def _is_cif_object(source: object) -> bool:
    """True when *source* is an in-memory CIF document or block."""
    return isinstance(source, (gemmi.cif.Document, gemmi.cif.Block))


@dataclass
class ReflectionData:
    """Measured, and optionally calculated, structure factors.

    ``batch`` is ``HKLF 5`` component/overlap data or ``HKLF 4`` batch data;
    negative values mean different things in those formats. ``sigma_known`` is
    ``False`` when σ is only a placeholder and must not be used for
    significance-based weighting.
    """

    hkl: np.ndarray
    f_sq_meas: np.ndarray
    sigma: np.ndarray
    f_calc: np.ndarray | None = None
    batch: np.ndarray | None = None
    sigma_known: bool = True

    def __len__(self) -> int:
        return len(self.hkl)

    @property
    def has_f_calc(self) -> bool:
        """``True`` when calculated structure factors came with the data."""
        return self.f_calc is not None

    @property
    def has_overlap_groups(self) -> bool:
        """``True`` when some batch numbers are negative.

        Only meaningful for ``HKLF 5``; in ``HKLF 4`` a negative batch may mark
        an *R*\\ :sub:`free` reflection instead.
        """
        return self.batch is not None and bool(np.any(self.batch < 0))


#: Reflection source: CIF source, ``.hkl`` path, or ready data.
ReflectionSource = CifSource | ReflectionData


#: Default ``TWIN`` matrix: inversion, i.e. racemic twinning.
_DEFAULT_TWIN_MATRIX: tuple[float, ...] = (-1.0, 0.0, 0.0,
                                           0.0, -1.0, 0.0,
                                           0.0, 0.0, -1.0)


@dataclass
class ShelxParameters:
    """Refined parameters from a SHELX ``.res``/``.ins`` model.

    Includes OSF, ``WGHT``, ``EXTI``, ``FVAR``, ``TWIN``/``BASF`` and the full
    ``HKLF N S r11…r33 sm`` transform/scales. A negative ``TWIN`` count means
    general and racemic twinning together; components ``m+1…2m`` are then the
    Friedel opposites of ``1…m``.
    """

    osf: float = 1.0
    wght_a: float = 0.1
    wght_b: float = 0.0
    exti: float = 0.0
    wavelength: float = 0.71073
    free_variables: list[float] = field(default_factory=list)
    twin_matrix: tuple[float, ...] | None = None
    twin_components: int = 2
    twin_racemic: bool = False
    basf: list[float] = field(default_factory=list)
    hklf: int = 4
    hklf_scale: float = 1.0
    hklf_matrix: tuple[float, ...] | None = None
    hklf_sigma_scale: float = 1.0

    @property
    def is_twinned(self) -> bool:
        """``True`` when the refinement used a twin law or ``HKLF 5`` data."""
        return self.twin_matrix is not None or self.hklf == 5

    def twin_fractions(self) -> list[float]:
        """Return the volume fraction of each twin component.

        ``BASF`` gives components ``2…n``; component 1 takes the remainder. An
        absent ``BASF`` means equal fractions. Values are clamped, and an
        inconsistent sum falls back to equal fractions.
        """
        count = max(self.twin_components, len(self.basf) + 1)
        if not self.basf:
            return [1.0 / count] * count
        others = [min(max(value, 0.0), 1.0) for value in self.basf[:count - 1]]
        others += [0.0] * (count - 1 - len(others))
        first = 1.0 - sum(others)
        if first < 0.0:  # inconsistent BASF - fall back to equal fractions
            return [1.0 / count] * count
        return [first, *others]


# ---------------------------------------------------------------------------
# SHELX .hkl
# ---------------------------------------------------------------------------

def read_shelx_hkl(path: str | Path) -> ReflectionData:
    """Read a SHELX ``HKLF 4`` reflection file.

    Accepts fixed-format ``3I4,2F8.2`` data and free-format fallback files.
    Reading stops at the terminating ``0 0 0`` record, as SHELX does.
    """
    return parse_shelx_hkl(Path(path).read_text(errors='replace'), source=path)


def parse_shelx_hkl(text: str, *, source: str | Path = '<text>') -> ReflectionData:
    """Parse SHELX ``HKLF 4`` reflection text.

    Tries the NumPy fixed-format fast path first, then falls back to per-record
    parsing for free-format or irregular files.
    """
    lines = text.splitlines()
    data = _parse_fixed_format(lines)
    if data is not None:
        return data

    hkl: list[tuple[int, int, int]] = []
    f_sq: list[float] = []
    sig: list[float] = []
    batch: list[int] = []

    for raw in lines:
        line = raw.rstrip('\r')
        if not line.strip():
            continue
        parsed = _parse_hkl_line(line)
        if parsed is None:
            continue
        h, k, l, fsq, s, n = parsed
        if h == 0 and k == 0 and l == 0:
            break
        hkl.append((h, k, l))
        f_sq.append(fsq)
        sig.append(s)
        batch.append(n)

    if not hkl:
        raise ValueError(f'No reflections found in {source}')

    return ReflectionData(
        hkl=np.array(hkl, dtype=np.int32),
        f_sq_meas=np.array(f_sq, dtype=float),
        sigma=np.array(sig, dtype=float),
        batch=np.array(batch, dtype=np.int32),
    )


#: Column boundaries of ``3I4,2F8.2`` plus optional batch.
_HKL_COLUMNS: tuple[tuple[int, int], ...] = (
    (0, 4), (4, 8), (8, 12), (12, 20), (20, 28), (28, 32),
)
_HKL_RECORD_WIDTH: int = 32

#: Character classes used by :func:`_fixed_format_mask`.
_CHAR_DIGIT: int = 1
_CHAR_IN_INTEGER: int = 2
_CHAR_IN_NUMBER: int = 4


def _build_char_classes() -> np.ndarray:
    """Byte-value to character-class table for :func:`_fixed_format_mask`."""
    table = np.zeros(256, dtype=np.uint8)
    for code in b'0123456789':
        table[code] = _CHAR_DIGIT | _CHAR_IN_INTEGER | _CHAR_IN_NUMBER
    # ``np.array()`` pads short records with NUL; treat it as blank.
    for code in b' +-\x00':
        table[code] = _CHAR_IN_INTEGER | _CHAR_IN_NUMBER
    for code in b'.eE':
        table[code] = _CHAR_IN_NUMBER
    return table


_CHAR_CLASS: np.ndarray = _build_char_classes()


def _terminator_row(codes: np.ndarray) -> int | None:
    """Index of the terminating ``0 0 0`` record, if present.

    Detected from the three index fields before any conversion, so normal
    trailers such as SADABS scaling reports are ignored just as SHELX ignores
    them.
    """
    zero = codes == 0x30
    blank = (codes == 0x20) | (codes == 0x00)
    terminator = np.ones(len(codes), dtype=bool)
    for start, stop in _HKL_COLUMNS[:3]:
        field_zero = zero[:, start:stop]
        terminator &= (field_zero | blank[:, start:stop]).all(axis=1)
        terminator &= field_zero.any(axis=1)
    found = np.flatnonzero(terminator)
    return int(found[0]) if found.size else None


def _fixed_format_mask(codes: np.ndarray) -> np.ndarray:
    """Mask records whose fixed-format fields can contain numbers."""
    classes = _CHAR_CLASS[codes]
    valid = np.ones(len(codes), dtype=bool)
    for position, (start, stop) in enumerate(_HKL_COLUMNS[:5]):
        field = classes[:, start:stop]
        allowed = _CHAR_IN_INTEGER if position < 3 else _CHAR_IN_NUMBER
        valid &= (field & allowed).all(axis=1)
        valid &= (field & _CHAR_DIGIT).any(axis=1)
    start, stop = _HKL_COLUMNS[5]
    return valid & (classes[:, start:stop] & _CHAR_IN_INTEGER).all(axis=1)


def _convert_fixed_records(chars: np.ndarray) -> ReflectionData | None:
    """Convert a byte matrix of fixed-format ``HKLF`` records."""
    def column(start: int, stop: int) -> np.ndarray:
        block = np.ascontiguousarray(chars[:, start:stop])
        return block.view(f'S{stop - start}').ravel()

    try:
        hkl = np.stack(
            [column(*bounds).astype(np.int32) for bounds in _HKL_COLUMNS[:3]],
            axis=1,
        )
    except ValueError:
        return None

    # SHELX stops at the terminating ``0 0 0`` record.
    end = np.flatnonzero(~hkl.any(axis=1))
    stop = int(end[0]) if end.size else len(hkl)
    if stop == 0:
        return None
    hkl = hkl[:stop]
    chars = chars[:stop]

    try:
        f_sq = column(*_HKL_COLUMNS[3]).astype(float)
        sigma = column(*_HKL_COLUMNS[4]).astype(float)
        raw_batch = np.char.strip(column(*_HKL_COLUMNS[5]))
        batch = np.ones(stop, dtype=np.int32)
        given = raw_batch != b''
        if given.any():
            batch[given] = raw_batch[given].astype(np.int32)
    except ValueError:
        return None

    return ReflectionData(hkl=hkl, f_sq_meas=f_sq, sigma=sigma, batch=batch)


def _parse_fixed_format(lines: list[str]) -> ReflectionData | None:
    """Parse fixed-format ``HKLF`` records column-wise with NumPy.

    The list is cut at the ``0 0 0`` terminator first, so normal trailers are
    ignored. If conversion then fails, clearly non-fixed-format rows are
    dropped once and conversion is retried; if that still fails, the caller
    falls back to record-by-record parsing.
    """
    try:
        rows = np.array(lines, dtype=f'S{_HKL_RECORD_WIDTH}')
    except (UnicodeEncodeError, ValueError):  # non-ASCII text
        return None
    if rows.size == 0:
        return None

    # ``np.array()`` pads short records with NUL, so blank lines become empty.
    rows = rows[np.char.strip(rows) != b'']
    if rows.size == 0:
        return None

    chars = rows.view('S1').reshape(len(rows), _HKL_RECORD_WIDTH)
    end = _terminator_row(chars.view(np.uint8))
    if end is not None:
        chars = chars[:end]
        if len(chars) == 0:
            return None

    data = _convert_fixed_records(chars)
    if data is not None:
        return data

    keep = _fixed_format_mask(chars.view(np.uint8))
    if keep.all() or not keep.any():
        return None
    return _convert_fixed_records(chars[keep])


def _parse_hkl_line(line: str) -> tuple[int, int, int, float, float, int] | None:
    """Parse one SHELX reflection record.

    The sixth column defaults to ``1``. In ``HKLF 5`` it carries component and
    overlap-group information.
    """
    if len(line) >= 28:
        try:
            batch = 1
            tail = line[28:32].strip()
            if tail:
                batch = int(tail)
            return (
                int(line[0:4]), int(line[4:8]), int(line[8:12]),
                float(line[12:20]), float(line[20:28]), batch,
            )
        except ValueError:
            pass
    fields = line.split()
    if len(fields) >= 5:
        try:
            batch = int(fields[5]) if len(fields) > 5 else 1
            return (
                int(fields[0]), int(fields[1]), int(fields[2]),
                float(fields[3]), float(fields[4]), batch,
            )
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# CIF reflection loops (fcf style)
# ---------------------------------------------------------------------------

def read_cif_reflections(source: CifSource) -> ReflectionData | None:
    """Read the first usable fcf-style ``_refln_*`` loop from a CIF."""
    for block in _cif_blocks(source):
        data = read_block_reflections(block)
        if data is not None:
            return data
    return None


def read_block_reflections(block) -> ReflectionData | None:
    """Read one block's fcf-style ``_refln_*`` loop, if present."""
    h_col = block.find_values('_refln_index_h')
    if not h_col:
        return None
    k_col = block.find_values('_refln_index_k')
    l_col = block.find_values('_refln_index_l')
    if not k_col or not l_col:
        return None

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
        return None

    if sig_col:
        sigma = np.array([_num(v) for v in sig_col], dtype=float)
        sigma_known = True
    else:
        sigma = np.ones_like(f_sq)
        sigma_known = False

    hkl = np.array(
        [[int(_num(a)), int(_num(b)), int(_num(c))]
         for a, b, c in zip(h_col, k_col, l_col)],
        dtype=np.int32,
    )

    f_calc = _cif_f_calc(block, len(hkl))
    return ReflectionData(hkl=hkl, f_sq_meas=f_sq, sigma=sigma,
                          f_calc=f_calc, sigma_known=sigma_known)


def _cif_f_calc(block, n: int) -> np.ndarray | None:
    """Extract ``F_calc`` from a CIF reflection loop.

    Uses ``_refln_F_calc`` or ``sqrt(_refln_F_squared_calc)`` and applies
    ``_refln_phase_calc`` in degrees when present.
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


def _num_column(values) -> np.ndarray:
    """Convert a CIF loop column to ``float``.

    Uses NumPy's fast path first, then falls back to :func:`_num` for esds or
    CIF nulls.
    """
    try:
        return np.array(values, dtype=float)
    except (TypeError, ValueError):
        return np.array([_num(v) for v in values], dtype=float)


def read_block_raw_reflections(block) -> ReflectionData | None:
    """Read a raw ``_diffrn_refln_*`` loop from one CIF block.

    This is unmerged measured data: effectively ``HKLF 4`` in CIF form. The
    uncertainty column is accepted as ``_diffrn_refln_intensity_u`` or the
    older ``_diffrn_refln_intensity_sigma``. Missing scale-group codes default
    to batch 1.
    """
    h_col = block.find_values('_diffrn_refln_index_h')
    if not h_col:
        return None
    k_col = block.find_values('_diffrn_refln_index_k')
    l_col = block.find_values('_diffrn_refln_index_l')
    intensity = block.find_values('_diffrn_refln_intensity_net')
    if not k_col or not l_col or not intensity:
        return None

    sigma_col = (block.find_values('_diffrn_refln_intensity_u')
                 or block.find_values('_diffrn_refln_intensity_sigma'))
    if sigma_col:
        sigma = _num_column(sigma_col)
        sigma_known = True
    else:
        sigma = np.ones(len(intensity))
        sigma_known = False

    hkl = np.stack(
        [_num_column(c).astype(np.int32) for c in (h_col, k_col, l_col)],
        axis=1,
    )
    batch_col = block.find_values('_diffrn_refln_scale_group_code')
    batch = (_num_column(batch_col).astype(np.int32) if batch_col
             else np.ones(len(hkl), dtype=np.int32))

    return ReflectionData(hkl=hkl, f_sq_meas=_num_column(intensity),
                          sigma=sigma, batch=batch, sigma_known=sigma_known)


def read_cif_raw_reflections(source: CifSource) -> ReflectionData | None:
    """Read the first usable raw ``_diffrn_refln_*`` loop from a CIF."""
    for block in _cif_blocks(source):
        data = read_block_raw_reflections(block)
        if data is not None:
            return data
    return None


def embedded_shelx_hkl(source: CifSource) -> str | None:
    """Return embedded SHELX ``.hkl`` text from a CIF, if present."""
    for block in _cif_blocks(source):
        text = block_shelx_hkl(block)
        if text is not None:
            return text
    return None


def block_shelx_hkl(block) -> str | None:
    """Return the ``_shelx_hkl_file`` text of a single CIF block, or ``None``."""
    value = block.find_value('_shelx_hkl_file')
    if not value:
        return None
    text = gemmi.cif.as_string(value)
    return text if text and text.strip() else None


def read_reflections(source: ReflectionSource) -> ReflectionData:
    """Read reflections from a path, CIF object or ready data.

    CIF sources are tried in this order: fcf-style ``_refln_*``, embedded
    ``_shelx_hkl_file``, then raw ``_diffrn_refln_*`` data.
    """
    if isinstance(source, ReflectionData):
        return source

    if _is_cif_object(source):
        data = _cif_object_reflections(source)
        if data is None:
            raise ValueError('No reflections found in the given CIF data')
        return data

    path = Path(source)
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

    try:
        data = read_cif_raw_reflections(path)
    except Exception:  # noqa: BLE001 - not a CIF after all
        data = None
    if data is not None:
        return data

    return read_shelx_hkl(path)


def _cif_object_reflections(source: CifSource) -> ReflectionData | None:
    """Reflections of an in-memory document or block, or ``None``.

    Preference order: processed ``_refln_*`` loop, embedded ``.hkl``, then raw
    ``_diffrn_refln_*`` measurements.
    """
    data = read_cif_reflections(source)
    if data is not None:
        return data
    text = embedded_shelx_hkl(source)
    if text is not None:
        return parse_shelx_hkl(text, source='<cif> (_shelx_hkl_file)')
    return read_cif_raw_reflections(source)


#: Sibling extensions searched for reflection data, by preference.
_REFLECTION_SUFFIXES: tuple[str, ...] = ('.hkl', '.fcf', '.fco', '.cif')


def has_reflections(source: ReflectionSource) -> bool:
    """Cheaply test whether *source* contains usable reflection data."""
    if isinstance(source, ReflectionData):
        return len(source) > 0
    if _is_cif_object(source):
        return any(block_has_reflections(block) for block in _cif_blocks(source))

    path = Path(source)
    if not path.is_file():
        return False
    if path.suffix.lower() == '.hkl':
        return True
    return any(block_has_reflections(block) for block in _cif_blocks(path))


def block_has_reflections(block) -> bool:
    """True if a CIF block has supported reflection data."""
    if block.find_values('_refln_index_h'):
        return True
    if block.find_value('_shelx_hkl_file'):
        return True
    return bool(block.find_values('_diffrn_refln_index_h'))


def find_reflection_file(model_path: str | Path) -> Path | None:
    """Locate the reflection data for *model_path*.

    Searches the model file first, then same-basename ``.hkl``, ``.fcf``,
    ``.fco`` and ``.cif`` siblings.
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

#: CIF tags known to carry a complete SHELX ``.res``/``.ins`` file.
_EMBEDDED_RES_TAGS: tuple[str, ...] = (
    '_shelx_res_file',
    '_iucr_refine_instructions_details',
    '_shelxl_version_number_res_file',
)


def embedded_shelx_res(source: CifSource) -> str | None:
    """Return embedded SHELX ``.res`` text from a CIF, if present."""
    for block in _cif_blocks(source):
        text = block_shelx_res(block)
        if text is not None:
            return text
    return None


def block_shelx_res(block) -> str | None:
    """Return embedded SHELX ``.res`` text from one block.

    Only text containing an ``FVAR`` card is accepted.
    """
    for tag in _EMBEDDED_RES_TAGS:
        value = block.find_value(tag)
        if not value:
            continue
        text = gemmi.cif.as_string(value)
        if text and re.search(r'^\s*FVAR', text, re.MULTILINE | re.IGNORECASE):
            return text
    return None


def block_shelx_parameters(block) -> ShelxParameters | None:
    """Refined SHELX parameters from a block's embedded ``.res``, or ``None``."""
    text = block_shelx_res(block)
    return _parse_shelx_text(text) if text is not None else None


def read_shelx_parameters(source: CifSource) -> ShelxParameters | None:
    """Read refined SHELX parameters for *source*.

    Path sources are tried as ``.res``/``.ins`` itself, same-basename sibling
    ``.res``/``.ins``, then embedded SHELX text. In-memory CIF objects only
    offer embedded SHELX text.
    """
    if _is_cif_object(source):
        text = embedded_shelx_res(source)
        return _parse_shelx_text(text) if text is not None else None

    path = Path(source)
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
    """Extract SHELX refinement parameters from instruction text.

    Reads ``CELL``, ``FVAR``, ``WGHT``, ``EXTI``, ``TWIN``, ``BASF`` and
    ``HKLF`` cards. Continuation markers (``=``) are ignored for these cards.
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
        elif upper.startswith('TWIN'):
            values = _floats(line[4:])
            if len(values) >= 9:
                params.twin_matrix = tuple(values[:9])
            else:  # a bare TWIN card means inversion (racemic) twinning
                params.twin_matrix = _DEFAULT_TWIN_MATRIX
            count = None
            if len(values) >= 10:
                count = int(values[9])
            elif len(values) == 1:  # "TWIN n" - matrix omitted, count given
                count = int(values[0])
            if count is not None:
                # Negative means general plus racemic twinning: |n| total
                # components, with the second half the Friedel opposites.
                params.twin_components = max(abs(count), 2)
                params.twin_racemic = count < 0
        elif upper.startswith('BASF'):
            params.basf = _floats(line[4:])
        elif upper.startswith('HKLF'):
            values = _floats(line[4:])
            if values:
                params.hklf = abs(int(values[0]))
            if len(values) >= 2:
                params.hklf_scale = values[1]
            if len(values) >= 11:
                matrix = tuple(values[2:11])
                if matrix != (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
                    params.hklf_matrix = matrix
            if len(values) >= 12:
                params.hklf_sigma_scale = values[11]
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
