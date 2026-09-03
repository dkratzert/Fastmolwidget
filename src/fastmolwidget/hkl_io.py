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

Sources
-------
Every CIF reader here takes a :data:`CifSource`: a path, an already parsed
:class:`gemmi.cif.Document`, or a single :class:`gemmi.cif.Block`.  Host
applications that keep an edited document in memory (FinalCif, for instance)
can therefore hand over the block the user is looking at instead of writing a
temporary file.

Everything in this module is Qt-free and only depends on ``gemmi``,
``numpy`` and ``shelxfile``.
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
    'read_block_reflections',
    'read_cif_document',
    'read_cif_reflections',
    'read_reflections',
    'read_shelx_hkl',
    'read_shelx_parameters',
]

#: Anything the CIF readers in this module accept: a file path, a parsed
#: document, or a single data block.
CifSource = str | Path | gemmi.cif.Document | gemmi.cif.Block


@lru_cache(maxsize=2)
def _parse_cif_document(path: str, fingerprint: tuple[int, int]):
    """Parse a CIF, memoised on *path* and *fingerprint*.

    *fingerprint* is not used in the body — it is only part of the cache key,
    so that an edited file is re-read instead of served from the cache.
    """
    return gemmi.cif.read(path)


def read_cif_document(path: str | Path):
    """Read a CIF into a :class:`gemmi.cif.Document`, with a small cache.

    Building one residual-density map needs the same CIF up to five times
    over — to find the reflection data, to read the embedded ``.hkl``, to read
    the embedded SHELX instructions and to build the model — and for a
    self-contained SHELXL CIF that file is megabytes large.  Parsing it once
    and handing out the same document is worth more than any of those readers
    can save individually.

    The cache key includes the file's modification time and size, so an edited
    file is always re-read.  Only the two most recent documents are kept, to
    bound the memory a large CIF can occupy.

    :param path: Path to the CIF.
    :returns: The parsed document.  Treat it as read-only; it is shared.
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
    """Return the real data blocks of *doc*, skipping any ``global_`` block.

    A ``global_`` block holds values inherited by the blocks that follow it;
    it is not a structure of its own, so every reader here ignores it.
    ``gemmi`` represents it as a block whose name is empty or ``'global'``.
    """
    return [block for block in doc
            if block.name and block.name.lower() != 'global']


def _cif_blocks(source: CifSource) -> list:
    """Return the data blocks of *source*, whatever kind of CIF source it is.

    A single block is wrapped in a list, a document is stripped of its
    ``global_`` block, and a path is parsed (through the document cache).

    :param source: A path, a :class:`gemmi.cif.Document` or a
        :class:`gemmi.cif.Block`.
    :returns: The data blocks, or an empty list when a path cannot be parsed
        as a CIF — callers treat that as "no data here" rather than an error.
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
    """A set of measured (and optionally calculated) structure factors.

    :param hkl: ``(N, 3)`` integer array of Miller indices.
    :param f_sq_meas: ``(N,)`` array of measured F² values, on the arbitrary
        scale of the raw data.
    :param sigma: ``(N,)`` array of σ(F²).
    :param f_calc: Optional ``(N,)`` complex array of calculated structure
        factors, present only when the source file already contained them
        (an fcf-style CIF).  ``None`` means *compute them yourself*.
    :param batch: Optional ``(N,)`` integer array with the sixth ``.hkl``
        column.  Its meaning depends on the ``HKLF`` format: for ``HKLF 5`` it
        is the twin-component number, negative on every record of an overlap
        group but the last; for ``HKLF 4`` it is the batch number, which some
        programs make negative to flag a reflection for the *R*\\ :sub:`free`
        set.  The two must not be confused.
    :param sigma_known: ``False`` when the source file carried no standard
        uncertainties and *sigma* holds a placeholder.  Consumers that judge
        the *significance* of an observation — such as the down-weighting of
        weak data in :func:`fastmolwidget.density.calculate_residual_density`
        — must not use σ in that case.
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

        .. warning::
           Only meaningful once the file is known to be ``HKLF 5``.  In
           ``HKLF 4`` data a negative batch number marks an
           *R*\\ :sub:`free` reflection instead, so callers must check the
           declared format first.
        """
        return self.batch is not None and bool(np.any(self.batch < 0))


#: Anything the reflection readers accept: a CIF source, a ``.hkl`` path, or
#: reflections that were already read.
ReflectionSource = CifSource | ReflectionData


#: The ``TWIN`` matrix SHELXL assumes when the card carries no numbers:
#: inversion, i.e. racemic twinning.
_DEFAULT_TWIN_MATRIX: tuple[float, ...] = (-1.0, 0.0, 0.0,
                                           0.0, -1.0, 0.0,
                                           0.0, 0.0, -1.0)


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
    :param twin_matrix: The ``TWIN`` 3×3 matrix in row-major order, or
        ``None`` when the structure is not twinned.  A bare ``TWIN`` card
        means inversion (racemic) twinning.
    :param twin_components: ``|n|`` from the ``TWIN`` card (default 2) — the
        total number of components.
    :param twin_racemic: ``True`` when the ``TWIN`` count was negative, i.e.
        general and racemic twinning are refined together.  Components
        ``m+1…2m`` are then the Friedel opposites of components ``1…m``.
    :param basf: The ``BASF`` batch scale factors — the volume fractions of
        twin components 2…n.  An empty list means 'perfect' twinning with all
        components equal.
    :param hklf: The ``HKLF`` format number (4 or 5).
    :param hklf_scale: ``S`` from the ``HKLF`` card; multiplies F² and σ(F²).
    :param hklf_matrix: The ``HKLF`` 3×3 index-transformation matrix in
        row-major order, or ``None`` for the identity.  Applied *before* the
        twin law, and required whenever the reflection file is indexed on a
        different setting from the model.
    :param hklf_sigma_scale: ``sm`` from the ``HKLF`` card; multiplies σ.
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
        """Return the volume fraction of every twin component.

        ``BASF`` gives the fractions of components 2…n and the first component
        takes the remainder.  An absent ``BASF`` means 'perfect' twinning, so
        all components share the volume equally.  Values are clamped because a
        refinement that has not converged can leave nonsense in ``BASF``.

        :returns: ``n`` fractions summing to 1.
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

    Real files run to tens of thousands of records, so the fixed-format layout
    is first parsed column-wise with NumPy (:func:`_parse_fixed_format`).  That
    fast path only applies when *every* record is fixed-format; free-format or
    otherwise irregular files fall back to the record-by-record parser, which
    produces exactly the same result.

    :param text: The reflection records.
    :param source: Only used in the error message.
    :raises ValueError: If no usable reflection could be parsed.
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


#: Column boundaries of a SHELX ``HKLF`` record (``3I4,2F8.2`` + batch).
_HKL_COLUMNS: tuple[tuple[int, int], ...] = (
    (0, 4), (4, 8), (8, 12), (12, 20), (20, 28), (28, 32),
)
_HKL_RECORD_WIDTH: int = 32

#: Character classes used by :func:`_fixed_format_mask`, as bit flags.
_CHAR_DIGIT: int = 1
_CHAR_IN_INTEGER: int = 2
_CHAR_IN_NUMBER: int = 4


def _build_char_classes() -> np.ndarray:
    """Byte-value → character-class table for :func:`_fixed_format_mask`."""
    table = np.zeros(256, dtype=np.uint8)
    for code in b'0123456789':
        table[code] = _CHAR_DIGIT | _CHAR_IN_INTEGER | _CHAR_IN_NUMBER
    # NUL is the padding np.array() adds to records shorter than the record
    # width, so it has to count as blank rather than as a foreign character.
    for code in b' +-\x00':
        table[code] = _CHAR_IN_INTEGER | _CHAR_IN_NUMBER
    for code in b'.eE':
        table[code] = _CHAR_IN_NUMBER
    return table


_CHAR_CLASS: np.ndarray = _build_char_classes()


def _fixed_format_mask(codes: np.ndarray) -> np.ndarray:
    """Mark the records whose fixed-format columns can hold a number.

    A purely character-level test: every field must consist of characters that
    may appear in a number and must contain at least one digit.  The
    classification is a single table lookup over the byte matrix, after which
    only the six narrow field slices are reduced.

    :param codes: ``(N, _HKL_RECORD_WIDTH)`` byte values of the records.
    :returns: ``(N,)`` boolean mask of the records worth converting.
    """
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
    """Convert a byte matrix of ``HKLF`` records column by column.

    :param chars: ``(N, _HKL_RECORD_WIDTH)`` ``S1`` matrix of the records.
    :returns: The reflections, or ``None`` when a column does not convert -
        which means the data is not fixed-format after all.
    """
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

    # SHELX stops reading at the terminating 0 0 0 record.
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

    Every record is turned into a row of a fixed-width byte matrix, so each
    column can be converted in one C-level cast instead of one Python call per
    line.  This is the difference between ~0.1 s and a few ms for a 40 000
    reflection file, and that cost is paid every time a residual-density map is
    computed.

    Real files do contain the odd stray record - a comment, or, as in
    ``41467_2015_BFncomms9288_MOESM1369_ESM.cif``, a CIF item that ended up
    inside the ``_shelx_hkl_file`` text block.  A single one of those used to
    send the entire file down the record-by-record parser, so if the straight
    conversion fails, the records that cannot be fixed-format numbers are
    dropped (:func:`_fixed_format_mask`) - exactly what the record-by-record
    parser does with them - and the conversion is tried once more.  Clean
    files never reach that second attempt and pay nothing for it.  If the
    retry fails too, the file really is not fixed-format and the
    record-by-record parser takes over.

    :param lines: The raw records, blank ones included.
    :returns: The reflections, or ``None`` when the data is not strictly
        fixed-format and the record-by-record parser has to take over.
    """
    try:
        rows = np.array(lines, dtype=f'S{_HKL_RECORD_WIDTH}')
    except (UnicodeEncodeError, ValueError):  # non-ASCII text
        return None
    if rows.size == 0:
        return None

    # np.array() pads short records with NUL, so blank lines become empty.
    rows = rows[np.char.strip(rows) != b'']
    if rows.size == 0:
        return None

    chars = rows.view('S1').reshape(len(rows), _HKL_RECORD_WIDTH)
    data = _convert_fixed_records(chars)
    if data is not None:
        return data

    keep = _fixed_format_mask(chars.view(np.uint8))
    if keep.all() or not keep.any():
        return None
    return _convert_fixed_records(chars[keep])


def _parse_hkl_line(line: str) -> tuple[int, int, int, float, float, int] | None:
    """Parse one SHELX ``.hkl`` record, fixed-format first, free-format after.

    The sixth column (batch / twin-component number) is optional and defaults
    to ``1``; for ``HKLF 5`` data it carries the domain number and the overlap
    grouping.

    :returns: ``(h, k, l, F², σ, batch)`` or ``None`` when the line is not a
        reflection record.
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
    """Read an fcf-style reflection loop from a CIF.

    Recognises ``_refln_index_h/k/l`` together with either
    ``_refln_F_squared_meas`` or ``_refln_F_meas``.  When the loop also
    provides calculated structure factors (``_refln_F_squared_calc`` /
    ``_refln_F_calc``, optionally with ``_refln_phase_calc``) these are
    returned as well, so they do not have to be recomputed.

    :param source: A path to a CIF (or ``.fcf``) file, a parsed document, or a
        single data block.
    :returns: The reflections of the first block that has them, or ``None``
        when there is no reflection loop at all.
    """
    for block in _cif_blocks(source):
        data = read_block_reflections(block)
        if data is not None:
            return data
    return None


def read_block_reflections(block) -> ReflectionData | None:
    """Read the fcf-style reflection loop of a single CIF block.

    :param block: A :class:`gemmi.cif.Block`.
    :returns: The reflections, or ``None`` when the block has no usable
        ``_refln_*`` loop.
    """
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


def embedded_shelx_hkl(source: CifSource) -> str | None:
    """Return the SHELX ``.hkl`` text embedded in a CIF, if there is one.

    Self-contained CIFs written by SHELXL carry the complete reflection file
    in ``_shelx_hkl_file``, which makes the CIF alone sufficient to compute a
    residual-density map.

    :param source: A path to the CIF, a parsed document, or a single block.
    :returns: The embedded reflection records, or ``None`` when absent.
    """
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
    """Read reflections from a ``.hkl`` file, a CIF or an in-memory source.

    For a path the format is chosen from the file suffix: ``.hkl`` uses
    :func:`read_shelx_hkl`.  For anything else the file is treated as a CIF and
    tried in this order — an fcf-style ``_refln_*`` loop, then a ``.hkl``
    embedded in ``_shelx_hkl_file`` — before falling back to the SHELX reader.
    A :class:`gemmi.cif.Document` or :class:`gemmi.cif.Block` is tried the same
    way, and a :class:`ReflectionData` is returned unchanged.

    :param source: The reflection source.
    :raises ValueError: If no reflections could be read.
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

    return read_shelx_hkl(path)


def _cif_object_reflections(source: CifSource) -> ReflectionData | None:
    """Reflections of an in-memory document or block, or ``None``."""
    data = read_cif_reflections(source)
    if data is not None:
        return data
    text = embedded_shelx_hkl(source)
    if text is not None:
        return parse_shelx_hkl(text, source='<cif> (_shelx_hkl_file)')
    return None


#: Sibling extensions searched for reflection data, in order of preference.
_REFLECTION_SUFFIXES: tuple[str, ...] = ('.hkl', '.fcf', '.fco', '.cif')


def has_reflections(source: ReflectionSource) -> bool:
    """Cheaply test whether *source* holds usable reflection data.

    Avoids parsing the whole file: a ``.hkl`` only has to exist, and a CIF is
    scanned for an fcf-style loop or an embedded ``_shelx_hkl_file``.

    :param source: A path, an in-memory document or block, or already read
        :class:`ReflectionData`.
    """
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
    """True when a CIF block carries an fcf loop or an embedded ``.hkl``."""
    if block.find_values('_refln_index_h'):
        return True
    return bool(block.find_value('_shelx_hkl_file'))


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


def embedded_shelx_res(source: CifSource) -> str | None:
    """Return the SHELX ``.res`` text embedded in a CIF, if there is one.

    Deposited CIFs frequently carry the complete final ``.res`` file in a
    semicolon-delimited text field, which lets us recover the exact refined
    ``FVAR`` / ``WGHT`` / ``EXTI`` values even when no separate ``.res`` file
    is available.

    :param source: A path to the CIF, a parsed document, or a single block.
    :returns: The embedded SHELX text, or ``None`` when absent.
    """
    for block in _cif_blocks(source):
        text = block_shelx_res(block)
        if text is not None:
            return text
    return None


def block_shelx_res(block) -> str | None:
    """Return the SHELX ``.res`` text embedded in a single CIF block.

    Only text that actually contains an ``FVAR`` card is accepted, so a field
    holding something else than a real SHELX file is ignored.
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
    """Read refined SHELX parameters for the model given by *source*.

    For a path the sources are tried in this order:

    1. *source* itself when it is a ``.res``/``.ins`` file;
    2. a ``.res`` (then ``.ins``) file sitting next to a CIF with the same
       basename;
    3. a SHELX ``.res`` block embedded inside the CIF
       (see :func:`embedded_shelx_res`).

    An in-memory document or block only offers the embedded block, since there
    is no directory to look in.

    :param source: The model file (``.res``, ``.ins`` or ``.cif``), a parsed
        document, or a single block.
    :returns: The refined parameters, or ``None`` when none could be found.
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
                # A negative count means general *and* racemic twinning: |n|
                # components in total, the second half being the Friedel
                # opposites of the first.
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
