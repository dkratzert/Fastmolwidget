"""The headline numbers of a SHELXL refinement, parsed from its output.

Qt-free.  SHELXL prints the same summary lines to its console and to the
``.lst`` file, so :class:`RefinementSummary` reads both: :meth:`feed` takes
console lines one at a time while SHELXL runs, :meth:`update_from_list_file`
reads the final values -- and the weighting-scheme suggestion, which only the
``.lst`` file carries -- once it is done.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_NUM = r'(-?\d+(?:\.\d*)?)'
_ESD = r'(-?\d+\.\d+\(\d+\))'

_CYCLE = re.compile(r'wR2\s*=\s*' + _NUM + r'\s+before cycle\s+(\d+)\s+for\s+(\d+)\s+data and\s+'
                    r'(\d+)\s*/\s*(\d+)\s+parameters')
_GOOF = re.compile(r'GooF = S =\s*' + _NUM + r';\s+Restrained GooF =\s*' + _NUM
                   + r'\s+for\s+(\d+)\s+restraints')
_FINAL_GOOF = re.compile(r'wR2 =\s*' + _NUM + r',\s*GooF = S =\s*' + _NUM
                         + r',\s*Restrained GooF =\s*' + _NUM + r'\s+for all data')
_R1 = re.compile(r'R1 =\s*' + _NUM + r'\s+for\s+(\d+)\s+Fo > 4sig\(Fo\)\s+and\s+' + _NUM
                 + r'\s+for all\s+(\d+)\s+data')
_SHIFT = re.compile(r'Mean shift/esd =\s*' + _NUM + r'\s+Maximum =\s*' + _NUM
                    + r'\s+for\s+(.+?)(?:\s+at\s+[\d:]+)?\s*$')
_FLACK = re.compile(r'Flack x =\s*' + _ESD + r'\s+by classical fit')
# In the .lst file this is a continuation line without the "Flack x =".
_PARSONS = re.compile(_ESD + r"\s+from\s+\d+\s+selected quotients \(Parsons")
_PEAK = re.compile(r'(Highest peak|Deepest hole)\s+' + _NUM + r'\s+at\s+(.*?)\s*$')
_NEAREST = re.compile(r'\[\s*' + _NUM + r'\s*A from\s+(\S+)\s*\]')
_WGHT = re.compile(r'Recommended weighting scheme:\s*WGHT\s+(.*?)\s*$')

#: Where the part of a ``.lst`` file holding the final values starts.
_FINAL_SECTION = 'Final Structure Factor Calculation'


@dataclass
class Peak:
    """A difference-map extremum: height in e/Å³, the nearest atom and how far."""
    height: float
    position: str
    distance: float | None = None
    atom: str | None = None


@dataclass
class RefinementSummary:
    """The numbers one looks at first after a SHELXL run; ``None`` = not seen yet."""

    #: Least-squares cycles requested (``L.S.``/``CGLS``), if known.
    cycles: int | None = None
    #: The cycle SHELXL is in, or the last one it ran.
    cycle: int | None = None
    #: Whether the final structure-factor calculation has been reached.
    final: bool = False
    data: int | None = None
    parameters: int | None = None
    restraints: int | None = None
    wr2: float | None = None
    goof: float | None = None
    restrained_goof: float | None = None
    r1_obs: float | None = None
    n_obs: int | None = None
    r1_all: float | None = None
    n_all: int | None = None
    mean_shift: float | None = None
    max_shift: float | None = None
    max_shift_parameter: str | None = None
    flack: str | None = None
    flack_parsons: str | None = None
    peak: Peak | None = None
    hole: Peak | None = None
    #: SHELXL's suggested ``WGHT`` parameters (``.lst`` only).
    wght: tuple[float, ...] | None = None

    def feed(self, line: str) -> bool:
        """Take one line of SHELXL output; returns whether a value changed."""
        if (match := _CYCLE.search(line)) is not None:
            wr2, cycle, data, refined, total = match.groups()
            self.wr2 = float(wr2)
            self.data = int(data)
            self.parameters = int(total)
            if int(refined) == 0:
                # The final structure-factor calculation, not a l.s. cycle:
                # it is numbered one past the last cycle that ran.
                self.final = True
                self.cycle = int(cycle) - 1
            else:
                self.cycle = int(cycle)
            return True
        if (match := _FINAL_GOOF.search(line)) is not None:
            self.wr2, self.goof, self.restrained_goof = map(float, match.groups())
            return True
        if (match := _GOOF.search(line)) is not None:
            goof, restrained, restraints = match.groups()
            self.goof, self.restrained_goof = float(goof), float(restrained)
            self.restraints = int(restraints)
            return True
        if (match := _R1.search(line)) is not None:
            r1_obs, n_obs, r1_all, n_all = match.groups()
            self.r1_obs, self.n_obs = float(r1_obs), int(n_obs)
            self.r1_all, self.n_all = float(r1_all), int(n_all)
            return True
        if (match := _SHIFT.search(line)) is not None:
            mean, maximum, parameter = match.groups()
            self.mean_shift, self.max_shift = float(mean), float(maximum)
            self.max_shift_parameter = ' '.join(parameter.split())
            return True
        if (match := _FLACK.search(line)) is not None:
            self.flack = match.group(1)
            return True
        if (match := _PARSONS.search(line)) is not None:
            self.flack_parsons = match.group(1)
            return True
        if (match := _PEAK.search(line)) is not None:
            kind, height, where = match.groups()
            peak = Peak(float(height), where)
            if (nearest := _NEAREST.search(where)) is not None:
                peak.distance, peak.atom = float(nearest.group(1)), nearest.group(2)
                peak.position = where[:nearest.start()].strip()
            if kind == 'Highest peak':
                self.peak = peak
            else:
                self.hole = peak
            return True
        if (match := _WGHT.search(line)) is not None:
            try:
                self.wght = tuple(float(value) for value in match.group(1).split())
            except ValueError:
                return False
            return True
        return False

    def update_from_list_file(self, path: str | Path) -> bool:
        """Read the final values from a ``.lst`` file; ``False`` if it cannot be read.

        Only the tail is parsed -- from the last cycle's shift/esd line (or,
        for an unfinished run, its last cycle) on: a list file runs to
        megabytes, and every value wanted here is in that part.
        """
        try:
            text = Path(path).read_text(encoding='latin-1')
        except OSError:
            return False
        final = text.rfind(_FINAL_SECTION)
        if final < 0:
            start = text.rfind(' before cycle')
        else:
            shift = text.rfind('Mean shift/esd', 0, final)
            start = final if shift < 0 else shift
        start = text.rfind('\n', 0, max(start, 0)) + 1
        for line in text[start:].splitlines():
            self.feed(line)
        return True

    @classmethod
    def from_list_file(cls, path: str | Path, cycles: int | None = None) -> RefinementSummary | None:
        """The summary of the refinement *path* records, or ``None`` if unreadable."""
        summary = cls(cycles=cycles)
        return summary if summary.update_from_list_file(path) else None


def space_group_of(shx) -> tuple[str | None, str]:
    """The space group of a parsed SHELX model, as ``(short name, explanation)``.

    Derived from the ``LATT`` and ``SYMM`` cards, which are what SHELXL
    refines in.  The name in ``TITL ... in P-31c`` is free text and can be
    wrong -- a non-centrosymmetric model titled with its centrosymmetric
    supergroup, say -- so it is only the fallback when the cards do not
    describe a tabulated setting.
    """
    from fastmolwidget.density import _shelx_spacegroup

    try:
        group = _shelx_spacegroup(shx)
    except Exception:  # noqa: BLE001 - malformed cards must not break display
        group = None
    if group is not None:
        return group.short_name(), f'{group.hm} (No. {group.number}), from LATT and SYMM'
    title = getattr(shx, 'space_group', None)
    if title:
        return title, 'From the title; LATT and SYMM do not match a tabulated setting'
    return None, ''
