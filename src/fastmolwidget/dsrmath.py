from __future__ import annotations

import random
import string
from math import sqrt, radians, cos, sin

import numpy as np


class SymmetryElement:
    """Symmetry operation in SHELX notation."""
    symm_ID = 1
    __slots__ = ['ID', 'centric', 'matrix', 'symms', 'trans']

    def __init__(self, symms, centric=False):
        """Build the operation from SHELX-style axis expressions."""
        self.centric = centric
        self.symms = symms
        self.ID = SymmetryElement.symm_ID
        SymmetryElement.symm_ID += 1
        lines = []
        trans = []
        for symm in self.symms:
            line, t = self._parse_line(symm)
            lines.append(line)
            trans.append(t)
        self.matrix = np.array(lines, dtype=float)
        self.trans = np.array(trans, dtype=float)
        if centric:
            self.matrix *= -1
            self.trans *= -1

    def __str__(self) -> str:
        m = self.matrix.astype(int)
        string = f"|{m[0, 0]:2} {m[0, 1]:2} {m[0, 2]:2}|   |{float(self.trans[0]):>4.2}| \n" \
                 f"|{m[1, 0]:2} {m[1, 1]:2} {m[1, 2]:2}| + |{float(self.trans[1]):>4.2}| \n" \
                 f"|{m[2, 0]:2} {m[2, 1]:2} {m[2, 2]:2}|   |{float(self.trans[2]):>4.2}| \n"
        return string

    def __repr__(self) -> str:
        return self.toShelxl()

    def __eq__(self, other: SymmetryElement) -> bool:
        """Compare two symmetry operations modulo lattice translations."""
        m = np.array_equal(self.matrix, other.matrix)
        t1 = self.trans % 1
        t2 = other.trans % 1
        t = np.array_equal(t1, t2)
        return m and t

    def __sub__(self, other: SymmetryElement) -> np.ndarray | float:
        """Return the translation difference, or ``999.0`` if inequivalent."""
        if not self == other:
            return 999.0
        return self.trans - other.trans

    def applyLattSymm(self, lattSymm):
        """Return a copy with the lattice translation of ``lattSymm`` added."""
        newSymm = SymmetryElement(self.toShelxl().split(','))
        newSymm.trans = (self.trans + lattSymm.trans)
        newSymm.centric = self.centric
        return newSymm

    def toShelxl(self):
        """Return the operation in SHELX syntax."""
        axes = ['X', 'Y', 'Z']
        lines = []
        for i in range(3):
            text = str(float(self.trans[i])) if self.trans[i] else ''
            for j in range(3):
                s = '' if not self.matrix[i, j] else axes[j]
                if self.matrix[i, j] < 0:
                    s = '-' + s
                elif s:
                    s = '+' + s
                text += s
            lines.append(text)
        return ', '.join(lines)

    def _parse_line(self, symm):
        symm = symm.upper().replace(' ', '')
        chars = ['X', 'Y', 'Z']
        line = []
        for char in chars:
            element, symm = self._partition(symm, char)
            line.append(element)
        if symm:
            trans = self._float(symm)
        else:
            trans = 0
        return line, trans

    def _float(self, string):
        try:
            return float(string)
        except ValueError:
            if '/' in string:
                string = string.replace('/', './') + '.'
                return eval(f'{string}')

    def _partition(self, symm, char):
        parts = symm.partition(char)
        if parts[1]:
            if parts[0]:
                sign = parts[0][-1]
            else:
                sign = '+'
            if sign == '-':
                return -1, ''.join((parts[0][:-1], parts[2]))
            else:
                return 1, ''.join((parts[0], parts[2])).replace('+', '')
        else:
            return 0, symm


def my_isnumeric(value: str):
    """True if ``value`` can be converted to a number."""
    try:
        float(value)
    except ValueError:
        return False
    return True


def mean(values):
    """Return the arithmetic mean."""
    return sum(values) / float(len(values))


def median(nums):
    """Return the median."""
    ls = sorted(nums)
    n = len(ls)
    if n == 0:
        raise ValueError("Need a non-empty iterable")
    # Odd-length list.
    elif n % 2 == 1:
        # ``//`` is floor division.
        return ls[n // 2]
    else:
        i = n // 2
        return (ls[i - 1] + ls[i]) / 2


def std_dev(data):
    """Return the sample standard deviation."""
    if len(data) == 0:
        return 0
    K = data[0]
    n = 0
    Sum = 0
    Sum_sqr = 0
    for x in data:
        n += 1
        Sum += x - K
        Sum_sqr += (x - K) * (x - K)
    variance = (Sum_sqr - (Sum * Sum) / n) / (n - 1)
    # Use sample variance.
    return sqrt(variance)


def nalimov_test(data):
    """Return indices flagged as outliers by the Nalimov test."""
    # Critical q values by degrees of freedom.
    f = {1 : 1.409, 2: 1.645, 3: 1.757, 4: 1.814, 5: 1.848, 6: 1.870, 7: 1.885, 8: 1.895,
         9 : 1.903, 10: 1.910, 11: 1.916, 12: 1.920, 13: 1.923, 14: 1.926, 15: 1.928,
         16: 1.931, 17: 1.933, 18: 1.935, 19: 1.936, 20: 1.937, 30: 1.945}
    fact = sqrt(float(len(data)) / (len(data) - 1))
    fval = len(data) - 2
    if fval < 2:
        return []
    outliers = []
    if fval in f:
        # Fallback threshold for larger samples.
        q_crit = f[fval]
    else:
        q_crit = 1.95
    for num, i in enumerate(data):
        q = abs(((i - median(data)) / std_dev(data)) * fact)
        if q > q_crit:
            outliers.append(num)
    return outliers


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """Return a random ID such as ``"L5J74W"``."""
    return ''.join(random.choice(chars) for _ in range(size))


def atomic_distance(p1: list, p2: list, cell=None, shortest_dist=False):
    """Return the distance between two points.

    With ``shortest_dist=True``, fold the fractional difference into the
    nearest periodic image before measuring it.
    """
    a, b, c, al, be, ga = 1, 1, 1, 1, 1, 1
    if cell:
        a, b, c = cell[:3]
        al = radians(cell[3])
        be = radians(cell[4])
        ga = radians(cell[5])
    if shortest_dist:
        x1, y1, z1 = [x + 99.5 for x in p1]
        x2, y2, z2 = [x + 99.5 for x in p2]
        dx = (x1 - x2) % 1 - 0.5
        dy = (y1 - y2) % 1 - 0.5
        dz = (z1 - z2) % 1 - 0.5
    else:
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        dx = (x1 - x2)
        dy = (y1 - y2)
        dz = (z1 - z2)
    if cell:
        return sqrt((a * dx) ** 2 + (b * dy) ** 2 + (c * dz) ** 2 + 2 * b * c * cos(al) * dy * dz + \
                    2 * dx * dz * a * c * cos(be) + 2 * dx * dy * a * b * cos(ga))
    else:
        return sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def determinante(a):
    """Return the determinant of a 3×3 matrix."""
    return (a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2])
            - a[1][0] * (a[0][1] * a[2][2] - a[2][1] * a[0][2])
            + a[2][0] * (a[0][1] * a[1][2] - a[1][1] * a[0][2]))


def subtract_vect(a, b):
    """Return ``a - b``."""
    return (a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2])


def dice_coefficient(a, b, case_insens=True):
    """Return the Sørensen-Dice coefficient of two strings."""
    if case_insens:
        a = a.lower()
        b = b.lower()
    if not len(a) or not len(b):
        return 0.0
    if len(a) == 1:
        a = a + '.'
    if len(b) == 1:
        b = b + '.'
    a_bigram_list = []
    for i in range(len(a) - 1):
        a_bigram_list.append(a[i:i + 2])
    b_bigram_list = []
    for i in range(len(b) - 1):
        b_bigram_list.append(b[i:i + 2])
    a_bigrams = set(a_bigram_list)
    b_bigrams = set(b_bigram_list)
    overlap = len(a_bigrams & b_bigrams)
    dice_coeff = overlap * 2.0 / (len(a_bigrams) + len(b_bigrams))
    return round(dice_coeff, 6)


def dice_coefficient2(a, b, case_insens=True):
    """Return the reversed Dice distance.

    Duplicate bigrams count distinctly, so ``"AA"`` and ``"AAAA"`` are not a
    perfect match. ``0`` is best, ``1`` worst.
    """
    if case_insens:
        a = a.lower()
        b = b.lower()
    if not len(a) or not len(b):
        return 1.0
    # Fast path for identical strings.
    if a == b:
        return 0.0
    # Single unequal characters cannot match.
    if len(a) == 1 or len(b) == 1:
        return 1.0
    # Build sorted bigram lists.
    a_bigram_list = [a[i:i + 2] for i in range(len(a) - 1)]
    b_bigram_list = [b[i:i + 2] for i in range(len(b) - 1)]
    a_bigram_list.sort()
    b_bigram_list.sort()
    # Local lengths for speed/readability.
    lena = len(a_bigram_list)
    lenb = len(b_bigram_list)
    # Match counters.
    matches = i = j = 0
    while i < lena and j < lenb:
        if a_bigram_list[i] == b_bigram_list[j]:
            matches += 2
            i += 1
            j += 1
        elif a_bigram_list[i] < b_bigram_list[j]:
            i += 1
        else:
            j += 1
    score = float(matches) / float(lena + lenb)
    score = 1 - score
    return round(score, 6)


def fft(x):
    """Return the recursive FFT of ``x``."""
    from cmath import exp, pi
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [exp(-2j * pi * k / N) * odd[k] for k in range(int(N / 2))]
    return [even[k] + T[k] for k in range(int(N / 2))] + \
        [even[k] - T[k] for k in range(int(N / 2))]


def levenshtein(s1, s2):
    """Return the Levenshtein edit distance."""
    s1 = s1.lower()
    s2 = s2.lower()
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Rows are one element longer than the strings.
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def distance(x1, y1, z1, x2, y2, z2, round_out=False):
    """Return the Euclidean distance for orthogonal axes."""
    import math as m
    d = m.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    if round_out:
        return round(d, round_out)
    else:
        return d


def vol_unitcell(a, b, c, al, be, ga):
    """Return the unit-cell volume."""
    ca, cb, cg = cos(radians(al)), cos(radians(be)), cos(radians(ga))
    v = a * b * c * sqrt(1 + 2 * ca * cb * cg - ca ** 2 - cb ** 2 - cg ** 2)
    return v


def almost_equal(a, b, places=3):
    """True if ``a`` and ``b`` agree to ``places`` decimal places."""
    return round(abs(a - b), places) == 0


def frac_to_cart(frac_coord: list, cell: list | tuple) -> list:
    """Convert fractional coordinates to Cartesian coordinates."""
    a, b, c, alpha, beta, gamma = cell
    x, y, z = frac_coord
    alpha = radians(alpha)
    beta = radians(beta)
    gamma = radians(gamma)
    cosastar = (cos(beta) * cos(gamma) - cos(alpha)) / (sin(beta) * sin(gamma))
    sinastar = sqrt(1 - cosastar ** 2)
    xc = a * x + (b * cos(gamma)) * y + (c * cos(beta)) * z
    yc = 0 + (b * sin(gamma)) * y + (-c * sin(beta) * cosastar) * z
    zc = 0 + 0 + (c * sin(beta) * sinastar) * z
    return [xc, yc, zc]


def cart_to_frac(cart_coord: list, cell: list) -> tuple:
    """Convert Cartesian coordinates to fractional coordinates."""
    a, b, c, alpha, beta, gamma = cell
    xc, yc, zc = cart_coord
    alpha = radians(alpha)
    beta = radians(beta)
    gamma = radians(gamma)
    cosastar = (cos(beta) * cos(gamma) - cos(alpha)) / (sin(beta) * sin(gamma))
    sinastar = sqrt(1 - cosastar ** 2)
    z = zc / (c * sin(beta) * sinastar)
    y = (yc - (-c * sin(beta) * cosastar) * z) / (b * sin(gamma))
    x = (xc - (b * cos(gamma)) * y - (c * cos(beta)) * z) / a
    return x, y, z
