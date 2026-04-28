from __future__ import annotations

import pytest
import numpy as np

from fastmolwidget.dsrmath import (
    SymmetryElement,
    almost_equal,
    atomic_distance,
    cart_to_frac,
    determinante,
    dice_coefficient,
    dice_coefficient2,
    distance,
    fft,
    frac_to_cart,
    id_generator,
    levenshtein,
    mean,
    median,
    nalimov_test,
    std_dev,
    subtract_vect,
    vol_unitcell,
)


# ── SymmetryElement ──────────────────────────────────────────────────────────

class TestSymmetryElement:
    def test_eq_same_translation(self):
        s1 = SymmetryElement(['0.5', '0.5', '0.5'])
        s2 = SymmetryElement(['0.5', '0.5', '0.5'])
        assert s1 == s2

    def test_eq_differs_by_lattice_translation(self):
        s1 = SymmetryElement(['1.5', '1.5', '1.5'])
        s2 = SymmetryElement(['0.5', '0.5', '0.5'])
        assert s1 == s2

    def test_neq_different_translation(self):
        s3 = SymmetryElement(['1', '0.5', '0.5'])
        s4 = SymmetryElement(['0.5', '0.5', '0.5'])
        assert s3 != s4


# ── mean ─────────────────────────────────────────────────────────────────────

class TestMean:
    def test_integers(self):
        assert mean([1, 2, 3, 4, 1, 2, 3, 4]) == 2.5

    def test_with_large_value(self):
        assert round(mean([1, 2, 3, 4, 1, 2, 3, 4.1, 1000000]), 4) == 111113.3444


# ── median ───────────────────────────────────────────────────────────────────

class TestMedian:
    def test_single_element(self):
        assert median([2]) == 2

    def test_even_length(self):
        assert median([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]) == 2.5

    def test_odd_length(self):
        assert median([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4.1, 1000000]) == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Need a non-empty iterable"):
            median([])


# ── std_dev ──────────────────────────────────────────────────────────────────

class TestStdDev:
    L1 = [1.334, 1.322, 1.345, 1.451, 1.000, 1.434, 1.321, 1.322]
    L2 = [1.234, 1.222, 1.345, 1.451, 2.500, 1.234, 1.321, 1.222]

    def test_std_dev_l1(self):
        assert round(std_dev(self.L1), 8) == 0.13797871

    def test_std_dev_l2(self):
        assert round(std_dev(self.L2), 8) == 0.43536797

    def test_median_of_l1(self):
        assert median(self.L1) == 1.328

    def test_mean_of_l1(self):
        assert mean(self.L1) == 1.316125


# ── nalimov_test ─────────────────────────────────────────────────────────────

def test_nalimov_test():
    data = [1.120, 1.234, 1.224, 1.469, 1.145, 1.222, 1.123, 1.223, 1.2654, 1.221, 1.215]
    assert nalimov_test(data) == [3]


# ── id_generator ─────────────────────────────────────────────────────────────

def test_id_generator_single_char():
    assert id_generator(1, 'a') == 'a'


# ── atomic_distance ──────────────────────────────────────────────────────────

def test_atomic_distance():
    cell = [10.5086, 20.9035, 20.5072, 90, 94.13, 90]
    coord1 = [-0.186843, 0.282708, 0.526803]
    coord2 = [-0.155278, 0.264593, 0.600644]
    assert atomic_distance(coord1, coord2, cell) == 1.5729229943265979


# ── determinante ─────────────────────────────────────────────────────────────

def test_determinante():
    m1 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert determinante(m1) == 8


# ── subtract_vect ────────────────────────────────────────────────────────────

def test_subtract_vect():
    assert subtract_vect([1, 2, 3], [3, 2, 2]) == (-2, 0, 1)


# ── dice_coefficient ─────────────────────────────────────────────────────────

class TestDiceCoefficient:
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ('hallo', 'holla', 0.25),
            ('Banze', 'Benzene', 0.444444),
            ('halo', 'Haaallo', 0.75),
            ('hallo', 'Haaallo', 0.888889),
            ('hallo', 'Hallo', 1.0),
            ('aaa', 'BBBBB', 0.0),
        ],
    )
    def test_dice_coefficient(self, a, b, expected):
        assert dice_coefficient(a, b) == expected


# ── dice_coefficient2 ────────────────────────────────────────────────────────

class TestDiceCoefficient2:
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ('hallo', 'holla', 0.75),
            ('Banze', 'Benzene', 0.6),
            ('halo', 'Haaallo', 0.333333),
            ('hallo', 'Haaallo', 0.2),
            ('hallo', 'Hallo', 0.0),
            ('aaa', 'BBBBB', 1.0),
            ('', '', 1.0),
        ],
    )
    def test_dice_coefficient2(self, a, b, expected):
        assert dice_coefficient2(a, b) == expected


# ── fft ──────────────────────────────────────────────────────────────────────

def test_fft():
    result = fft([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    formatted = ' '.join("%5.3f" % abs(f) for f in result)
    assert formatted == "4.000 2.613 0.000 1.082 0.000 1.082 0.000 2.613"


# ── levenshtein ──────────────────────────────────────────────────────────────

def test_levenshtein():
    assert levenshtein('hallo', 'holla') == 2


# ── distance ─────────────────────────────────────────────────────────────────

class TestDistance:
    def test_diagonal(self):
        assert distance(1, 1, 1, 2, 2, 2, 4) == 1.7321

    def test_axis_aligned(self):
        assert distance(1, 0, 0, 2, 0, 0, 4) == 1.0


# ── vol_unitcell ─────────────────────────────────────────────────────────────

def test_vol_unitcell():
    assert vol_unitcell(2, 2, 2, 90, 90, 90) == 8.0


# ── almost_equal ─────────────────────────────────────────────────────────────

class TestAlmostEqual:
    def test_close_values(self):
        assert almost_equal(1.0001, 1.0005) is True

    def test_different_values(self):
        assert almost_equal(1.1, 1.0005) is False

    def test_integers(self):
        assert almost_equal(2, 1) is False


# ── frac_to_cart / cart_to_frac ──────────────────────────────────────────────

CELL = [10.5086, 20.9035, 20.5072, 90, 94.13, 90]


def test_frac_to_cart():
    coord1 = [-0.186843, 0.282708, 0.526803]
    assert frac_to_cart(coord1, CELL) == [-2.741505423999065, 5.909586678000002, 10.775200700893734]


def test_cart_to_frac():
    coords = [-2.74150542399906, 5.909586678, 10.7752007008937]
    assert cart_to_frac(coords, CELL) == (-0.1868429999999998, 0.28270799999999996, 0.5268029999999984)

