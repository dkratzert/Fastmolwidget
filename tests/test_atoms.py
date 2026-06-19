import unittest

from fastmolwidget.atoms import get_radius, get_radius_from_element, get_atomic_number, get_element, get_atomlabel, \
    get_element_color


class TestAtoms(unittest.TestCase):

    def test_get_radius(self):
        r = get_radius(6)
        self.assertEqual(0.77, r)

    def test_get_radius_from_element(self):
        r = get_radius_from_element('F')
        self.assertEqual(0.72, r)

    def test_get_radius_from_element_with_suffix(self):
        self.assertEqual(get_radius_from_element('O'), get_radius_from_element('O2'))

    def test_get_radius_from_element_unknown_falls_back_to_carbon(self):
        self.assertEqual(get_radius_from_element('C'), get_radius_from_element('Xx'))

    def test_get_atomic_number(self):
        r = get_atomic_number('F')
        self.assertEqual(9, r)

    def test_get_element(self):
        r = get_element(7)
        self.assertEqual('N', r)

    def test_get_atomlabel(self):
        l = get_atomlabel('C12')
        self.assertEqual('C', l)
        l = get_atomlabel('Te1+')
        self.assertEqual('Te', l)

    def test_get_element_color(self):
        c = get_element_color('F')
        self.assertEqual('#90e001', c)
