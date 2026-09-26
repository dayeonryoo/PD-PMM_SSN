"""Unit tests for the Q1 element kernels in python/fem_q1.py.

Run from the repository root with:
    python3 -m unittest discover -s python/tests -t python
"""

import unittest

import numpy as np

import fem_q1 as fem

TIGHT = 1e-12


class TestVelocityFieldWConstant(unittest.TestCase):
    def test_is_constant_everywhere_and_unit_norm(self):
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        for x, y in [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (0.3, 0.9)]:
            wx, wy = fem.velocity_field_w_constant(x, y)
            self.assertAlmostEqual(float(wx), -inv_sqrt2, delta=TIGHT)
            self.assertAlmostEqual(float(wy), inv_sqrt2, delta=TIGHT)
            self.assertAlmostEqual(float(wx) ** 2 + float(wy) ** 2, 1.0, delta=TIGHT)

    def test_broadcasts_over_arrays(self):
        x = np.linspace(0.0, 1.0, 7)
        wx, wy = fem.velocity_field_w_constant(x, x)
        self.assertEqual(wx.shape, (7,))
        self.assertEqual(wy.shape, (7,))


class TestShape(unittest.TestCase):
    def test_partition_of_unity(self):
        # sum_i phi_i == 1 and sum_i grad(phi_i) == 0 at any reference point.
        for s, t in [(-1.0, -1.0), (0.0, 0.0), (0.3, -0.7), (1.0, 1.0)]:
            phi, dphids, dphidt = fem.shape(s, t)
            self.assertAlmostEqual(phi.sum(), 1.0, delta=TIGHT)
            self.assertAlmostEqual(dphids.sum(), 0.0, delta=TIGHT)
            self.assertAlmostEqual(dphidt.sum(), 0.0, delta=TIGHT)

    def test_nodal_basis_property_at_vertices(self):
        # phi_i is 1 at vertex i and 0 at the other three.
        vertices = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        for i, (s, t) in enumerate(vertices):
            phi, _, _ = fem.shape(s, t)
            expected = np.zeros(4)
            expected[i] = 1.0
            np.testing.assert_allclose(phi, expected, atol=TIGHT)


class TestDeriv(unittest.TestCase):
    def test_unit_square_element_jacobian_and_derivatives(self):
        # Reference-to-physical map of [0,1]^2 has jac = 1/4 everywhere.
        xl = np.array([[0.0, 1.0, 1.0, 0.0]])
        yl = np.array([[0.0, 0.0, 1.0, 1.0]])
        phi, dphidx, dphidy, jac = fem.deriv(0.0, 0.0, xl, yl)

        self.assertAlmostEqual(float(jac[0]), 0.25, delta=TIGHT)
        # Physical derivatives of a partition of unity still sum to zero.
        self.assertAlmostEqual(float(dphidx.sum()), 0.0, delta=TIGHT)
        self.assertAlmostEqual(float(dphidy.sum()), 0.0, delta=TIGHT)
        self.assertAlmostEqual(float(phi.sum()), 1.0, delta=TIGHT)

    def test_vectorises_over_elements(self):
        # Two translated copies of the same element must give identical results.
        xl = np.array([[0.0, 0.5, 0.5, 0.0], [0.5, 1.0, 1.0, 0.5]])
        yl = np.array([[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]])
        _, dphidx, dphidy, jac = fem.deriv(0.3, -0.2, xl, yl)
        self.assertEqual(dphidx.shape, (2, 4))
        self.assertEqual(jac.shape, (2,))
        np.testing.assert_allclose(dphidx[0], dphidx[1], atol=TIGHT)
        np.testing.assert_allclose(dphidy[0], dphidy[1], atol=TIGHT)

    def test_raises_on_inverted_element(self):
        # Swapping two vertices reverses the orientation, so jac <= 0.
        xl = np.array([[0.0, 0.0, 1.0, 1.0]])
        yl = np.array([[0.0, 1.0, 1.0, 0.0]])
        with self.assertRaises(ValueError):
            fem.deriv(0.0, 0.0, xl, yl)


class TestGauss2x2(unittest.TestCase):
    def test_rule_integrates_bilinears_exactly(self):
        # sum of weights = area of [-1,1]^2 = 4, and the rule is exact for
        # the Q1 shape functions (each integrates to 1 over the reference square).
        pts = fem.gauss_2x2()
        self.assertEqual(len(pts), 4)
        self.assertAlmostEqual(sum(wt for _, _, wt in pts), 4.0, delta=TIGHT)

        total = np.zeros(4)
        for s, t, wt in pts:
            phi, _, _ = fem.shape(s, t)
            total += phi * wt
        np.testing.assert_allclose(total, np.ones(4), atol=TIGHT)


class TestInterpolateAndSource(unittest.TestCase):
    def test_interpolate_xy_recovers_element_centre(self):
        xl = np.array([[0.0, 1.0, 1.0, 0.0]])
        yl = np.array([[0.0, 0.0, 1.0, 1.0]])
        phi, _, _ = fem.shape(0.0, 0.0)
        xx, yy = fem.interpolate_xy(phi, xl, yl)
        self.assertAlmostEqual(float(xx[0]), 0.5, delta=TIGHT)
        self.assertAlmostEqual(float(yy[0]), 0.5, delta=TIGHT)

    def test_gauss_transprt_evaluates_wind_at_interpolated_point(self):
        xl = np.array([[0.0, 1.0, 1.0, 0.0]])
        yl = np.array([[0.0, 0.0, 1.0, 1.0]])
        phi, _, _ = fem.shape(0.0, 0.0)
        wx, wy = fem.gauss_transprt(phi, xl, yl)
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        np.testing.assert_allclose(wx, [-inv_sqrt2], atol=TIGHT)
        np.testing.assert_allclose(wy, [inv_sqrt2], atol=TIGHT)

    def test_gauss_source_is_identically_zero(self):
        xl = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
        yl = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        phi, _, _ = fem.shape(0.0, 0.0)
        src = fem.gauss_source(phi, xl, yl)
        np.testing.assert_array_equal(src, np.zeros(2))


class TestUniform1dCoords(unittest.TestCase):
    def test_spans_unit_interval_uniformly(self):
        x = fem.uniform_1d_coords(5)
        np.testing.assert_allclose(x, [0.0, 0.25, 0.5, 0.75, 1.0], atol=TIGHT)
        self.assertEqual(x[0], 0.0)
        self.assertEqual(x[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
