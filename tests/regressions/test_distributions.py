import unittest
from unittest import TestCase

import numpy as np
from scipy.special import beta
from scipy.stats import beta as scipy_beta

from mitoolspro.regressions.distributions import GeneralizedBetaDistribution


class TestGeneralizedBetaDistribution(TestCase):
    def setUp(self):
        self.standard_beta = GeneralizedBetaDistribution(alpha=2, beta=2, a=0, b=1)
        self.general_beta = GeneralizedBetaDistribution(alpha=2, beta=2, a=-1, b=1)
        self.asymmetric_beta = GeneralizedBetaDistribution(alpha=1, beta=3, a=0, b=1)
        self.test_points = np.linspace(0, 1, 100)
        self.test_points_general = np.linspace(-1, 1, 100)

    def test_initialization(self):
        beta_dist = GeneralizedBetaDistribution(alpha=2, beta=2, a=0, b=1)
        self.assertEqual(beta_dist.alpha, 2)
        self.assertEqual(beta_dist.beta, 2)
        self.assertEqual(beta_dist.a, 0)
        self.assertEqual(beta_dist.b, 1)
        with self.assertRaises(ValueError):
            GeneralizedBetaDistribution(alpha=-1, beta=2, a=0, b=1)
        with self.assertRaises(ValueError):
            GeneralizedBetaDistribution(alpha=2, beta=-1, a=0, b=1)
        with self.assertRaises(ValueError):
            GeneralizedBetaDistribution(alpha=2, beta=2, a=1, b=0)

    def test_pdf_standard_beta(self):
        expected = scipy_beta.pdf(self.test_points, a=2, b=2)
        actual = self.standard_beta.pdf_vectorized(self.test_points)
        np.testing.assert_almost_equal(actual, expected, decimal=10)

    def test_pdf_general_beta(self):
        x_transformed = (self.test_points_general + 1) / 2
        expected = scipy_beta.pdf(x_transformed, a=2, b=2) / 2
        actual = self.general_beta.pdf_vectorized(self.test_points_general)
        np.testing.assert_almost_equal(actual, expected, decimal=10)

    def test_pdf_asymmetric_beta(self):
        expected = scipy_beta.pdf(self.test_points, a=1, b=3)
        actual = self.asymmetric_beta.pdf_vectorized(self.test_points)
        np.testing.assert_almost_equal(actual, expected, decimal=10)

    def test_pdf_boundary_conditions(self):
        self.assertEqual(self.standard_beta.pdf(0), 0)
        self.assertEqual(self.standard_beta.pdf(1), 0)
        self.assertEqual(self.general_beta.pdf(-1), 0)
        self.assertEqual(self.general_beta.pdf(1), 0)

    def test_pdf_normalization(self):
        for dist in [self.standard_beta, self.general_beta, self.asymmetric_beta]:
            x = np.linspace(dist.a, dist.b, 1000)
            y = dist.pdf_vectorized(x)
            integral = np.trapz(y, x)
            np.testing.assert_almost_equal(integral, 1.0, decimal=5)

    def test_pdf_symmetry(self):
        x = self.test_points
        y = 1 - x
        np.testing.assert_almost_equal(
            self.standard_beta.pdf_vectorized(x),
            self.standard_beta.pdf_vectorized(y),
            decimal=10,
        )

    def test_pdf_non_negativity(self):
        for dist in [self.standard_beta, self.general_beta, self.asymmetric_beta]:
            x = np.linspace(dist.a, dist.b, 1000)
            y = dist.pdf_vectorized(x)
            self.assertTrue(np.all(y >= 0))

    def test_pdf_special_cases(self):
        uniform_beta = GeneralizedBetaDistribution(alpha=1, beta=1, a=0, b=1)
        np.testing.assert_almost_equal(
            uniform_beta.pdf_vectorized(self.test_points),
            np.ones_like(self.test_points),
            decimal=10,
        )

        u_shaped_beta = GeneralizedBetaDistribution(alpha=0.5, beta=0.5, a=0, b=1)
        x = np.array([0, 0.5, 1])
        y = u_shaped_beta.pdf_vectorized(x)
        self.assertTrue(y[0] > y[1])
        self.assertTrue(y[2] > y[1])

    def test_pdf_parameter_effects(self):
        beta1 = GeneralizedBetaDistribution(alpha=1, beta=2, a=0, b=1)
        beta2 = GeneralizedBetaDistribution(alpha=2, beta=2, a=0, b=1)
        beta3 = GeneralizedBetaDistribution(alpha=2, beta=1, a=0, b=1)

        x = np.array([0.7, 0.3])
        y1 = beta1.pdf_vectorized(x)
        y2 = beta2.pdf_vectorized(x)
        y3 = beta3.pdf_vectorized(x)

        self.assertTrue(y2[0] > y1[0])
        self.assertTrue(y1[1] > y2[1])
        self.assertTrue(y3[0] > y2[0])

    def test_pdf_scale_invariance(self):
        beta1 = GeneralizedBetaDistribution(alpha=2, beta=2, a=0, b=1)
        beta2 = GeneralizedBetaDistribution(alpha=2, beta=2, a=0, b=2)

        x1 = self.test_points
        x2 = 2 * x1
        np.testing.assert_almost_equal(
            beta1.pdf_vectorized(x1), beta2.pdf_vectorized(x2) * 2, decimal=10
        )

    def test_pdf_caching(self):
        x = 0.5
        result1 = self.standard_beta.pdf(x)
        result2 = self.standard_beta.pdf(x)
        self.assertEqual(result1, result2)

    def test_pdf_out_of_bounds(self):
        x = np.array([-1, 0, 0.5, 1, 2])
        y = self.standard_beta.pdf_vectorized(x)
        expected = np.array([0, 0, 1.5, 0, 0])
        np.testing.assert_almost_equal(y, expected, decimal=10)


if __name__ == "__main__":
    unittest.main()
