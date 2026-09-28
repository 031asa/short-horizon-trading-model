import unittest
import numpy as np
from utils.opening_fill_distribution import weighted_quantile,weighted_km,normal_diagnostics


class DistributionTests(unittest.TestCase):
    def test_quantile_respects_weights(self):
        np.testing.assert_array_equal(weighted_quantile([1,5,9],[1,8,1],[0,.1,.5,.9,1]),[1,1,5,5,9])

    def test_censored_order_is_not_a_late_fill(self):
        curve=weighted_km([1,2,2],[1,0,0],[1,1,1])
        self.assertAlmostEqual(curve.cdf.iloc[-1],1/3)
        self.assertAlmostEqual(curve.survival.iloc[-1],2/3)

    def test_event_precedes_censor_at_same_time(self):
        curve=weighted_km([1,1,2],[1,0,1],[1,1,1])
        self.assertAlmostEqual(curve.cdf.iloc[0],1/3)
        self.assertAlmostEqual(curve.cdf.iloc[-1],1)

    def test_zero_time_fills_retained(self):
        c=weighted_km([0,1,2],[1,1,0],[.5,.25,.25])
        self.assertEqual(c.cdf.iloc[0],.5)
        self.assertEqual(c.cdf.iloc[-1],.75)

    def test_weighted_moments_and_discreteness(self):
        r=normal_diagnostics([-1,0,1],[1,2,1])
        self.assertEqual(r['mean'],0)
        self.assertEqual(r['skew'],0)
        self.assertAlmostEqual(r['excess_kurtosis'],-1)
        self.assertGreaterEqual(r['normal_cdf_gap'],.25-1e-12)
