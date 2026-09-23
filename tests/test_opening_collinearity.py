import unittest
import numpy as np
from scripts.check_factor_collinearity import diagnose

class CollinearityTests(unittest.TestCase):
    def test_orthogonal(self):
        x=np.array([[1,1],[1,-1],[-1,1],[-1,-1.]])
        vif,_,condition,rank,_=diagnose(x)
        np.testing.assert_allclose(vif,1);self.assertAlmostEqual(condition,1);self.assertEqual(rank,2)

    def test_linear_combination_and_constant(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(100,2))
        v,_,c,r,active=diagnose(np.column_stack([x,x.sum(1),np.ones(100)]))
        self.assertTrue(np.isinf(v[:3]).all());self.assertTrue(np.isnan(v[3]));self.assertEqual(r,2);self.assertTrue(np.isinf(c));self.assertFalse(active[3])

    def test_scale_and_offset_invariance(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(80,3));x[:,2]+=.6*x[:,0];w=np.arange(1,81)
        a=diagnose(x,w);b=diagnose(x*np.array([100,.1,5])+20,w)
        np.testing.assert_allclose(a[0],b[0]);self.assertAlmostEqual(a[2],b[2])

if __name__=='__main__':unittest.main()
