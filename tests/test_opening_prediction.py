"""Leakage, class mapping and metric contracts of the small-model pilot."""
import unittest
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.cache/model-packages'))
import numpy as np
import pandas as pd
from utils.opening_prediction import folds, weights, score, daily_loss, fit_model, class_prior, factors, specs, objective

class PredictionTests(unittest.TestCase):
    def test_whole_date_walk_forward(self):
        dates=[f'{i:02}' for i in range(38)]
        split=folds(dates)
        self.assertEqual([len(b) for a,b in split],[5,5,5,3])
        self.assertEqual([len(a) for a,b in split],[20,25,30,35])
        self.assertEqual([x for a,b in split for x in b],dates[20:])
        for a,b in split:self.assertLess(max(a),min(b))

    def test_equal_date_weights_and_prior(self):
        d=np.array(['a','a','a','b'])
        w=weights(d)
        self.assertAlmostEqual(w[:3].sum(),w[3])
        p=class_prior(np.array([-1,-1,-1,1]),d)
        self.assertAlmostEqual(p[0],p[2])
        self.assertGreater(p[1],0)

    def test_flat_is_a_class_not_removed(self):
        y=np.array([-1,0,1])
        metrics,cm=score(y,np.array([[.8,.1,.1],[.1,.1,.8],[.1,.1,.8]]))
        self.assertAlmostEqual(metrics['accuracy'],2/3)
        self.assertEqual(metrics['flat_recall'],0)
        self.assertEqual(cm.sum(),3)
        self.assertEqual(cm[1,2],1)

    def test_day_equal_logloss_not_pooled(self):
        y=np.array([-1,-1,-1,1]); p=np.array([[.8,.1,.1]]*3+[[.1,.1,.8]])
        self.assertAlmostEqual(daily_loss(y,p,['a','a','a','b']),-np.log(.8))

    def test_scaler_uses_training_data_only(self):
        x=np.array([[-3.],[-2.],[-1.],[0.],[1.],[2.],[3.],[4.],[5.]])
        y=np.array([-1,-1,0,0,1,1,-1,0,1]); dates=np.array(['a']*3+['b']*6)
        m=fit_model(x,y,dates,1)
        mean=m['scale'].mean_.copy(); coef=m['model'].coef_.copy()
        m.predict_proba(np.array([[1e12],[-1e12]]))
        np.testing.assert_array_equal(m['scale'].mean_,mean)
        np.testing.assert_array_equal(m['model'].coef_,coef)
        self.assertAlmostEqual(mean[0],np.average(x[:,0],weights=weights(dates)))
        np.testing.assert_array_equal(m.classes_,[-1,0,1])

    def test_small_fixed_catalog(self):
        self.assertEqual(len(specs()),12)
        self.assertEqual(len(factors(1)),6)
        self.assertEqual(factors(1)[:3],factors(5)[:3])
        self.assertTrue(all('_h5s' in f for f in factors(5)[3:]))

    def test_objective_gradient_and_hessian(self):
        rng=np.random.default_rng(12)
        x=np.column_stack([rng.normal(size=(12,2)),np.ones(12)])
        y=np.arange(12)%3;w=np.ones(12)/12;t=rng.normal(size=(3,3))*.1
        loss,g,h=objective(t,x,y,w,.03,True);eps=1e-5
        for k in range(9):
            a=t.copy();b=t.copy();a.flat[k]+=eps;b.flat[k]-=eps
            la,ga=objective(a,x,y,w,.03);lb,gb=objective(b,x,y,w,.03)
            self.assertAlmostEqual((la-lb)/(2*eps),g.flat[k],places=7)
            np.testing.assert_allclose((ga-gb).ravel()/(2*eps),h[:,k],atol=1e-8)

    def test_separable_signal_and_constant_column(self):
        rng=np.random.default_rng(3);x=rng.normal(size=120)
        y=np.where(x<-.4,-1,np.where(x>.4,1,0))
        xx=np.column_stack([x,np.ones(120)])
        m=fit_model(xx,y,np.repeat(np.arange(12),10),10)
        p=m.predict_proba(xx)
        self.assertGreater(np.mean(np.argmax(p,axis=1)-1==y),.95)
        np.testing.assert_allclose(p.sum(1),1)
        self.assertLess(m['model'].gradient_max,1e-8)

if __name__=='__main__':unittest.main()
