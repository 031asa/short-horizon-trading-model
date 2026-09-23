import unittest
import numpy as np
import pandas as pd
from utils.opening_anchored import candidates,pick,validation_metrics,explained_by_core
from utils.opening_task_window import ALL,BASE,INTERACTIONS
from utils.opening_task_window import bounded_features
from utils.opening_ic import SessionData,ICConfig
from test_opening_no_cold import fixture,START

class AnchoredTests(unittest.TestCase):
    def test_exhaustive_and_fixed_core(self):
        rows=candidates(ALL)
        self.assertEqual(sum(r['structure']=='add1' for r in rows),22)
        self.assertEqual(sum(r['structure']=='add2' for r in rows),231)
        self.assertEqual(sum(r['structure']=='composite' for r in rows),6)
        self.assertEqual(len(rows),260)
        for r in rows:
            self.assertTrue(set(BASE).issubset(r['features']));self.assertLessEqual(len(r['features']),6)
            for f in r['features']:
                if f in INTERACTIONS:self.assertTrue(set(INTERACTIONS[f]).issubset(r['features']))

    def test_snapshot_only_pool_still_keeps_core(self):
        self.assertEqual(len(candidates(BASE)),1)
        self.assertEqual(candidates(BASE)[0]['structure'],'baseline')

    def row(self,**kwargs):
        return dict(dict(accuracy3=.60,accuracy5to10=.55,window=3,features=BASE,
                         logloss3=.8,C=1.,candidate_id='base'),**kwargs)

    def test_priority_accuracy_then_persistence_then_wait(self):
        a=self.row(window=1,accuracy3=.590,accuracy5to10=.8)
        b=self.row(window=5,accuracy3=.60,accuracy5to10=.55)
        c=self.row(window=3,accuracy3=.596,accuracy5to10=.558)
        self.assertIs(pick([a,b,c]),c)
        d=self.row(window=2,accuracy3=.596,accuracy5to10=.554)
        self.assertIs(pick([a,b,c,d]),d)

    def test_simplicity_loss_and_regularization_ties(self):
        a=self.row(features=BASE+['extra'],logloss3=.7)
        b=self.row(logloss3=.8,C=.1)
        c=self.row(logloss3=.79,C=1.)
        d=self.row(logloss3=.79,C=.1)
        self.assertIs(pick([a,b,c,d]),d)

    def test_date_equal_and_flat_retained(self):
        y=np.tile(np.array([1,1,0,-1])[:,None],(1,10));p=np.tile([.1,.1,.8],(4,1))
        m=validation_metrics(y,p,np.array(['a','a','a','b']))
        self.assertAlmostEqual(m['accuracy3'],1/3)
        self.assertAlmostEqual(m['accuracy5to10'],1/3)

    def test_core_r2_detects_linear_duplicate(self):
        rng=np.random.default_rng(9);f=pd.DataFrame(rng.normal(size=(200,3)),columns=BASE)
        f['duplicate']=2*f[BASE[0]]-f[BASE[1]]+3;f['independent']=rng.normal(size=200);f['trade_date']=np.repeat(np.arange(10),20)
        r=explained_by_core(f,['duplicate','independent'])
        self.assertAlmostEqual(r['duplicate'],1.)
        self.assertLess(r['independent'],.1)

    def test_no_volume_denominator_produces_missing_not_infinity(self):
        frame=fixture();frame.Volume=100
        row=bounded_features(SessionData(frame,START,ICConfig()),0,3)
        self.assertTrue(np.isnan(row['D06_PreQIVolume']))
        self.assertTrue(np.isnan(row['F08_SignedVolume']))
        self.assertFalse(np.isinf([row[f] for f in ALL]).any())

    def test_new_windows_remain_causal_under_future_and_past_change(self):
        frame=fixture()
        for w in [6,7,9,10]:
            task=3.;end=task+w
            original=bounded_features(SessionData(frame,START,ICConfig()),task,w)
            changed=frame.copy();outside=(changed.Datetime<START+pd.Timedelta(seconds=task))|(changed.Datetime>START+pd.Timedelta(seconds=end))
            changed.loc[outside,'Volume']=999999;changed.loc[outside,'LastPrice']=200;changed.loc[outside,'AskVolume1']=88888
            rebuilt=bounded_features(SessionData(changed,START,ICConfig()),task,w)
            np.testing.assert_allclose([original[f] for f in ALL],[rebuilt[f] for f in ALL],equal_nan=True)

if __name__=='__main__':unittest.main()
