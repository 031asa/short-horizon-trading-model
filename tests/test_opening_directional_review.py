import unittest
import numpy as np
import pandas as pd
from test_opening_ic import fixture,START
from utils.opening_ic import ICConfig,SessionData,feature_registry
from utils.directional_review import reviewed_registry,variants,enrich_atomic,variant_values


class DirectionalReviewTests(unittest.TestCase):
    def test_original_priors_and_definitions_preserved(self):
        base=feature_registry(ICConfig());review=reviewed_registry(base, conservative=False)
        old=review.loc[review.review_origin.eq('original')]
        for col in ('factor','definition','role','unit'):
            self.assertEqual(base[col].tolist(),old[col].tolist())
        self.assertEqual(base.expected_sign_signed.tolist(),old.original_expected_sign_signed.tolist())
        fixed=base.expected_sign_signed.ne('uncertain')
        self.assertEqual(base.loc[fixed,'expected_sign_signed'].tolist(),old.loc[fixed,'expected_sign_signed'].tolist())
        self.assertEqual(len(variants()),28)
        self.assertEqual(int((old.review_change.eq('new_hypothesis') & old.evaluate).sum()),96)
        self.assertFalse(review.factor.duplicated().any())
        names=set(base.factor)
        for spec in variants():self.assertTrue(set(spec['inputs'])<=names)

    def test_conservative_priors_withdraw_extra_assumptions(self):
        base=feature_registry(ICConfig());current=reviewed_registry(base)
        old=current.loc[current.review_origin.eq('original')]
        self.assertEqual(base.expected_sign_signed.tolist(),old.expected_sign_signed.tolist())
        self.assertTrue(current.loc[current.review_origin.eq('derived'),'expected_sign_signed'].eq('uncertain').all())
        self.assertEqual(base.expected_sign_absolute.tolist(),old.expected_sign_absolute.tolist())
        self.assertTrue(current.loc[current.review_change.eq('assumption_withdrawn'),'aggressive_expected_sign_signed'].isin(['positive','negative']).all())

    def test_gating_direction_and_neutral_zero(self):
        vals={'m':np.array([2.,-2.,2.,-2.,0.]),'o':np.array([1.,-1.,-1.,1.,0.])}
        s=dict(inputs=['m','o'],operation='aligned')
        np.testing.assert_array_equal(variant_values(s,vals),[2,-2,0,0,0])
        s['operation']='opposed'
        np.testing.assert_array_equal(variant_values(s,vals),[0,0,2,-2,0])

    def test_quote_rates_algebra_and_zero_activity(self):
        s=dict(inputs=list('abcdef'),operation='quote_rv')
        values={k:np.array([v,0.]) for k,v in zip('abcdef',[3,1,2,2,4,6])}
        np.testing.assert_allclose(variant_values(s,values),[1.25,0])

    def test_scaled_age_and_stable_thin_depth(self):
        x={'d':np.array([1,-1,0]),'age':np.array([1.,3.,8.])}
        np.testing.assert_allclose(variant_values(dict(inputs=['d','age'],operation='signed_log'),x),[np.log(2),-np.log(4),0])
        y=variant_values(dict(inputs=['d','age'],operation='thin'),{'d':np.ones(3),'age':np.array([-1000.,0.,1000.])})
        np.testing.assert_allclose(y,[1,.5,0])

    def sample(self,frame=None,t=20):
        frame=fixture() if frame is None else frame
        return pd.DataFrame([SessionData(frame,START,ICConfig()).features(t)])

    def test_missing_parent_never_becomes_inactive_zero(self):
        a=self.sample();a['M01_MADeviation_h5s']=0.
        a['A07_OFIDirection_h5s']=np.nan;a['A07_OFIDirection_h5s__status']='ZERO_DENOM'
        b=enrich_atomic(a)
        for stem in ('TrendConfirmed','CounterflowDeviation'):
            self.assertTrue(np.isnan(b['M01_'+stem+'_h5s'].iloc[0]))
            self.assertIn('ZERO_DENOM',b['M01_'+stem+'_h5s__status'].iloc[0])

    def test_left_censored_age_is_not_filled(self):
        a=self.sample();a['A05_QIState']=0.;a['A05_StateAge']=np.nan
        a['A05_StateAge__status']='LEFT_CENSORED';a['A05_StateAgeLowerBound']=20.
        b=enrich_atomic(a)
        self.assertTrue(np.isnan(b.A05_SignedStateAge.iloc[0]))
        self.assertIn('LEFT_CENSORED',b.A05_SignedStateAge__status.iloc[0])

    def test_future_truncation_and_market_perturbation(self):
        f=fixture();f.LastPrice=100+np.sin(np.arange(len(f))*.7)*.8
        f.BidVolume1=20+np.arange(len(f))%7*5;f.AskVolume1=40+np.arange(len(f))%5*4
        t=20;end=START+pd.Timedelta(seconds=t)
        changed=f.copy();future=changed.Datetime>end
        changed.loc[future,['LastPrice','BidPrice1','AskPrice1']]+=100
        changed.loc[future,['BidVolume1','AskVolume1']]*=10
        changed.loc[future,'Volume']+=10000
        columns=[v['factor'] for v in variants()]
        expected=enrich_atomic(self.sample(f,t))[columns]
        pd.testing.assert_frame_equal(expected,enrich_atomic(self.sample(f.loc[~future].copy(),t))[columns])
        pd.testing.assert_frame_equal(expected,enrich_atomic(self.sample(changed,t))[columns])

    def test_row_local_and_label_independent(self):
        a=pd.concat([self.sample(t=10),self.sample(t=20)],ignore_index=True).copy()
        a['y_decision_1s']=[1,2]
        expected=enrich_atomic(a);altered=a.copy();altered['y_decision_1s']=[1e9,-1e9]
        actual=enrich_atomic(altered)
        cols=[s['factor'] for s in variants()]
        pd.testing.assert_frame_equal(expected[cols],actual[cols])
        pd.testing.assert_frame_equal(expected[cols].iloc[:1],enrich_atomic(a.iloc[:1])[cols])
        pd.testing.assert_frame_equal(expected[a.columns],a)

    def test_gap_parent_status_propagates(self):
        a=self.sample();a['C01_PriceRV_h5s']=np.nan;a['C01_PriceRV_h5s__status']='NO_COVERAGE'
        b=enrich_atomic(a)
        self.assertTrue(np.isnan(b.C01_QIRV_h5s.iloc[0]))
        self.assertIn('NO_COVERAGE',b.C01_QIRV_h5s__status.iloc[0])


if __name__=='__main__':unittest.main()
