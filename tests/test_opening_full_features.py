"""Hand-calculated cases for the full PDF, including event maturity and aliases."""
import unittest
import numpy as np
import pandas as pd
from test_opening_ic import fixture, START
from utils.opening_ic import ICConfig,SessionData,feature_registry
from utils.factor_catalog import FAMILIES


class FullFeatureTests(unittest.TestCase):
    def setUp(self):self.cfg=ICConfig()

    def test_registry_covers_every_family_and_preserves_unfixed_priors(self):
        r=feature_registry(self.cfg)
        self.assertEqual(set(r.family_id),set(FAMILIES));self.assertEqual(len(FAMILIES),59)
        self.assertFalse(r.factor.duplicated().any())
        self.assertTrue(r.loc[r.role.eq('quality'),'evaluate'].eq(False).all())
        self.assertEqual(r.set_index('factor').loc['F05_Momentum_h10s','expected_sign_signed'],'uncertain')

    def test_side_addition_depletion_and_quote_sync(self):
        f=fixture();f.BidPrice1=100.;f.AskPrice1=101.
        f.BidVolume1=20+np.arange(len(f))%3*10
        f.AskVolume1=30+np.arange(len(f))%4*5
        data=SessionData(f,START,self.cfg);x=data.features(10)
        i=20;s=10;c=5;w=np.full(10,.5)
        qb=f.BidVolume1.to_numpy();qa=f.AskVolume1.to_numpy()
        ub=np.maximum(np.diff(qb[s:i+1]),0).sum()/(c*np.average(qb[s:i],weights=w))
        ca=np.maximum(-np.diff(qa[s:i+1]),0).sum()/(c*np.average(qa[s:i],weights=w))
        self.assertAlmostEqual(x['B04_BidAddition_h5s'],ub)
        self.assertAlmostEqual(x['B05_AskDepletion_h5s'],ca)
        self.assertEqual(x['B04_BidSameShare_h5s'],1.)
        self.assertTrue(np.isnan(x['B08_CoShare_h5s']))
        f.loc[20,'BidPrice1']=100.2
        x=SessionData(f,START,self.cfg).features(10)
        self.assertEqual(x['B08_SingleShare_h5s'],1.)
        self.assertEqual(x['B04_BidSameCount_h5s'],9)

    def test_ordered_path_and_nonzero_runs(self):
        f=fixture([0,1,2,3,4,5]);f.LastPrice=[100,100.8,100.4,100.4,100,100.2]
        x=SessionData(f,START,self.cfg).features(5)
        self.assertEqual(x['C03_Range_h5s'],4)
        self.assertEqual(x['C04_Drawdown_h5s'],4)
        self.assertEqual(x['C04_Runup_h5s'],4)
        self.assertEqual(x['C06_SignedRun_h5s'],1)
        self.assertAlmostEqual(x['C06_ReversalShare_h5s'],2/3)
        self.assertEqual(x['C05_ZeroCount_h5s'],1)

    def test_linear_trend_residual_current_point_not_fitted(self):
        f=fixture();f.LastPrice=100+np.arange(len(f))*.2
        x=SessionData(f,START,self.cfg).features(10)
        self.assertAlmostEqual(x['M11_TrendSlope_h5s'],2.)
        self.assertAlmostEqual(x['M11_TrendResidual_h5s'],0.,places=9)
        f.loc[20,'LastPrice']+=2
        y=SessionData(f,START,self.cfg).features(10)
        self.assertAlmostEqual(y['M11_TrendSlope_h5s'],2.)
        self.assertAlmostEqual(y['M11_TrendResidual_h5s'],10.)

    def test_historical_ma_identities_and_outer_warmup(self):
        f=fixture();f.LastPrice=100+(np.arange(len(f))%13)*.2
        data=SessionData(f,START,self.cfg);x=data.features(20)
        self.assertAlmostEqual(x['M06_MeanGap_h10s_short5s'],x['M01_MADeviation_h10s']-x['M01_MADeviation_h5s'])
        previous=data.features(18)
        speed=(x['decision_lastprice']-previous['decision_lastprice'])/.2/2
        self.assertAlmostEqual(x['M04_DeviationSpeed_h5s_lag2s']+x['M05_MeanSpeed_h5s_lag2s'],speed)
        early=data.features(9)
        self.assertTrue(np.isnan(early['M07_AboveShare_h10s_g5s']))
        self.assertEqual(early['M07_AboveShare_h10s_g5s__status'],'WARMUP')

    def test_age_is_censored_until_event_and_resets_after_gap(self):
        f=fixture();data=SessionData(f,START,self.cfg);x=data.features(10)
        self.assertTrue(np.isnan(x['C07_PriceAge']))
        self.assertEqual(x['C07_PriceAgeLowerBound'],10)
        self.assertEqual(x['C07_PriceAgeLeftCensored'],1)
        f.loc[18:,'LastPrice']=100.2
        x=SessionData(f,START,self.cfg).features(10)
        self.assertEqual(x['C07_PriceAge'],1)
        f=f.drop(index=range(19,25))
        x=SessionData(f,START,self.cfg).features(13)
        self.assertTrue(np.isnan(x['C07_PriceAge']))
        self.assertEqual(x['C07_PriceAgeLowerBound'],.5)

    def test_recovery_maturity_and_nonoverlap(self):
        f=fixture();f.BidVolume1=100.;f.AskVolume1=100.
        # Independent drops at 0.5, 3.0, 5.5, 8.0, each fully recovers after .5s.
        for t in (.5,3,5.5,8):f.loc[f.Datetime.eq(START+pd.Timedelta(seconds=t)),'BidVolume1']=50
        data=SessionData(f,START,self.cfg)
        x=data.features(9.5)
        self.assertEqual(x['B09_BidSameCount_h10s'],3)
        self.assertEqual(x['B09_BidImmatureCount_h10s'],1)
        self.assertEqual(x['B09_BidUnrecoveredCount_h10s'],0)
        self.assertEqual(x['B09_BidRecovery_h10s'],1)
        self.assertEqual(x['B09_BidSuccessTime_h10s'],.5)
        y=data.features(10)
        self.assertEqual(y['B09_BidSameCount_h10s'],4)
        self.assertEqual(y['B09_BidImmatureCount_h10s'],0)
        self.assertTrue(np.isnan(y['B09_AskRecovery_h10s']))
        # Drop with a quote exit is not a failed same-quote recovery.
        f.loc[12:,'BidPrice1']=99.6
        z=SessionData(f,START,self.cfg).features(9.5)
        self.assertEqual(z['B09_BidExitCount_h10s'],1)
        self.assertEqual(z['B09_BidSameCount_h10s'],2)
        self.assertTrue(np.isnan(z['B09_BidRecovery_h10s']))

    def test_opening_history_does_not_fill_gaps_or_include_current_extreme(self):
        f=fixture([0,.5,1,10,10.5]);f.LastPrice=[100,100.2,100.4,101,101.2]
        x=SessionData(f,START,self.cfg).features(10.5)
        self.assertEqual(x['R03_NewHigh'],1)
        self.assertAlmostEqual(x['R03_PriorHighDistance'],1)
        self.assertEqual(x['R02_DistanceHigh'],0)
        self.assertEqual(x['R06_price_OpeningCoverage'],1.5)
        expected=(.5*500+.5*501+.5*505)/1.5
        self.assertAlmostEqual(x['M02_OpeningMADeviation'],506-expected)

    def test_aliases_and_two_segments_have_no_overlapping_increments(self):
        f=fixture();f.Volume=100+np.arange(len(f))**2
        x=SessionData(f,START,self.cfg).features(20)
        self.assertEqual(x['R05_VolumeRateChange_lag5s'],x['D03_VolumeRateChange_lag5s'])
        expected=((40**2-30**2)/5)-((30**2-20**2)/5)
        self.assertEqual(x['D03_VolumeRateChange_lag5s'],expected)

    def test_lagged_history_ends_before_signal_and_constant_correlation_undefined(self):
        f=fixture();f.LastPrice=100+(np.arange(len(f))%7)*.2
        f.BidVolume1=20+np.arange(len(f))%9
        a=SessionData(f,START,self.cfg).features(10)
        prefix=f.loc[f.Datetime<=START+pd.Timedelta(seconds=10)]
        b=SessionData(prefix,START,self.cfg).features(10)
        self.assertEqual(a['D09_LagPairCount_h10s_lag2s'],16)
        self.assertAlmostEqual(a['D09_LagOFICorr_h10s_lag2s'],b['D09_LagOFICorr_h10s_lag2s'])
        x=SessionData(fixture(),START,self.cfg).features(10)
        self.assertTrue(np.isnan(x['D09_SignedVolumeCorr_h5s']))
        self.assertEqual(x['D09_SignedVolumeCorr_h5s__status'],'ZERO_VARIANCE')

    def test_incremental_prices_telescope_and_missing_endpoint_does_not_disappear(self):
        f=fixture();f.LastPrice=100+np.arange(len(f))%9*.2
        y=SessionData(f,START,self.cfg).labels(10)
        for anchor in ('decision','arrival'):
            self.assertEqual(sum(y[f'y_{anchor}_incremental_{u}s'] for u in range(1,31)),y[f'y_{anchor}_30s'])
        f=f.drop(index=[24,25,26,27,28])
        y=SessionData(f,START,self.cfg).labels(10)
        self.assertTrue(np.isnan(y['y_decision_incremental_10s']))

    def test_neutral_band_does_not_count_same_side_return_as_crossing(self):
        from utils.opening_features import FeatureEngine
        f=fixture(np.arange(0,4.5,.5));d=SessionData(f,START,self.cfg)
        e=FeatureEngine(d)
        states=[-1,0,-1,0,1,0,1,0,-1]
        e.records=[{'M01_MADeviation_h5s':float(v),'_mean_h5s':500.-v} for v in states]
        r=e.records[-1];e.derived(8,5,r)
        self.assertEqual(r['M08_CrossUpRate_h5s_g5s'],.25)
        self.assertEqual(r['M08_CrossDownRate_h5s_g5s'],.25)
        self.assertEqual(r['M08_CrossAge_h5s_g5s'],0)

    def test_r04_time_weighted_prior_percentile_excludes_current(self):
        f=fixture();f.BidPrice1=99.;f.AskPrice1=100.+(np.arange(len(f))%3)*.2
        f.loc[20,'AskPrice1']=101.
        d=SessionData(f,START,self.cfg);x=d.features(10)
        spread=np.rint((f.AskPrice1-f.BidPrice1)/.2).to_numpy()
        self.assertAlmostEqual(x['R04_SpreadDifference'],spread[20]-spread[:20].mean())
        self.assertAlmostEqual(x['R04_SpreadPercentile'],1.)
        # A derived state begins at its first valid historical computation.
        values=np.array([row['D01_VolumeRate_h5s'] for row in d._extended.records])
        prior=values[:20];prior=prior[np.isfinite(prior)]
        expected=np.mean((prior<values[20])+.5*(prior==values[20]))
        self.assertAlmostEqual(x['R04_VolumeRatePercentile_h5s'],expected)


if __name__=='__main__':unittest.main()
