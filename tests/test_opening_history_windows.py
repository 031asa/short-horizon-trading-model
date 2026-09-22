import unittest
import numpy as np
import pandas as pd
from test_opening_ic import fixture,START
from utils.opening_ic import ICConfig,SessionData,feature_registry
from utils.directional_review import reviewed_registry,enrich_atomic


class HistoryWindowTests(unittest.TestCase):
    def test_window_is_the_actual_price_lookback(self):
        f=fixture();ticks=np.arange(len(f))**2%31;f.LastPrice=100+ticks*.2
        x=SessionData(f,START,ICConfig(history_seconds=tuple(range(1,11)))).features(20)
        for h in range(1,11):
            self.assertAlmostEqual(x[f'F05_Momentum_h{h}s'],ticks[40]-ticks[40-2*h])

    def test_short_histories_do_not_relax_support(self):
        f=fixture();f.LastPrice=100+np.arange(len(f))*.2
        x=SessionData(f,START,ICConfig(history_seconds=tuple(range(1,11)))).features(20)
        self.assertTrue(np.isnan(x['M11_TrendSlope_h1s']))
        self.assertEqual(x['M11_TrendSlope_h1s__status'],'NOT_ENOUGH_EVENTS')
        for h in (1,2,3):
            self.assertTrue(np.isnan(x[f'D09_SignedVolumeCorr_h{h}s']))
            self.assertEqual(x[f'D09_SignedVolumeCorr_h{h}s__status'],'NOT_ENOUGH_EVENTS')
        self.assertEqual(x['B09_BidRecovery_h10s__status'],'NOT_ENOUGH_EVENTS')

    def test_window_expansion_keeps_old_values_and_exact_events(self):
        f=fixture();f.LastPrice=100+np.sin(np.arange(len(f)))*.6
        f.BidVolume1=20+np.arange(len(f))%9*3
        old=pd.DataFrame([SessionData(f,START,ICConfig()).features(20)])
        windows=tuple(range(1,11));cfg=ICConfig(history_seconds=windows)
        new=enrich_atomic(pd.DataFrame([SessionData(f,START,cfg).features(20)]),windows)
        old=enrich_atomic(old)
        names=reviewed_registry(feature_registry(ICConfig())).factor.tolist()
        for n in names:
            np.testing.assert_allclose(old[n],new[n],equal_nan=True)
            self.assertEqual(old[n+'__status'].iloc[0],new[n+'__status'].iloc[0])

    def test_new_historical_derived_values_remain_causal(self):
        f=fixture();f.LastPrice=100+np.cos(np.arange(len(f)))*.8
        cfg=ICConfig(history_seconds=tuple(range(1,11)))
        future=f.Datetime>START+pd.Timedelta(seconds=20)
        changed=f.copy();changed.loc[future,'LastPrice']+=100
        columns=[f'{name}_h{h}s'+suffix for h in (1,3,7,9) for name,suffix in
                 [('M04_DeviationSpeed','_lag2s'),('M09_AbsDeviationMean','_g5s'),('R04_PriceRVPercentile','')]]
        results=[pd.DataFrame([SessionData(frame,START,cfg).features(20)])[columns] for frame in (f,f.loc[~future].copy(),changed)]
        pd.testing.assert_frame_equal(results[0],results[1]);pd.testing.assert_frame_equal(results[0],results[2])


if __name__=='__main__':unittest.main()
