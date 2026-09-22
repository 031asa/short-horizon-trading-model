"""Numerical, causal, timing and missing-data tests for the IC experiment."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.opening_ic import (ICConfig, SessionData, build_session_atomic,
                              correlation_pair, feature_registry, limit_fill_evidence)
from utils.opening_schedule import ScheduleConfig

START = pd.Timestamp("2026-09-01 09:30", tz="Asia/Shanghai")


def fixture(seconds=None):
    seconds = np.arange(0, 81, .5) if seconds is None else np.asarray(seconds)
    n = len(seconds)
    return pd.DataFrame({
        "Datetime": [START+pd.Timedelta(seconds=float(s)) for s in seconds],
        "LastPrice": 100., "BidPrice1": 99.8, "AskPrice1": 100.2,
        "BidVolume1": 20., "AskVolume1": 30., "Volume": np.arange(n)+100,
        "source_row": np.arange(n),
    })


class CausalFeatureTests(unittest.TestCase):
    def test_missing_and_observed_openings_share_timezone_schema(self):
        populated=build_session_atomic(fixture(),START,"AM","TEST",ICConfig())
        missing=build_session_atomic(fixture().iloc[:0],START,"AM","TEST",ICConfig())
        for name in populated.columns:
            if name.endswith("_time"):
                self.assertEqual(str(populated[name].dtype),"datetime64[us, Asia/Shanghai]")
                self.assertEqual(missing[name].dtype,populated[name].dtype)
                self.assertEqual(pd.concat([missing,populated])[name].dtype,populated[name].dtype)

    def test_original_ofi_example_and_quote_position_sign(self):
        f = fixture([0, .5])
        f["BidPrice1"], f["AskPrice1"] = 100., 100.2
        f.loc[1, ["BidVolume1", "AskVolume1"]] = [35., 20.]
        x = SessionData(f, START, ICConfig()).features(.5)
        self.assertAlmostEqual(x["F03_NOFI_latest"], 25/52.5)
        self.assertAlmostEqual(x["F07_QuotePosition"], 1.)

    def test_ofi_quote_replacements_use_correct_old_or_new_queue(self):
        for side, delta, expected in [("BidPrice1", .2, 20), ("BidPrice1", -.2, -20),
                                      ("AskPrice1", .2, 30), ("AskPrice1", -.2, -30)]:
            f = fixture([0,.5])
            f.loc[1,side] += delta
            data = SessionData(f,START,ICConfig())
            self.assertEqual(data.ofi[1],expected)

    def test_last_instant_shock_not_given_fake_history_weight(self):
        f = fixture()
        f.loc[f.Datetime.eq(START+pd.Timedelta(seconds=10)), "LastPrice"] = 100.4
        x = SessionData(f, START, ICConfig()).features(10)
        self.assertEqual(x["M01_MADeviation_h5s"], 2.)
        self.assertEqual(x["F05_LastChange"], 2.)
        self.assertEqual(x["F06_PathEfficiency_h5s"], 1.)

    def test_weighting_order_distinguishes_a03_and_b01(self):
        f = fixture([0, 1, 2])
        f.BidVolume1 = [100,100,100]
        f.AskVolume1 = [10,190,190]
        cfg = ICConfig(history_seconds=(2,))
        x = SessionData(f,START,cfg).features(2)
        self.assertAlmostEqual(x["A03_QIMean_h2s"], ((90/110)+(-90/290))/2)
        self.assertAlmostEqual(x["B01_MeanQueueImbalance_h2s"],0.)

    def test_zero_activity_has_real_zero_and_undefined_ratios(self):
        f=fixture(); f.Volume=100
        x=SessionData(f,START,ICConfig()).features(10)
        for name in ["F02_QIChange","F03_NOFI_latest","F04_QuoteShift","F05_LastChange",
                     "F05_Momentum_h5s","F06_PathEfficiency_h5s","C01_PriceRV_h5s",
                     "D01_VolumeRate_h5s","A07_OFIActivity_h5s"]:
            self.assertEqual(x[name],0.,name)
        for name in ["F08_SignedVolume_h5s","A07_OFIDirection_h5s"]:
            self.assertTrue(np.isnan(x[name]),name)

    def test_future_mutations_and_truncation_leave_all_features_unchanged(self):
        f=fixture()
        f.LastPrice += (np.arange(len(f))%7)*.2
        f.BidVolume1 += np.arange(len(f))%11
        cfg=ICConfig()
        a=SessionData(f,START,cfg).features(20)
        prefix=f.loc[f.Datetime.le(START+pd.Timedelta(seconds=20))]
        b=SessionData(prefix,START,cfg).features(20)
        changed=f.copy()
        changed.loc[changed.Datetime.gt(START+pd.Timedelta(seconds=20)),"LastPrice"]=1000.
        c=SessionData(changed,START,cfg).features(20)
        for name in feature_registry(cfg).factor:
            np.testing.assert_allclose([a[name],a[name]],[b[name],c[name]],equal_nan=True,err_msg=name)

    def test_gaps_reset_windows_and_do_not_create_increment_evidence(self):
        f=fixture([0,.5,1,10,10.5])
        data=SessionData(f,START,ICConfig())
        x=data.features(10)
        self.assertTrue(np.isnan(x["F02_QIChange"]))
        self.assertTrue(np.isnan(x["F03_NOFI_latest"]))
        self.assertTrue(np.isnan(x["F05_Momentum_h5s"]))
        self.assertEqual(x["coverage_price_h5s"],0)
        self.assertTrue(np.isfinite(x["F01_QI"]))

    def test_negative_volume_resets_flow_without_destroying_price_history(self):
        f=fixture(); f.loc[f.Datetime.ge(START+pd.Timedelta(seconds=9)),"Volume"]-=100
        x=SessionData(f,START,ICConfig()).features(10)
        self.assertTrue(np.isnan(x["F08_SignedVolume_h5s"]))
        self.assertTrue(np.isnan(x["D01_VolumeRate_h5s"]))
        self.assertEqual(x["F05_Momentum_h5s"],0.)

    def test_exact_window_threshold_and_warmup_are_explicit(self):
        data=SessionData(fixture(),START,ICConfig())
        self.assertTrue(np.isnan(data.features(7.5)["M01_MADeviation_h10s"]))
        self.assertTrue(np.isfinite(data.features(8)["M01_MADeviation_h10s"]))
        self.assertEqual(data.features(8)["coverage_price_h10s"],8.)

    def test_distinct_arrival_and_decision_labels_use_timestamps(self):
        f=fixture([0,.4,1.,1.4,2.,2.4,3.,3.4])
        f.LastPrice=[100,100.2,100.4,100.6,100.8,101,101.2,101.4]
        y=SessionData(f,START,ICConfig()).labels(0)
        self.assertEqual(y["arrival_delay_seconds"],1.)
        self.assertEqual(y["y_decision_1s"],2.)
        self.assertEqual(y["y_arrival_1s"],2.)
        self.assertEqual(y["y_arrival_1s_end_time"],START+pd.Timedelta(seconds=2))

    def test_too_late_future_and_gap_are_missing_not_zero(self):
        f=fixture([0,.5,3,3.5])
        y=SessionData(f,START,ICConfig()).labels(0)
        self.assertEqual(y["y_decision_1s_status"],"LATE_FUTURE")
        self.assertTrue(np.isnan(y["y_decision_1s"]))
        self.assertEqual(y["y_decision_3s_status"],"GAPPED_OR_INVALID_FUTURE")

    def test_quote_only_invalid_does_not_erase_price_or_queue_features(self):
        f=fixture(); f.loc[f.Datetime.eq(START+pd.Timedelta(seconds=10)),"AskPrice1"]=0
        x=SessionData(f,START,ICConfig()).features(10)
        self.assertTrue(np.isnan(x["F07_QuotePosition"]))
        self.assertTrue(np.isnan(x["F03_NOFI_latest"]))
        self.assertEqual(x["F05_Momentum_h5s"],0.)
        self.assertTrue(np.isfinite(x["F01_QI"]))

    def test_empty_opening_retains_scheduled_tasks_without_fake_values(self):
        rows=build_session_atomic(fixture().iloc[:0],START,"AM","TEST",ICConfig())
        self.assertEqual(len(rows),250)
        self.assertTrue(rows.F01_QI.isna().all())
        self.assertTrue(rows.y_decision_1s.isna().all())


class CorrelationAndEvidenceTests(unittest.TestCase):
    def test_attachment_example(self):
        n,ic,rank,status=correlation_pair([-.8,-.4,0,.4,.8],[-2,0,-1,1,3],3)
        self.assertEqual(n,5);self.assertEqual(status,"ok")
        self.assertAlmostEqual(ic,.9041944301794651)
        self.assertAlmostEqual(rank,.9)

    def test_ties_are_average_rank_and_zero_returns_remain(self):
        n,_,rank,status=correlation_pair([0,1,1,2],[0,0,1,1],3)
        self.assertEqual(n,4);self.assertEqual(status,"ok")
        self.assertAlmostEqual(rank,1/np.sqrt(2))

    def test_constants_and_pairwise_missing(self):
        self.assertEqual(correlation_pair([1,1,1],[0,1,2],3)[3],"CONSTANT_FACTOR")
        self.assertEqual(correlation_pair([0,1,2],[0,0,0],3)[3],"CONSTANT_LABEL")
        self.assertEqual(correlation_pair([0,1,np.nan],[0,1,2],3)[3],"NOT_ENOUGH_PAIRS")

    def test_confirmed_quote_and_trade_evidence_are_distinct(self):
        self.assertEqual(limit_fill_evidence(1,100,100.2,99.8,100,0),(True,"quote"))
        self.assertEqual(limit_fill_evidence(1,100,99.8,99.8,100.2,1),(True,"lastprice"))
        self.assertEqual(limit_fill_evidence(1,100,100,99.8,100.2,1),(False,"none"))
        self.assertEqual(limit_fill_evidence(-1,100,99.8,100,100.2,0),(True,"quote"))
        self.assertEqual(limit_fill_evidence(-1,100,100.2,99.8,100.2,1),(True,"lastprice"))
        self.assertEqual(limit_fill_evidence(1,100,99.6,99.6,99.8,1),(True,"both_visible"))


if __name__=="__main__":
    unittest.main()
