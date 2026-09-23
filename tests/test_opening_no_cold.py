import unittest
import numpy as np
import pandas as pd
from utils.opening_ic import SessionData,ICConfig
from utils.opening_schedule import ScheduleConfig,build_task_schedule
from utils.opening_task_window import bounded_features,ALL
from utils.opening_small_combinations import design_stats

START=pd.Timestamp('2026-09-01 09:30',tz='Asia/Shanghai')
def fixture():
    sec=np.arange(0,20,.5);n=len(sec)
    return pd.DataFrame(dict(Datetime=[START+pd.Timedelta(seconds=x) for x in sec],
      LastPrice=100+(np.arange(n)%4)*.2,BidPrice1=99.8,AskPrice1=100.8,
      BidVolume1=20+(np.arange(n)%5),AskVolume1=30-(np.arange(n)%7),Volume=np.arange(n)*5+100,source_row=np.arange(n)))

class NoColdTests(unittest.TestCase):
    def test_schedule_starts_at_open_without_warmup(self):
        s=build_task_schedule(START,ScheduleConfig(cold_start_seconds=0,observation_seconds=(1.,3.,10.)))
        self.assertEqual(len(s),180);self.assertEqual(s.task_elapsed_seconds.min(),0);self.assertEqual(s.decision_elapsed_seconds.min(),1)

    def test_pre_task_and_future_do_not_change_features(self):
        f=fixture();d=SessionData(f,START,ICConfig());x=bounded_features(d,5,3)
        altered=f.copy();mask=(f.Datetime<START+pd.Timedelta(seconds=5))|(f.Datetime>START+pd.Timedelta(seconds=8))
        altered.loc[mask,'BidVolume1']=90000;altered.loc[mask,'LastPrice']=110;altered.loc[mask,'Volume']=999999
        y=bounded_features(SessionData(altered,START,ICConfig()),5,3)
        np.testing.assert_allclose([x[n] for n in ALL],[y[n] for n in ALL],equal_nan=True)
        self.assertGreaterEqual(x['used_start'],5);self.assertLessEqual(x['source_seconds'],8)

    def test_first_cumulative_volume_is_not_counted(self):
        f=fixture();a=bounded_features(SessionData(f,START,ICConfig()),0,3)
        f.Volume+=1000000;b=bounded_features(SessionData(f,START,ICConfig()),0,3)
        self.assertEqual(a['D01_LogVolumeRate'],b['D01_LogVolumeRate'])

    def test_latest_ofi_cannot_cross_task_boundary(self):
        f=fixture();f=f[f.Datetime!=START+pd.Timedelta(seconds=.5)].reset_index(drop=True)
        x=bounded_features(SessionData(f,START,ICConfig()),.6,.4)
        self.assertTrue(np.isnan(x['F03_OFI_latest']))
        self.assertTrue(np.isfinite(x['F07_QuotePosition']))

    def test_short_window_does_not_borrow_old_snapshot(self):
        f=fixture();f.Datetime+=pd.Timedelta(seconds=.2)
        x=bounded_features(SessionData(f,START,ICConfig()),5,1)
        self.assertEqual(x['snapshot_count'],2);self.assertTrue(np.isnan(x['F03_OFI_window']))
        self.assertTrue(np.isfinite(x['F03_OFI_latest']))

    def test_no_linear_duplicate_enters_basis(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(80,2));y=np.column_stack([x,x[:,0]+2*x[:,1]])
        self.assertFalse(design_stats(y,np.repeat(np.arange(8),10))['ok'])
        self.assertTrue(design_stats(x,np.repeat(np.arange(8),10))['ok'])

if __name__=='__main__':unittest.main()
