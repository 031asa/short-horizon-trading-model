import unittest
import numpy as np
import pandas as pd
from utils.opening_initial_offset import safe_initial_sweep, OFFSETS, STAGES, select_offset, daily_summary
from utils.opening_two_stage import simulate
from tests.test_opening_two_stage import session


class InitialOffsetTests(unittest.TestCase):
    def check_equivalent(self, d, t=0):
        for side in [-1,1]:
            for signal in [-1,0,1]:
                indexes,prices,stages=safe_initial_sweep(d,t,side,signal)
                for j,k in enumerate(OFFSETS):
                    r=simulate(d,t,side,'C_limit_first',signal,c_limit_offset_ticks=int(k),c_signal_offset_ticks=19)
                    self.assertEqual(r['status'],'filled')
                    self.assertEqual((float(d.times[indexes[j]]),prices[j],stages[j]),
                                     (r['fill_seconds'],r['fill_ticks'],STAGES[(r['fill_kind'],r['fill_stage'])]))

    def test_random_paths_match_event_engine(self):
        rng=np.random.default_rng(20260928)
        for step in [.5,1.]:
            times=np.arange(0,13,step);n=len(times)
            for _ in range(5):
                p=100+np.cumsum(rng.integers(-7,8,n))
                self.check_equivalent(session(times=times,price=p,bid=p-1,ask=p+1,volume=np.cumsum(rng.integers(0,4,n))))

    def test_same_limit_skip_and_deadline(self):
        self.check_equivalent(session())

    def test_price_cross_and_old_order_priority(self):
        for when in [3.,3.5,4.,10.,10.5,11.]:
            p=np.full(25,100.);p[int(when*2)]=70
            self.check_equivalent(session(price=p,volume=np.arange(25)))

    def test_nonzero_task_start(self):
        self.check_equivalent(session(times=np.arange(.1,12.6,.5)),t=.1)

    def test_tie_break_nearest19_then_shallower(self):
        curve=pd.DataFrame({'k':[20,18,0],'cost_bp':[1,1,2],'early_contribution_bp':[.2,.2,.1]})
        self.assertEqual(select_offset(curve,'early'),18)
        self.assertEqual(select_offset(curve,'total'),18)

    def test_no_early_fill_is_zero_contribution_not_fake_conditional(self):
        rows=[]
        for side in [-1,1]:
            rows.append(dict(trade_date='2026-09-01',nominal_second=0,direction=side,k=19,
                             elapsed_seconds=3.5,stage=0,cost_bp=1.,market_cost_bp=2.))
        d=daily_summary(pd.DataFrame(rows))
        self.assertTrue(d.early_contribution_bp.eq(0).all())
        self.assertTrue(d.early_conditional_saving_bp.isna().all())
        self.assertTrue(d.later_contribution_bp.eq(1).all())
        self.assertTrue(d.initial_limit.eq(1).all())

    def test_at_three_seconds_counts_early_and_signed_losses_preserved(self):
        rows=[]
        for side in [-1,1]:
            rows.extend([dict(trade_date='2026-09-01',nominal_second=0,direction=side,k=19,elapsed_seconds=3.,stage=0,cost_bp=3.,market_cost_bp=2.),
                         dict(trade_date='2026-09-01',nominal_second=1,direction=side,k=19,elapsed_seconds=4.,stage=3,cost_bp=4.,market_cost_bp=2.)])
        d=daily_summary(pd.DataFrame(rows))
        self.assertTrue(d.early_fill.eq(.5).all())
        self.assertTrue(d.early_contribution_bp.eq(-.5).all())
        self.assertTrue(d.later_contribution_bp.eq(-1).all())
        np.testing.assert_allclose(d.early_fill*d.early_conditional_saving_bp,d.early_contribution_bp)
