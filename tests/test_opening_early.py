import unittest
import numpy as np
from tests.test_opening_two_stage import session
from utils.opening_repeated import simulate_repeated


class EarlyDecisionTests(unittest.TestCase):
    def test_nonadverse_keeps_initial_limit_and_original_three_second_action(self):
        d=session();d.p[2:]=108
        r=simulate_repeated(d,0,1,{1:-1,3:1},decisions=(3,),early_market_seconds=(1,))
        old=simulate_repeated(d,0,1,{3:1},decisions=(3,))
        for key in ['fill_seconds','fill_ticks','cost_bp','fill_stage']:
            self.assertEqual(r[key],old[key])
        self.assertEqual([e['time'] for e in r['trace'] if e['event']=='submit'],[0,3])

    def test_adverse_early_market_uses_arrival_price_both_directions(self):
        for t in [1,2]:
            for side in [-1,1]:
                d=session();d.a[int(2*(t+1))]=109;d.b[int(2*(t+1))]=91
                r=simulate_repeated(d,0,side,{t:side,3:-side},decisions=(3,),early_market_seconds=(t,))
                self.assertEqual(r['fill_seconds'],t+1)
                self.assertEqual(r['fill_ticks'],109 if side==1 else 91)

    def test_missing_early_signal_falls_back_without_dropping_order(self):
        d=session();r=simulate_repeated(d,0,1,{1:None,3:1},decisions=(3,),early_market_seconds=(1,))
        self.assertEqual((r['fill_seconds'],r['fill_stage']),(4,'signal3'))
        self.assertEqual(r['missing_decisions'],[1])

    def test_initial_order_fills_before_early_replacement_at_same_snapshot(self):
        d=session();d.a[4]=81;d.b[4]=80
        r=simulate_repeated(d,0,1,{1:1,3:1},decisions=(3,),early_market_seconds=(1,))
        self.assertEqual((r['fill_ticks'],r['fill_stage']),(81,'initial'))
        self.assertTrue(r['pending_cancelled'])

    def test_initial_command_can_still_be_pending_at_early_check(self):
        d=session(times=np.arange(13));r=simulate_repeated(d,0,1,{1:1,3:-1},decisions=(3,),early_market_seconds=(1,))
        self.assertEqual((r['status'],r['fill_seconds'],r['fill_stage']),('filled',3,'signal1'))

    def test_market_still_in_flight_at_three_is_not_replaced(self):
        d=session(times=np.arange(13));r=simulate_repeated(d,0,1,{2:1,3:-1},decisions=(3,),early_market_seconds=(2,))
        self.assertEqual((r['fill_seconds'],r['fill_stage']),(4,'signal2'))
        self.assertEqual(len([e for e in r['trace'] if e['event']=='submit']),2)
        self.assertTrue(any(e['event']=='signal_already_market' for e in r['trace']))

    def test_cannot_use_unregistered_early_schedule(self):
        for early in [(0,),(.5,),(1,2),(3,)]:
            with self.assertRaises(ValueError):simulate_repeated(session(),0,1,{3:1},decisions=(3,),early_market_seconds=early)

    def test_stale_early_source_keeps_order_if_the_path_remains_valid(self):
        d=session(times=np.r_[0,.4,np.arange(1.4,12.4,.5)])
        new=simulate_repeated(d,0,1,{1:1,3:1},decisions=(3,),early_market_seconds=(1,))
        old=simulate_repeated(d,0,1,{3:1},decisions=(3,))
        self.assertEqual(new['status'],'filled')
        self.assertEqual(new['missing_decisions'],[1])
        for key in ['fill_seconds','fill_ticks','cost_bp','fill_stage']:self.assertEqual(new[key],old[key])
