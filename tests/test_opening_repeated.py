import unittest
import numpy as np
from tests.test_opening_two_stage import session
from utils.opening_repeated import simulate_repeated
from utils.opening_two_stage import simulate


class RepeatedTests(unittest.TestCase):
    def test_single_decision_matches_reference(self):
        rng = np.random.default_rng(31)
        for step in [.5, 1.]:
            times = np.arange(0, 13, step); n = len(times)
            for _ in range(12):
                p = 100+np.cumsum(rng.integers(-9, 10, n))
                d = session(times=times, price=p, bid=p-2, ask=p+2, volume=np.cumsum(rng.integers(0, 4, n)))
                for side in [-1, 1]:
                    for signal in [-1, 0, 1]:
                        old = simulate(d, 0, side, 'C_limit_first', signal, c_limit_offset_ticks=19)
                        new = simulate_repeated(d, 0, side, {3:signal}, decisions=(3,))
                        self.assertEqual(old['status'], new['status'])
                        for k in ['fill_seconds','fill_ticks','cost_bp','fill_kind','pending_cancelled']:
                            self.assertEqual(old[k],new[k])
                        self.assertEqual(old['fill_stage'].replace('signal','signal3'),new['fill_stage'])

    def test_missing_later_signal_keeps_order(self):
        d = session()
        r = simulate_repeated(d,0,1,{3:0,6:None,9:None})
        self.assertEqual(r['missing_decisions'],[6,9])
        self.assertEqual((r['fill_stage'],r['fill_seconds']),('deadline',11))
        self.assertEqual(r['submissions'],2)

    def test_new_signal_at_six_uses_current_quote_after_delay(self):
        d = session(); d.a[14] = 108
        r = simulate_repeated(d,0,1,{3:0,6:1,9:-1})
        self.assertEqual((r['fill_seconds'],r['fill_ticks'],r['fill_stage']),(7,108,'signal6'))
        self.assertEqual(r['decisions_used'],[3,6])

    def test_old_limit_has_priority_at_new_arrival(self):
        for side in [-1,1]:
            d=session()
            if side==1:d.a[14]=81;d.b[14]=80
            else:d.b[14]=119;d.a[14]=120
            r=simulate_repeated(d,0,side,{3:0,6:side,9:side})
            self.assertEqual((r['fill_seconds'],r['fill_ticks'],r['fill_stage']),(7,100-side*19,'initial'))
            self.assertTrue(r['pending_cancelled'])

    def test_pending_nine_second_limit_does_not_cancel_deadline(self):
        d=session(times=np.arange(13)); d.p[9:]=101
        r=simulate_repeated(d,0,1,{3:0,6:0,9:0})
        submissions=[e for e in r['trace'] if e['event']=='submit']
        self.assertEqual([(e['time'],e['arrival_index']) for e in submissions],[(0,2),(9,11),(10,12)])
        self.assertEqual((r['fill_seconds'],r['fill_stage']),(12,'deadline'))

    def test_nine_second_limit_can_fill_after_deadline_submission(self):
        d=session(times=np.arange(13)); d.p[9:]=101; d.a[11]=82;d.b[11]=81
        r=simulate_repeated(d,0,1,{3:0,6:0,9:0})
        self.assertEqual((r['fill_seconds'],r['fill_stage'],r['fill_ticks']),(11,'signal9',82))
        self.assertTrue(r['deadline_submitted'])
        self.assertTrue(r['pending_cancelled'])

    def test_nine_second_market_pending_does_not_send_second_market(self):
        d=session(times=np.arange(13))
        r=simulate_repeated(d,0,1,{3:0,6:0,9:1})
        self.assertEqual((r['fill_seconds'],r['fill_stage']),(11,'signal9'))
        self.assertFalse(r['deadline_submitted'])
        self.assertEqual(len([e for e in r['trace'] if e['event']=='submit' and e['kind']=='market']),1)

    def test_fill_at_decision_snapshot_precedes_signal(self):
        d=session();d.a[12]=81;d.b[12]=80
        r=simulate_repeated(d,0,1,{3:0,6:1,9:1})
        self.assertEqual(r['fill_seconds'],6)
        self.assertEqual(r['decisions_used'],[3])

    def test_price_cross_requires_positive_volume_and_strict_cross(self):
        d=session();d.p[15]=80;d.dv[15]=1
        r=simulate_repeated(d,0,1,{3:0,6:0,9:0})
        self.assertEqual((r['fill_seconds'],r['fill_mechanism']),(7.5,'lastprice'))
        d.dv[15]=0
        self.assertEqual(simulate_repeated(d,0,1,{3:0,6:0,9:0})['fill_stage'],'deadline')

    def test_invalid_gap_not_silently_filled(self):
        d=session();d.time_edge[13]=False
        r=simulate_repeated(d,0,1,{3:0,6:0,9:0})
        self.assertEqual((r['status'],r['reason']),('invalid','gap_or_nonadjacent_snapshot'))
