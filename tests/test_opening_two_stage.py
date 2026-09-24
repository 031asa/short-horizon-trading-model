import unittest
import numpy as np
import pandas as pd
from utils.opening_ic import ICConfig, SessionData
from utils.opening_two_stage import simulate, date_block_interval, POLICIES


def session(times=None, price=None, bid=None, ask=None, volume=None):
    times = np.arange(0, 12.5, .5) if times is None else np.asarray(times, float)
    n = len(times)
    def arr(x, default):
        return np.full(n, default) if x is None else np.asarray(x, float)
    start = pd.Timestamp('2026-09-01 09:30', tz='Asia/Shanghai')
    frame = pd.DataFrame(dict(Datetime=start+pd.to_timedelta(times, unit='s'),
        LastPrice=arr(price, 100)*.2, BidPrice1=arr(bid, 98)*.2,
        AskPrice1=arr(ask, 102)*.2, BidVolume1=np.ones(n)*10,
        AskVolume1=np.ones(n)*10, Volume=arr(volume, 0), source_row=np.arange(n)))
    return SessionData(frame, start, ICConfig())


class TwoStageTests(unittest.TestCase):
    def test_c_passive_offset_both_sides_and_stages(self):
        for side in (1, -1):
            d = session(); d.p[6:] = 100+side
            r = simulate(d, 0, side, POLICIES[2], 0, c_limit_offset_ticks=1)
            orders = [e for e in r['trace'] if e['event']=='submit' and e['kind']=='limit']
            self.assertEqual([e['limit_ticks'] for e in orders], [100-side, 100])
            self.assertEqual(r['fill_stage'], 'deadline')

    def test_c_passive_offset_fills_at_offset_limit(self):
        for side in (1, -1):
            d = session()
            (d.a if side==1 else d.b)[4] = 100-side
            r = simulate(d, 0, side, POLICIES[2], 0, c_limit_offset_ticks=1)
            self.assertEqual((r['fill_ticks'], r['cost_ticks']), (100-side, -1))

    def test_c_offset_does_not_change_ab(self):
        for policy in POLICIES[:2]:
            self.assertEqual(simulate(session(),0,1,policy,1),
                             simulate(session(),0,1,policy,1,c_limit_offset_ticks=1))

    def test_delay_and_common_deadline(self):
        for policy in POLICIES:
            r = simulate(session(), 0, 1, policy, 0)
            self.assertEqual((r['fill_seconds'], r['fill_stage']), (11, 'deadline'))
            submitted = [e for e in r['trace'] if e['event']=='submit']
            self.assertEqual(submitted[-1]['time'], 10)
            arrivals = [e for e in r['trace'] if e['event']=='arrival']
            self.assertEqual(arrivals[0]['time'], 4 if policy=='B_observe_first' else 1)

    def test_no_matching_before_activation(self):
        ask = np.full(25, 102.); ask[1] = 100
        r = simulate(session(ask=ask), 0, 1, POLICIES[0], 0)
        self.assertEqual(r['fill_stage'], 'deadline')

    def test_marketable_limit_settles_limit(self):
        ask = np.full(25, 102.); ask[2] = 99
        r = simulate(session(ask=ask), 0, 1, POLICIES[0], 0)
        self.assertEqual((r['fill_ticks'], r['fill_seconds'], r['fill_mechanism']), (100, 1, 'quote'))

    def test_lastprice_strict_cross_requires_volume(self):
        price = np.full(25, 100.); price[3] = 99
        v = np.zeros(25); v[3:] = 1
        for volume, expected in [(v, 'lastprice'), (np.zeros(25), 'opposite_quote')]:
            r = simulate(session(price=price, volume=volume), 0, 1, POLICIES[0], 0)
            self.assertEqual(r['fill_mechanism'], expected)
        r = simulate(session(volume=np.arange(25)), 0, 1, POLICIES[0], 0)
        self.assertEqual(r['fill_kind'], 'market')

    def test_both_visible(self):
        price = np.full(25, 100.); price[2] = 99
        ask = np.full(25, 102.); ask[2] = 100
        r = simulate(session(price=price, ask=ask, volume=np.arange(25)), 0, 1, POLICIES[0], 0)
        self.assertEqual(r['fill_mechanism'], 'both_visible')

    def test_sell_mirror(self):
        buy = session(); sell = session()
        buy.a[4] = 100; sell.b[4] = 100
        for d, side in [(buy, 1), (sell, -1)]:
            r = simulate(d, 0, side, POLICIES[0], 0)
            self.assertEqual((r['cost_bp'], r['fill_seconds']), (0, 2))

    def test_existing_fill_before_signal(self):
        d = session(); d.a[6] = 100
        r = simulate(d, 0, 1, POLICIES[2], 1)
        self.assertEqual(r['fill_seconds'], 3)
        self.assertFalse(r['signal_used'])
        self.assertEqual(r['submissions'], 1)

    def test_old_limit_fills_while_market_pending(self):
        d = session(); d.a[7] = 100
        r = simulate(d, 0, 1, POLICIES[2], 1)
        self.assertEqual((r['fill_kind'], r['fill_seconds']), ('limit', 3.5))
        self.assertTrue(r['pending_cancelled'])
        self.assertEqual(r['submissions'], 2)

    def test_old_limit_priority_on_replacement_arrival(self):
        d = session(); d.a[8] = 100
        r = simulate(d, 0, 1, POLICIES[2], 1)
        self.assertEqual((r['fill_kind'], r['fill_seconds'], r['fill_ticks']), ('limit', 4, 100))
        self.assertTrue(r['pending_cancelled'])

    def test_market_arrival_uses_arrival_quote(self):
        d = session(); d.a[8] = 105
        for policy in POLICIES[1:]:
            r = simulate(d, 0, 1, policy, 1)
            self.assertEqual((r['fill_seconds'], r['fill_ticks'], r['fill_stage']), (4, 105, 'signal'))

    def test_flat_or_favourable_uses_limit(self):
        for side in [-1, 1]:
            for signal in [0, -side]:
                r = simulate(session(), 0, side, POLICIES[1], signal)
                self.assertEqual(r['signal_action'], 'limit')

    def test_limit_price_locked_at_decision(self):
        d = session(); d.p[6] = 103; d.p[7:9] = 104
        r = simulate(d, 0, 1, POLICIES[1], -1)
        self.assertEqual((r['fill_ticks'], r['fill_seconds']), (103, 4))

    def test_unchanged_limit_keeps_order(self):
        r = simulate(session(), 0, 1, POLICIES[2], 0)
        self.assertTrue(r['skipped_same_price'])
        self.assertEqual(r['submissions'], 2)  # initial and deadline
        self.assertEqual(r['replacements'], 1)

    def test_no_updates_at_six_or_nine(self):
        for policy in POLICIES:
            r = simulate(session(), 0, 1, policy, 0)
            self.assertTrue(all(e['time'] in [0, 3, 10] for e in r['trace'] if e['event']=='submit'))

    def test_deadline_limit_fill_priority(self):
        for index in [20, 21, 22]:
            d = session(); d.a[index] = 100
            r = simulate(d, 0, 1, POLICIES[0], None)
            self.assertEqual(r['fill_kind'], 'limit')
            self.assertEqual(r['deadline_submitted'], index>20)

    def test_missing_signal_not_flat(self):
        r = simulate(session(), 0, 1, POLICIES[1], None)
        self.assertEqual(r['reason'], 'missing_signal')
        d = session(); d.a[2] = 100
        r = simulate(d, 0, 1, POLICIES[2], None)
        self.assertEqual(r['status'], 'filled')

    def test_no_future_initial_source(self):
        r = simulate(session(times=np.arange(.1,12,.5)), 0, 1, POLICIES[0], 1)
        self.assertEqual(r['reason'], 'initial_no_source')

    def test_stale_source(self):
        r = simulate(session(times=np.arange(0,12,1.)), .6, 1, POLICIES[0], 1)
        self.assertEqual(r['reason'], 'initial_stale_source')

    def test_gap_not_skipped(self):
        d = session(times=np.r_[0,.5,2.,np.arange(2.5,12,.5)])
        self.assertEqual(simulate(d,0,1,POLICIES[0],1)['reason'], 'gap_or_nonadjacent_snapshot')

    def test_volume_reset_not_skipped(self):
        d = session(volume=np.r_[np.arange(5),np.zeros(20)])
        self.assertEqual(simulate(d,0,1,POLICIES[0],0)['reason'], 'volume_reset_or_invalid_edge')

    def test_bad_quote_not_skipped(self):
        d = session(); d.valid['book'][2] = False
        self.assertEqual(simulate(d,0,1,POLICIES[0],0)['reason'], 'invalid_path_snapshot')

    def test_insufficient_tail(self):
        r = simulate(session(times=np.arange(0,10.5,.5)), 0, 1, POLICIES[0], 0)
        self.assertEqual(r['reason'], 'insufficient_arrival_tail')

    def test_fills_unaffected_by_future_perturbation(self):
        d = session(); d.a[2] = 100
        a = simulate(d,0,1,POLICIES[2],1)
        d.p[3:] += 50; d.a[3:] += 50; d.b[3:] += 50
        b = simulate(d,0,1,POLICIES[2],-1)
        self.assertEqual(a,b)

    def test_actions_before_three_unaffected_by_signal(self):
        x = [simulate(session(),0,1,POLICIES[2],s) for s in [-1,1]]
        self.assertEqual(*[[e for e in r['trace'] if e['time']<3] for r in x])

    def test_ablation_equal_after_three_if_surviving(self):
        for signal in [-1,0,1]:
            x=[simulate(session(),0,1,p,signal) for p in POLICIES[1:]]
            self.assertEqual(x[0]['cost_bp'],x[1]['cost_bp'])
            self.assertEqual(x[0]['fill_seconds'],x[1]['fill_seconds'])

    def test_bootstrap_constant(self):
        np.testing.assert_allclose(date_block_interval(np.full(18,.2)), [.2,.2])
