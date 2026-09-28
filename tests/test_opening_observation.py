"""Contracts for the fixed-feature, one-second observation-window experiment."""
import unittest
import numpy as np
import pandas as pd
from utils.opening_observation import FEATURES, common_keys, select_window, extension_needed, block_indices
from utils.opening_task_window import bounded_features
from utils.opening_ic import ICConfig, SessionData
from utils.opening_schedule import ScheduleConfig
from tests.test_opening_ic import fixture, START


class ObservationTests(unittest.TestCase):
    def test_shorter_window_within_half_percentage_point(self):
        options = [dict(window=1, accuracy=.60, daily_sd=.02, logloss=.9, C=1),
                   dict(window=2, accuracy=.646, daily_sd=.08, logloss=.8, C=1),
                   dict(window=3, accuracy=.650, daily_sd=.01, logloss=.7, C=1)]
        self.assertEqual(select_window(options)['window'], 2)
        options[1]['accuracy'] = .6449
        self.assertEqual(select_window(options)['window'], 3)

    def test_common_keys_are_task_pairs_not_label_values(self):
        records = []
        for task in range(3):
            for window in [1, 2]:
                row = dict(trade_date='d', nominal_second=task, window=window,
                           cum_1=(-1 if task == 0 else 0), **{f: 1. for f in FEATURES})
                if task == 1 and window == 2:
                    row[FEATURES[-1]] = np.nan
                records.append(row)
        a = pd.DataFrame(records)
        self.assertEqual(common_keys(a, [1, 2]).nominal_second.tolist(), [0, 2])
        a['cum_1'] *= -1
        self.assertEqual(common_keys(a, [1, 2]).nominal_second.tolist(), [0, 2])

    def test_extension_does_not_use_final_or_test_metrics(self):
        decisions = [dict(fold=str(i), options=[dict(window=w, accuracy=.6 + w * .01)
                                               for w in range(1, 6)], test_accuracy=0.) for i in range(4)]
        self.assertTrue(extension_needed(decisions)['extend'])
        decisions.append(dict(fold='final', options=[dict(window=w, accuracy=1 if w == 1 else 0)
                                                    for w in range(1, 6)]))
        for row in decisions:
            row['test_accuracy'] = 1.
        self.assertTrue(extension_needed(decisions)['extend'])

    def test_actual_snapshot_clock_recovers_short_window_span(self):
        frame = fixture(np.arange(.2, 10, .5))
        frame['LastPrice'] += (np.arange(len(frame)) % 3) * .2
        frame['BidVolume1'] += np.arange(len(frame)) % 7
        d = SessionData(frame, START, ICConfig(schedule=ScheduleConfig(cold_start_seconds=0)))
        nominal = bounded_features(d, 0., 1.)
        actual = bounded_features(d, .2, 1.)
        self.assertTrue(np.isnan(nominal['A07_OFIDirection']))
        self.assertTrue(np.isfinite(actual['A07_OFIDirection']))
        self.assertGreaterEqual(actual['used_start'], .2)

    def test_zero_denominators_remain_missing(self):
        d = SessionData(fixture(), START, ICConfig())
        x = bounded_features(d, 1., 1.)
        self.assertTrue(np.isnan(x['A07_OFIDirection']))
        self.assertTrue(np.isnan(x['C05_CountImbalance']))

    def test_regular_half_second_snapshots_give_one_second_latency(self):
        d = SessionData(fixture(), START, ICConfig(horizons_seconds=(1,)))
        label = d.labels(3.)
        self.assertEqual(label['arrival_delay_seconds'], 1.)
        self.assertEqual(label['arrival_time'], label['y_decision_1s_end_time'])

    def test_bootstrap_pairs_share_date_indices(self):
        ix = block_indices(18)
        np.testing.assert_array_equal(ix, block_indices(18))
        self.assertEqual(ix.shape, (5000, 18))
        np.testing.assert_array_equal((ix[:, 1:5] - ix[:, :4]) % 18, 1)


if __name__ == '__main__':
    unittest.main()
