import unittest
import numpy as np
import pandas as pd
from utils.opening_horizon import STRATEGIES, summarize_orders, switch_decision


def fixture():
    rows = []
    # Unequal daily task counts must not turn into unequal date weights.
    for date, count, saving in [('2026-08-19', 1, .1), ('2026-08-20', 3, .5)]:
        for nominal in range(count):
            for side in [-1, 1]:
                for strategy in STRATEGIES:
                    rows.append(dict(trade_date=date, nominal_second=nominal, direction=side,
                        strategy=strategy, fold='1', cost_bp=1-saving if strategy == 'fixed_1s' else 1.,
                        elapsed_seconds=1. if strategy == 'M' else 3.5,
                        fill_kind='market' if strategy == 'M' else 'limit',
                        fill_stage='immediate' if strategy == 'M' else 'initial'))
    return pd.DataFrame(rows)


class HorizonComparisonTests(unittest.TestCase):
    def test_positive_supported_improvement_switches(self):
        result = switch_decision(np.full(18, .02))
        self.assertTrue(result['passes'])
        self.assertEqual(result['selected'], 'fixed_1s')

    def test_positive_mean_with_uncertainty_keeps_baseline(self):
        result = switch_decision(np.array([-1., 1.02]*9))
        self.assertGreater(result['mean_saving_bp'], 0)
        self.assertLess(result['low_bp'], 0)
        self.assertEqual(result['selected'], 'legacy_3s')

    def test_zero_and_negative_do_not_switch(self):
        for value in [0., -.01]:
            self.assertFalse(switch_decision(np.full(18, value))['passes'])

    def test_invalid_daily_inputs_rejected(self):
        for values in [[], [1, np.nan], [np.inf]]:
            with self.assertRaises(ValueError): switch_decision(values)

    def test_equal_dates_and_delayed_initial_fill(self):
        daily, summary, paired, choice = summarize_orders(fixture())
        row = summary[summary.minutes.eq(60) & summary.side.eq('both') & summary.strategy.eq('fixed_1s')].iloc[0]
        self.assertAlmostEqual(row.cost_bp, .7)
        self.assertAlmostEqual(row.saving_vs_legacy_bp, .3)
        self.assertEqual(row.initial_limit, 1)
        self.assertEqual(row.early_fill, 0)
        self.assertEqual(row.tasks, 4)

    def test_unpaired_or_duplicate_data_rejected(self):
        base = fixture()
        for broken in [base.iloc[1:], pd.concat([base, base.iloc[[0]]]), base.assign(direction=1)]:
            with self.assertRaises(ValueError): summarize_orders(broken)

    def test_fill_mix_excludes_misclassified_orders(self):
        broken = fixture(); broken.loc[0, 'fill_stage'] = 'unknown'
        with self.assertRaises(ValueError): summarize_orders(broken)
