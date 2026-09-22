"""Scientific checks for comparable samples and equal-day aggregation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_opening_ic import daily_evaluation, summarize_daily
from utils.opening_ic import ICConfig
import numpy as np
import pandas as pd


class EvaluationTests(unittest.TestCase):
    def test_shared_tasks_exclude_missing_factor_or_any_horizon_without_dropping_real_zero(self):
        rows=[]
        for observation in range(1,6):
            for task in range(6):
                row=dict(trade_date="2026-09-01", session="AM", task_time=task,
                         observation_seconds=observation, factor=float(task))
                for anchor in ("decision", "arrival"):
                    for u in range(1,6):
                        row[f"y_{anchor}_{u}s"]=float(task)
                if observation==4 and task==0:
                    row["factor"]=np.nan
                if observation==5 and task==1:
                    row["y_decision_5s"]=np.nan
                rows.append(row)
        registry=pd.DataFrame([dict(factor="factor", role="direction", history_seconds=0)])
        result=daily_evaluation(pd.DataFrame(rows),registry,ICConfig(minimum_pairs=3))
        common=result.loc[result.pair_set.eq("common_observations_horizons")]
        self.assertTrue(common.loc[common.anchor.eq("decision"),"n_pairs"].eq(4).all())
        self.assertTrue(common.loc[common.anchor.eq("arrival"),"n_pairs"].eq(5).all())
        own=result.loc[result.pair_set.eq("own") & result.anchor.eq("decision")
                       & result.observation_seconds.eq(1) & result.horizon_seconds.eq(1)].iloc[0]
        self.assertEqual(own.n_pairs,6)
        self.assertAlmostEqual(own.zero_return_share,1/6)
        self.assertAlmostEqual(own.rank_ic,1)

    def test_day_means_do_not_weight_large_days_more_or_use_validation_to_compute_development(self):
        rows=[]
        for date,pairs,corr in [("a",20,1.),("b",50,-1.),("c",50,.5)]:
            rows.append(dict(trade_date=date,session="AM",factor="factor",role="direction",
                             history_seconds=0,observation_seconds=1,horizon_seconds=1,
                             anchor="decision",target="signed",pair_set="own",ic=corr,rank_ic=corr,
                             n_pairs=pairs,scheduled_coverage=1.,zero_return_share=0.))
        summary=summarize_daily(pd.DataFrame(rows),["a","b"],["c"]).set_index("phase")
        self.assertEqual(summary.loc["development","mean_ic"],0.)
        self.assertEqual(summary.loc["development","mean_rank_ic"],0.)
        self.assertEqual(summary.loc["development","positive_day_share"],.5)
        self.assertEqual(summary.loc["validation","mean_rank_ic"],.5)
        self.assertAlmostEqual(summary.loc["all_exploratory","mean_rank_ic"],1/6)


if __name__ == "__main__":
    unittest.main()
