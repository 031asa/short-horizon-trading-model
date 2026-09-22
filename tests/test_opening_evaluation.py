"""Scientific checks for comparable samples and equal-day aggregation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_opening_ic import has_opening_data
from utils.ic_statistics import evaluate_all, summarize_daily, pair_matrix, STATUS
from utils.opening_ic import ICConfig, correlation_pair
import tempfile
import numpy as np
import pandas as pd


class EvaluationTests(unittest.TestCase):
    def test_exclude_entirely_missing_opening_but_keep_partial_opening(self):
        start=pd.Timestamp("2026-07-20 09:30",tz="Asia/Shanghai")
        frame=pd.DataFrame({"Datetime":[start-pd.Timedelta(seconds=1),start+pd.Timedelta(seconds=60)]})
        self.assertFalse(has_opening_data(frame,start))
        frame.loc[2]=start+pd.Timedelta(seconds=56)
        self.assertTrue(has_opening_data(frame,start))

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
        for row in rows:
            for anchor in ('decision','arrival'):
                for u in range(1,6):
                    row[f'y_{anchor}_incremental_{u}s']=row[f'y_{anchor}_{u}s']-(row[f'y_{anchor}_{u-1}s'] if u>1 else 0)
        registry=pd.DataFrame([dict(factor="factor",family_id='F01',role="direction", history_seconds=0,evaluate=True)])
        with tempfile.TemporaryDirectory() as folder:
            evaluate_all(pd.DataFrame(rows),registry,ICConfig(minimum_pairs=3,horizons_seconds=(1,2,3,4,5)),folder)
            result=pd.read_parquet(Path(folder)/'daily_ic.parquet')
        result=result.loc[result.label_type.eq('cumulative') & result.target.eq('signed')]
        common=result.loc[result.pair_set.eq("common_observations_horizons")]
        self.assertTrue(common.loc[common.anchor.eq("decision"),"n_pairs"].eq(4).all())
        self.assertTrue(common.loc[common.anchor.eq("arrival"),"n_pairs"].eq(5).all())
        own=result.loc[result.pair_set.eq("own") & result.anchor.eq("decision")
                       & result.observation_seconds.eq(1) & result.horizon_seconds.eq(1)].iloc[0]
        self.assertEqual(own.n_pairs,6)
        self.assertAlmostEqual(own.zero_return_share,1/6)
        self.assertAlmostEqual(own.rank_ic,1)

    def test_all_dates_are_equal_weighted_without_dataset_partition(self):
        rows=[]
        for date,pairs,corr in [("a",20,1.),("b",50,-1.),("c",50,.5)]:
            rows.append(dict(trade_date=date,session="AM",factor="factor",role="direction",
                             history_seconds=0,observation_seconds=1,horizon_seconds=1,
                             anchor="decision",target="signed",pair_set="own",ic=corr,rank_ic=corr,
                             n_pairs=pairs,scheduled_coverage=1.,zero_return_share=0.))
        summary=summarize_daily(pd.DataFrame(rows)).set_index('phase')
        self.assertEqual(summary.index.tolist(),['all_sample'])
        self.assertAlmostEqual(summary.loc['all_sample','mean_ic'],1/6)
        self.assertAlmostEqual(summary.loc['all_sample','mean_rank_ic'],1/6)
        self.assertAlmostEqual(summary.loc['all_sample','positive_day_share'],2/3)

    def test_matrix_correlations_match_scalar_pairwise_ranking(self):
        rng=np.random.default_rng(931)
        x=rng.integers(-2,3,size=(50,13)).astype(float)
        y=rng.integers(-3,4,size=(50,7)).astype(float)
        x[rng.random(x.shape)<.15]=np.nan;y[rng.random(y.shape)<.1]=np.nan
        x[:,0]=1.;y[:,0]=0.;x[:45,1]=np.nan
        n,ic,rank,status,zero=pair_matrix(x,y,20)
        for i in range(x.shape[1]):
            for j in range(y.shape[1]):
                expected=correlation_pair(x[:,i],y[:,j],20)
                self.assertEqual(n[i,j],expected[0]);self.assertEqual(STATUS[status[i,j]],expected[3])
                np.testing.assert_allclose([ic[i,j],rank[i,j]],expected[1:3],rtol=1e-12,atol=1e-12,equal_nan=True)


if __name__ == "__main__":
    unittest.main()
