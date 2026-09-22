"""Acceptance of generated artifacts against atomic rows, independent scalar IC, and aliases."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_ic import correlation_pair
from scripts.run_opening_ic import sha

def verify(folder):
    out=Path(folder);a=pd.read_parquet(out/'atomic.parquet') if (out/'atomic.parquet').exists() else pd.read_parquet(ROOT/'sample_snapshot_原子执行总表.parquet')
    s=pd.read_parquet(out/'summary_ic.parquet');r=pd.read_csv(out/'feature_registry.csv')
    assert len(a)==9500 and a.trade_date.nunique()==38 and r.family_id.nunique()==59
    assert s.phase.eq('all_sample').all() and s.horizon_seconds.nunique()==30
    assert len(s)==r.evaluate.sum()*5*30*2*2*2*2
    assert len(r)==len(set(r.factor))
    assert set(r.loc[r.evaluate,'factor'])==set(s.factor)
    for row in r.loc[r.kind.eq('alias')].itertuples():np.testing.assert_allclose(a[row.factor],a[row.alias_of],equal_nan=True)
    p=pd.read_csv(out/'factor_hypotheses.csv');cols=['factor','expected_sign_signed','expected_sign_absolute','prior_version']
    if 'review_origin' in r:
        original=r.loc[r.evaluate & r.review_origin.eq('original')].copy()
        for col in cols[1:]:original[col]=original['original_'+col]
        pd.testing.assert_frame_equal(p.loc[p.evaluate,cols].reset_index(drop=True),original[cols].reset_index(drop=True))
        frozen=pd.read_csv(out/'directional_hypotheses.csv')
        pd.testing.assert_frame_equal(frozen,r)
    else:
        pd.testing.assert_frame_equal(p.loc[p.evaluate,cols].reset_index(drop=True),r.loc[r.evaluate,cols].reset_index(drop=True))
    checks=0
    for anchor in ('decision','arrival'):
        cumulative=a[[f'y_{anchor}_{u}s' for u in range(1,31)]].to_numpy()
        incremental=a[[f'y_{anchor}_incremental_{u}s' for u in range(1,31)]].to_numpy()
        prev=np.column_stack([np.zeros(len(a)),cumulative[:,:-1]])
        np.testing.assert_allclose(incremental,cumulative-prev,equal_nan=True)
        valid=np.isfinite(cumulative)&np.isfinite(prev)
        assert np.array_equal(np.isfinite(incremental),valid)
        full=np.isfinite(incremental).all(axis=1)
        np.testing.assert_allclose(incremental[full].sum(axis=1),cumulative[full,-1])
        checks+=int(valid.size)
    for h in (5,10):
        for group in ('price','book','queue','ofi','volume','price_volume','position','volume_depth','price_ofi','full'):
            prefix='R06_'+group+'_';suffix=f'_h{h}s'
            np.testing.assert_allclose(a[prefix+'EndSeconds'+suffix]-a[prefix+'StartSeconds'+suffix],a[prefix+'Coverage'+suffix],equal_nan=True)
            np.testing.assert_allclose(a[prefix+'CoverageRatio'+suffix],a[prefix+'Coverage'+suffix]/h,equal_nan=True)
            assert a[prefix+'MaxInterval'+suffix].dropna().le(1).all()
        np.testing.assert_allclose(a[f'C02_TotalSquares_h{h}s'],a[f'C02_UpSquares_h{h}s']+a[f'C02_DownSquares_h{h}s'],equal_nan=True)
    # Independently re-select actual pairs for new factors / late horizons / incremental / absolute targets.
    comparisons=0;aggregate_comparisons=0
    candidates=s.loc[s.factor.isin(['A04_QISlope_h10s','B09_BidRecovery_h10s','D09_LagOFICorr_h10s_lag2s','M07_AboveShare_h5s_g5s','R04_PriceRVPercentile_h10s'])]
    candidates=candidates.sample(20,random_state=2209)
    for row in candidates.itertuples(index=False):
        prefix=f'y_{row.anchor}_'+('incremental_' if row.label_type=='incremental' else '')
        labels=[prefix+f'{u}s' for u in range(1,31)];ics=[];ranks=[];counts=[]
        for date,day in a.groupby('trade_date',sort=True):
            part=day.loc[day.observation_seconds.eq(row.observation_seconds)]
            if row.pair_set!='own':
                good=day[row.factor].notna()&day[labels].notna().all(axis=1)
                common=day.assign(good=good).groupby('task_time').good.agg(['count','sum'])
                keys=common.index[(common['count']==5)&(common['sum']==5)]
                part=part.loc[part.task_time.isin(keys)]
            x=part[row.factor].to_numpy(float);y=part[prefix+f'{row.horizon_seconds}s'].to_numpy(float)
            if row.target=='absolute':y=abs(y)
            n,ic,rank,status=correlation_pair(x,y,20)
            counts.append(n);ics.append(ic);ranks.append(rank);comparisons+=1
        for values,mean,median,std,days in [(ics,row.mean_ic,row.median_ic,row.std_ic,row.valid_ic_days),(ranks,row.mean_rank_ic,row.median_rank_ic,row.std_rank_ic,row.valid_days)]:
            v=np.array(values);v=v[np.isfinite(v)]
            expected=[v.mean() if len(v) else np.nan,np.median(v) if len(v) else np.nan,v.std(ddof=1) if len(v)>1 else np.nan,len(v)]
            np.testing.assert_allclose(expected,[mean,median,std,days],atol=1e-10,rtol=1e-10,equal_nan=True)
            aggregate_comparisons+=4
        np.testing.assert_allclose([np.mean(counts),np.sum(counts),np.mean(counts)/50],[row.mean_pairs,row.total_pairs,row.mean_scheduled_coverage],atol=1e-10)
    b09=s.loc[s.family_id.eq('B09')]
    result=dict(atomic_sha256=sha(out/'atomic.parquet' if (out/'atomic.parquet').exists() else ROOT/'sample_snapshot_原子执行总表.parquet'),summary_sha256=sha(out/'summary_ic.parquet'),atomic_rows=len(a),dates=a.trade_date.nunique(),families=r.family_id.nunique(),computed_outputs=int(r.evaluate.sum()),registry_outputs=len(r),
        phase='all_sample',incremental_validity_cells_checked=checks,independent_actual_day_pairs=comparisons,
        independent_summary_statistics=aggregate_comparisons,aliases_checked=int(r.kind.eq('alias').sum()),r06_window_identities='passed',
        b09_max_valid_days=int(b09.valid_days.max()),original_prior_record_preserved=True,
        reviewed_prior_matches_freeze='review_origin' in r)
    (out/'acceptance_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':verify(sys.argv[1] if len(sys.argv)>1 else ROOT/'result/opening_execution/.pending')

