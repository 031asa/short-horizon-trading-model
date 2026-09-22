"""Check real new-window values, causal prefixes and independent daily IC."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_ic import ICConfig,SessionData,correlation_pair
from utils.directional_review import enrich_atomic
from scripts.audit_opening_data import RAW
from scripts.run_opening_ic import sha


def verify(out):
    out=Path(out);ap=out/'atomic.parquet' if (out/'atomic.parquet').exists() else ROOT/'sample_snapshot_原子执行总表.parquet'
    a=pd.read_parquet(ap);reg=pd.read_csv(out/'feature_registry.csv');s=pd.read_parquet(out/'summary_ic.parquet')
    new=reg.loc[reg.window_extension];cfg=ICConfig(history_seconds=tuple(range(1,11)))
    assert len(a)==9500 and a.trade_date.nunique()==38 and int(reg.evaluate.sum())==1271
    assert len(reg)==2377 and reg.family_id.nunique()==59
    assert reg.loc[reg.evaluate,'factor'].str.replace(r'_h\d+s(?=_|$)','_hW',regex=True).nunique()==182
    assert set(new.history_seconds)=={1,2,3,4,6,7,8,9}
    for row in new.itertuples():
        assert np.array_equal(np.isfinite(a[row.factor]),a[row.factor+'__status'].eq('ok'))
    raw=pd.read_parquet(RAW);raw['Datetime']=pd.to_datetime(raw.Datetime).dt.tz_convert('Asia/Shanghai')
    raw['source_row']=np.arange(len(raw));dates=raw.Datetime.dt.strftime('%Y-%m-%d')
    causal=0
    for date,t in [('2026-07-21',44),('2026-07-21',47),('2026-07-21',57),('2026-09-11',64)]:
        start=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai');decision=start+pd.Timedelta(seconds=t)
        frame=raw.loc[dates.eq(date)&raw.Datetime.ge(start)&raw.Datetime.le(start+pd.Timedelta(seconds=96))].copy()
        modified=frame.copy();future=modified.Datetime.gt(decision)
        modified.loc[future,['LastPrice','BidPrice1','AskPrice1']]+=1000
        modified.loc[future,['BidVolume1','AskVolume1']]*=11
        modified.loc[future,'Volume']+=100000
        results=[]
        for f in (frame,frame.loc[frame.Datetime.le(decision)].copy(),modified):
            v=pd.DataFrame([SessionData(f,start,cfg).features(t)])
            results.append(enrich_atomic(v,cfg.history_seconds)[new.factor].to_numpy(float)[0])
        np.testing.assert_allclose(results[0],results[1],equal_nan=True)
        np.testing.assert_allclose(results[0],results[2],equal_nan=True)
        actual=a.loc[a.trade_date.eq(date)&a.decision_time.eq(decision),new.factor]
        for row in actual.to_numpy(float):np.testing.assert_allclose(row,results[0],atol=1e-10,equal_nan=True)
        causal+=len(new)*2
    # Every added history value is tested across different families and IC coordinates.
    templates=['F05_Momentum','A03_QIMean','M01_MADeviation','D09_SignedVolumeCorr','B03_BidSlope','R04_PriceRVPercentile','M07_MeanStateAge']
    comparisons=0;aggregate=0
    for h in (1,2,3,4,6,7,8,9):
        for j,stem in enumerate(templates):
            factor=f'{stem}_h{h}s'+('_g5s' if stem.startswith('M07') else '')
            row=s.loc[s.factor.eq(factor)].sample(1,random_state=100*h+j).iloc[0]
            prefix=f'y_{row.anchor}_'+('incremental_' if row.label_type=='incremental' else '')
            labelcols=[prefix+f'{u}s' for u in range(1,31)]
            slim=a[['trade_date','task_time','observation_seconds',factor]+labelcols]
            ic=[];rank=[];counts=[]
            for date,day in slim.groupby('trade_date'):
                part=day.loc[day.observation_seconds.eq(row.observation_seconds)]
                if row.pair_set!='own':
                    g=day.assign(valid=day[factor].notna()&day[labelcols].notna().all(axis=1)).groupby('task_time').valid.agg(['sum','count'])
                    part=part.loc[part.task_time.isin(g.index[g['sum'].eq(5)&g['count'].eq(5)])]
                y=part[prefix+f'{int(row.horizon_seconds)}s'].to_numpy(float)
                if row.target=='absolute':y=np.abs(y)
                n,c,r,status=correlation_pair(part[factor].to_numpy(float),y,20)
                ic.append(c);rank.append(r);counts.append(n);comparisons+=1
            for vals,method,daykey in [(ic,'ic','valid_ic_days'),(rank,'rank_ic','valid_days')]:
                x=np.array(vals);x=x[np.isfinite(x)]
                expected=[x.mean() if len(x) else np.nan,np.median(x) if len(x) else np.nan,x.std(ddof=1) if len(x)>1 else np.nan,len(x)]
                np.testing.assert_allclose(expected,[row['mean_'+method],row['median_'+method],row['std_'+method],row[daykey]],atol=1e-10,rtol=1e-10,equal_nan=True)
                aggregate+=4
            np.testing.assert_allclose([np.mean(counts),sum(counts),np.mean(counts)/50],[row.mean_pairs,row.total_pairs,row.mean_scheduled_coverage],atol=1e-10)
    default=s.loc[s.observation_seconds.eq(1)&s.anchor.eq('decision')&s.label_type.eq('cumulative')&s.target.eq('signed')&s.pair_set.eq('own')]
    probes=[]
    for h in range(1,11):
        row=default.loc[default.factor.eq(f'F05_Momentum_h{h}s')&default.horizon_seconds.eq(5)].iloc[0]
        v=row.mean_rank_ic;display='—' if not np.isfinite(v) else ('+' if v>0 else '')+f'{v:.3f}'
        probes.append(dict(h=h,factor=row.factor,display=display,valid_days=int(row.valid_days),coverage=float(row.mean_scheduled_coverage)))
    (out/'window_input_expected.json').write_text(json.dumps(probes,indent=2),encoding='utf-8')
    result=dict(atomic_sha256=sha(ap),summary_sha256=sha(out/'summary_ic.parquet'),status='passed',
        input_windows=list(range(1,11)),computed_expressions=182,computed_versions=1271,
        new_output_validity_checks=len(a)*len(new),causal_prefix_future_checks=causal,
        independent_daily_ic=comparisons,independent_summary_statistics=aggregate,window_input_probes=10)
    (out/'window_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':verify(sys.argv[1] if len(sys.argv)>1 else ROOT/'result/opening_execution/.window_pending')
