"""Independent real-data causal/IC checks for the directional research extension."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_ic import ICConfig,SessionData,correlation_pair
from utils.directional_review import enrich_atomic,variants
from scripts.run_opening_ic import sha
from scripts.audit_opening_data import RAW


def verify(out):
    out=Path(out)
    atomic_path=out/'atomic.parquet' if (out/'atomic.parquet').exists() else ROOT/'sample_snapshot_原子执行总表.parquet'
    a=pd.read_parquet(atomic_path);s=pd.read_parquet(out/'summary_ic.parquet');r=pd.read_csv(out/'feature_registry.csv')
    names=[v['factor'] for v in variants()]
    assert len(names)==28 and set(names)==set(r.loc[r.review_origin.eq('derived'),'factor'])
    # Exact parent validity intersection, including zeros and left-censored ages.
    for spec in variants():
        valid=np.isfinite(a[spec['inputs']]).all(axis=1)
        for parent in spec['inputs']:valid &= a[parent+'__status'].eq('ok')
        assert np.array_equal(valid.to_numpy(),a[spec['factor']].notna().to_numpy()),spec['factor']
        assert a.loc[~valid,spec['factor']+'__status'].str.startswith('PARENT_INVALID:').all()
    # Independently check the mechanisms' key identities on every real task.
    np.testing.assert_allclose(a.C01_QIRV_h5s,a.F01_QI*a.C01_PriceRV_h5s,equal_nan=True)
    gate_checks=0
    activation=[]
    for h in (5,10):
        m=a[f'M01_MADeviation_h{h}s'];o=a[f'A07_OFIDirection_h{h}s'];good=m.notna()&o.notna()
        for stem,positive in [('TrendConfirmed',True),('CounterflowDeviation',False)]:
            active=(m*o>0) if positive else (m*o<0)
            expected=m.where(active,0).where(good)
            factor=f'M01_{stem}_h{h}s'
            np.testing.assert_allclose(a[factor],expected,equal_nan=True)
            gate_checks+=len(a)
            activation.append(dict(factor=factor,valid_rows=int(good.sum()),active_rows=int((active&good).sum()),inactive_valid_rows=int((~active&good).sum())))
    pd.DataFrame(activation).to_csv(out/'门控因子激活覆盖.csv',index=False,encoding='utf-8-sig')
    # True source prefix truncation and changed future market data, including the gap date.
    raw=pd.read_parquet(RAW);raw['Datetime']=pd.to_datetime(raw.Datetime).dt.tz_convert('Asia/Shanghai')
    raw['source_row']=np.arange(len(raw));dates=raw.Datetime.dt.strftime('%Y-%m-%d')
    causal=0
    for date,t in [('2026-07-21',12),('2026-07-21',44),('2026-07-21',57),('2026-07-21',63),
                   ('2026-07-22',20),('2026-08-03',35),('2026-08-24',55),('2026-09-11',64)]:
        start=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai');decision=start+pd.Timedelta(seconds=t)
        f=raw.loc[dates.eq(date)&raw.Datetime.ge(start)&raw.Datetime.le(start+pd.Timedelta(seconds=96))].copy()
        prefix=f.loc[f.Datetime.le(decision)].copy()
        modified=f.copy();future=modified.Datetime.gt(decision)
        modified.loc[future,['LastPrice','BidPrice1','AskPrice1']]+=1000
        modified.loc[future,['BidVolume1','AskVolume1']]*=11
        modified.loc[future,'Volume']+=100000
        values=[]
        for frame in (f,prefix,modified):
            result=pd.DataFrame([SessionData(frame,start,ICConfig()).features(t)])
            values.append(enrich_atomic(result)[names].to_numpy(float)[0])
        np.testing.assert_allclose(values[0],values[1],equal_nan=True)
        np.testing.assert_allclose(values[0],values[2],equal_nan=True)
        actual=a.loc[a.trade_date.eq(date)&a.decision_time.eq(decision),names]
        assert len(actual)>0
        for row in actual.to_numpy(float):np.testing.assert_allclose(row,values[0],atol=1e-10,equal_nan=True)
        causal+=len(names)*2
    # Deterministic coverage of every new expression over 3 sampled coordinate combinations.
    comparisons=0;summary_comparisons=0
    for factor in names:
        probes=s.loc[s.factor.eq(factor)].sample(3,random_state=20260922)
        for row in probes.itertuples():
            prefix=f'y_{row.anchor}_'+('incremental_' if row.label_type=='incremental' else '')
            labels=[prefix+f'{u}s' for u in range(1,31)];ics=[];ranks=[];counts=[]
            for date,day in a.groupby('trade_date',sort=True):
                part=day.loc[day.observation_seconds.eq(row.observation_seconds)]
                if row.pair_set!='own':
                    good=day[factor].notna()&day[labels].notna().all(axis=1)
                    g=day.assign(good=good).groupby('task_time').good.agg(['count','sum'])
                    part=part.loc[part.task_time.isin(g.index[(g['count']==5)&(g['sum']==5)])]
                y=part[prefix+f'{row.horizon_seconds}s'].to_numpy(float)
                if row.target=='absolute':y=np.abs(y)
                n,ic,rank,status=correlation_pair(part[factor].to_numpy(float),y,20)
                counts.append(n);ics.append(ic);ranks.append(rank);comparisons+=1
            for vals,method,days_col in [(ics,'ic','valid_ic_days'),(ranks,'rank_ic','valid_days')]:
                vals=np.asarray(vals);vals=vals[np.isfinite(vals)]
                expected=[vals.mean() if len(vals) else np.nan,np.median(vals) if len(vals) else np.nan,
                          vals.std(ddof=1) if len(vals)>1 else np.nan,len(vals),int((vals>0).sum()),int((vals<0).sum())]
                observed=[getattr(row,'mean_'+method),getattr(row,'median_'+method),getattr(row,'std_'+method),
                          getattr(row,days_col),getattr(row,'positive_'+method+'_days'),getattr(row,'negative_'+method+'_days')]
                np.testing.assert_allclose(expected,observed,rtol=1e-10,atol=1e-10,equal_nan=True);summary_comparisons+=6
            np.testing.assert_allclose([np.mean(counts),np.sum(counts),np.mean(counts)/50],
                [row.mean_pairs,row.total_pairs,row.mean_scheduled_coverage],atol=1e-10)
    # Additional browser probes refer to full-precision data, not its embedded copy.
    probes=json.loads((out/'dashboard_qa_probes.json').read_text(encoding='utf-8'))
    probes=[p for p in probes if p['factor'] not in names]
    for factor,obs,anchor,kind,target,pairs,method,u in [
        ('C01_QIRV_h5s',1,'decision','cumulative','signed','own','rank_ic',30),
        ('M01_CounterflowDeviation_h10s',3,'arrival','incremental','absolute','common_observations_horizons','ic',5),
        ('A05_SignedStateAge',5,'decision','incremental','signed','own','rank_ic',10)]:
        row=s.loc[s.factor.eq(factor)&s.observation_seconds.eq(obs)&s.anchor.eq(anchor)&s.label_type.eq(kind)&
            s.target.eq(target)&s.pair_set.eq(pairs)&s.horizon_seconds.eq(u)].iloc[0]
        v=row['mean_'+method];display='—' if not np.isfinite(v) else ('+' if v>0 else '')+f'{v:.3f}'
        probes.append(dict(factor=factor,family_id=factor.split('_')[0],observation_seconds=obs,anchor=anchor,
            label_type=kind,target=target,pair_set=pairs,method=method,horizon_seconds=u,display=display,meanDisplay=display))
    (out/'dashboard_qa_probes.json').write_text(json.dumps(probes,ensure_ascii=False,indent=2),encoding='utf-8')
    result=dict(atomic_sha256=sha(atomic_path),summary_sha256=sha(out/'summary_ic.parquet'),new_variants=len(names),
        parent_validity_checks=len(a)*len(names),gate_identity_checks=gate_checks,causal_prefix_and_future_checks=causal,
        independent_daily_correlations=comparisons,independent_summary_statistics=summary_comparisons,
        offline_variant_probes=3,status='passed')
    (out/'directional_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':verify(sys.argv[1] if len(sys.argv)>1 else ROOT/'result/opening_execution/.review_pending')
