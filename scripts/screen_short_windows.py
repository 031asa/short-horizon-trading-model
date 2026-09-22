"""Compare W-second observations against the same W-second factor history."""
from pathlib import Path
from itertools import product
from datetime import datetime,timezone
import sys,json,re,hashlib, warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.ic_statistics import pair_matrix,STATUS

OUT=ROOT/'result/opening_execution'
RULES=dict(version='matched-short-windows-20260922-v1',windows=[1,2,3,4,5],
    ranking='fixed prior sign times daily mean Spearman IC over future seconds 1..5; rank at W=1',
    candidate_threshold=.05,minimum_valid_days=30,minimum_coverage=.8,minimum_agreeing_day_share=.6,
    support='Pearson score > 0 and arrival score > 0; both own and diagonal-common samples pass',
    common='per expression, same task valid at all five diagonal W configurations and all 30 horizons; separate anchor/response masks',
    bootstrap='2000 circular moving-block day resamples, block length 5 trading days, fixed seed 20260922; descriptive pointwise 95% interval, no multiple-testing correction',
    minimum_pairs=20,phase='all_sample',future_seconds=list(range(1,31)))

def catalog(registry):
    eligible=registry.loc[registry.kind.eq('computed') & registry.expected_sign_signed.isin(['positive','negative'])]
    groups=[];excluded=[]
    for _,r in eligible.loc[eligible.history_seconds.eq(0)&~eligible.factor.str.contains('_lag')].iterrows():
        groups.append(dict(expression=r.factor,name_zh=r.name_zh,prior=r.expected_sign_signed,
            mode='snapshot',factors=[r.factor]*5,definition=r.definition,family_id=r.family_id))
    rolling=eligible.loc[eligible.history_seconds.between(1,5)&eligible.factor.str.contains(r'_h\d+s$')].copy()
    rolling['expression']=rolling.factor.str.replace(r'_h\d+s$','',regex=True)
    for expression,g in rolling.groupby('expression',sort=False):
        g=g.sort_values('history_seconds');assert g.history_seconds.tolist()==[1,2,3,4,5]
        r=g.iloc[0];groups.append(dict(expression=expression,name_zh=r.name_zh,prior=r.expected_sign_signed,
            mode='rolling',factors=g.factor.tolist(),definition=r.definition,family_id=r.family_id))
    for stem in ['A08_OFIRateChange','B09_FullShareDifference','B09_RecoveryDifference','B09_SuccessTimeDifference']:
        excluded.append(dict(expression=stem,reason='前后两段各 2／5 秒，合计 4／10 秒；不属于五组普通回看' if stem.startswith('A08') else '固定 10 秒恢复历史及成熟门槛，不能称为 1–5 秒因子'))
    return groups,excluded

def bootstrap(values,draws):
    finite=np.isfinite(values);sample=values[draws];count=np.isfinite(sample).sum(axis=1)
    means=np.divide(np.nansum(sample,axis=1),count,out=np.full(len(draws),np.nan),where=count>0)
    return np.nanquantile(means,[.025,.975]).tolist() if finite.sum()>=2 else [np.nan,np.nan]

def diagonal_common_mask(x,y):
    """x is [matched W, task, expression]; y is [matched W, task, horizon]."""
    return np.isfinite(x).all(axis=0)&np.isfinite(y).all(axis=(0,2))[:,None]

def run():
    registry=pd.read_csv(OUT/'feature_registry.csv');groups,excluded=catalog(registry)
    rules={**RULES,'registered_at':datetime.now(timezone.utc).isoformat(),'expressions':[g['expression'] for g in groups]}
    (OUT/'短观察期筛选口径.json').write_text(json.dumps(rules,ensure_ascii=False,indent=2),encoding='utf-8')
    columns=['trade_date','task_time','observation_seconds','feature_source_age_seconds','arrival_delay_seconds']
    factor_names=list(dict.fromkeys(f for g in groups for f in g['factors']))
    labels={}
    for anchor,kind in product(['decision','arrival'],['cumulative','incremental']):
        labels[anchor,kind]=[f'y_{anchor}_'+('incremental_' if kind=='incremental' else '')+f'{u}s' for u in range(1,31)]
    atomic=pd.read_parquet(ROOT/'sample_snapshot_原子执行总表.parquet',columns=columns+factor_names+sum(labels.values(),[]))
    dates=sorted(atomic.trade_date.unique());assert len(atomic)==9500 and len(dates)==38
    combos=list(product(range(1,6),['decision','arrival'],['cumulative','incremental'],['own','diagonal_common']))
    shape=(len(dates),len(combos),len(groups),30)
    metrics={k:np.full(shape,np.nan) for k in ['ic','rank_ic','n_pairs','status']}
    redundancy=[]
    for di,date in enumerate(dates):
        day=atomic.loc[atomic.trade_date.eq(date)]
        frames=[day.loc[day.observation_seconds.eq(w)].sort_values('task_time') for w in range(1,6)]
        assert all(f.task_time.tolist()==frames[0].task_time.tolist() for f in frames)
        x=np.stack([f[[g['factors'][w] for g in groups]].to_numpy(float) for w,f in enumerate(frames)])
        ys={key:np.stack([f[cols].to_numpy(float) for f in frames]) for key,cols in labels.items()}
        masks={key:diagonal_common_mask(x,y) for key,y in ys.items()}
        for ci,(w,anchor,kind,pairs) in enumerate(combos):
            xx=x[w-1] if pairs=='own' else np.where(masks[anchor,kind],x[w-1],np.nan)
            n,ic,rank,status,_=pair_matrix(xx,ys[anchor,kind][w-1],20)
            for key,value in [('ic',ic),('rank_ic',rank),('n_pairs',n),('status',status)]:metrics[key][di,ci]=value
        _,_,r,_,_=pair_matrix(x[0],x[0],20);redundancy.append(r)
        if di%10==0:print(f'Compared {di+1}/{len(dates)} dates',flush=True)
    daily=[];summary=[];scores=[]
    rng=np.random.default_rng(20260922);starts=rng.integers(0,len(dates),size=(2000,int(np.ceil(len(dates)/5))))
    draws=((starts[:,:,None]+np.arange(5))%len(dates)).reshape(2000,-1)[:,:len(dates)]
    signs=np.array([1 if g['prior']=='positive' else -1 for g in groups])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for ci,(w,anchor,kind,pairs) in enumerate(combos):
            for gi,g in enumerate(groups):
                meta=dict(expression=g['expression'],factor=g['factors'][w-1],w=w,anchor=anchor,label_type=kind,pair_set=pairs,prior=g['prior'])
                ic=metrics['ic'][:,ci,gi];rank=metrics['rank_ic'][:,ci,gi];n=metrics['n_pairs'][:,ci,gi]
                for u in range(30):
                    v=rank[:,u];vdays=int(np.isfinite(v).sum());aligned=v*signs[gi]
                    summary.append(dict(**meta,horizon=u+1,mean_ic=float(np.nanmean(ic[:,u])),mean_rank_ic=float(np.nanmean(v)),valid_days=vdays,
                        mean_coverage=float(n[:,u].mean()/50),agreeing_days=int((aligned>0).sum()),positive_days=int((v>0).sum()),negative_days=int((v<0).sum())))
                    for di,date in enumerate(dates):daily.append(dict(**meta,horizon=u+1,trade_date=date,ic=ic[di,u],rank_ic=v[di],n_pairs=int(n[di,u]),status=STATUS[int(metrics['status'][di,ci,gi,u])]))
                # All first five ICs must exist on a day. Never treat five horizons as independent days.
                valid=np.isfinite(rank[:,:5]).all(axis=1);v=np.where(valid,rank[:,:5].mean(axis=1),np.nan);aligned=v*signs[gi]
                pearson=np.where(np.isfinite(ic[:,:5]).all(axis=1),ic[:,:5].mean(axis=1),np.nan)
                low,high=bootstrap(aligned,draws)
                extra=np.where(np.isfinite(rank[:,1:5]).all(axis=1),rank[:,1:5].mean(axis=1),np.nan)
                scores.append(dict(**meta,raw_score=float(np.nanmean(v)),aligned_score=float(np.nanmean(aligned)),raw_pearson=float(np.nanmean(pearson)),
                    aligned_pearson=float(np.nanmean(pearson)*signs[gi]),valid_days=int(valid.sum()),coverage=float(n[:,:5].min(axis=1).mean()/50),
                    agreeing_day_share=float((aligned>0).sum()/valid.sum()) if valid.sum() else np.nan,ci_low=low,ci_high=high,
                    raw_seconds_2_5=float(np.nanmean(extra)),aligned_seconds_2_5=float(np.nanmean(extra)*signs[gi])))
    summary=pd.DataFrame(summary);daily=pd.DataFrame(daily);scores=pd.DataFrame(scores)
    summary.to_parquet(OUT/'短观察期_逐期限汇总.parquet',index=False)
    daily.to_parquet(OUT/'短观察期_逐日IC.parquet',index=False,compression='zstd')
    scores.to_csv(OUT/'短观察期_五组对比.csv',index=False,encoding='utf-8-sig')
    # Recomputed diagonal own samples must reproduce the published full experiment.
    old=pd.read_parquet(OUT/'summary_ic.parquet',filters=[('target','==','signed'),('pair_set','==','own'),('factor','in',factor_names)],
        columns=['factor','observation_seconds','anchor','label_type','horizon_seconds','mean_ic','mean_rank_ic','valid_days','mean_scheduled_coverage'])
    check=summary.loc[summary.pair_set.eq('own')].merge(old,left_on=['factor','w','anchor','label_type','horizon'],right_on=['factor','observation_seconds','anchor','label_type','horizon_seconds'],suffixes=('','_old'),validate='one_to_one')
    assert len(check)==len(summary)//2
    for col in ['mean_ic','mean_rank_ic','valid_days']:np.testing.assert_allclose(check[col],check[col+'_old'],rtol=1e-10,atol=1e-10,equal_nan=True)
    np.testing.assert_allclose(check.mean_coverage,check.mean_scheduled_coverage,atol=1e-10)
    for _,block in daily.loc[daily.pair_set.eq('diagonal_common')].groupby(['expression','trade_date','anchor','label_type']):assert block.n_pairs.nunique()==1
    result=[]
    for g in groups:
        def score(w,anchor='decision',pairs='own',kind='cumulative'):
            return scores.loc[scores.expression.eq(g['expression'])&scores.w.eq(w)&scores.anchor.eq(anchor)&scores.pair_set.eq(pairs)&scores.label_type.eq(kind)].iloc[0]
        a=score(1);common=score(1,pairs='diagonal_common');arrival=score(1,anchor='arrival');arrival_common=score(1,anchor='arrival',pairs='diagonal_common')
        def passed(r):return r.aligned_score>=.05 and r.valid_days>=30 and r.coverage>=.8 and r.agreeing_day_share>=.6 and r.aligned_pearson>0
        candidate=passed(a) and passed(common) and arrival.aligned_score>0 and arrival_common.aligned_score>0
        result.append(dict(**g,candidate=bool(candidate),**{f'w{w}_raw':float(score(w).raw_score) for w in range(1,6)},
            own_score=float(a.aligned_score),common_score=float(common.aligned_score),arrival_score=float(arrival.aligned_score),
            own_days=int(a.valid_days),coverage=float(a.coverage),common_coverage=float(common.coverage),day_share=float(a.agreeing_day_share),
            ci_low=float(a.ci_low),ci_high=float(a.ci_high),incremental_2_5=float(score(1,kind='incremental').raw_seconds_2_5),
            arrival_incremental_2_5=float(score(1,anchor='arrival',kind='incremental').raw_seconds_2_5),
            delta_1_minus_5=float(a.aligned_score-score(5).aligned_score)))
    result=sorted(result,key=lambda r:r['own_score'] if np.isfinite(r['own_score']) else -999,reverse=True)
    correlations=[]
    rr=np.stack(redundancy)
    for i,j in product(range(len(groups)),repeat=2):
        if i<j:
            v=rr[:,i,j];correlations.append(dict(a=groups[i]['expression'],b=groups[j]['expression'],mean_rank_correlation=float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,valid_days=int(np.isfinite(v).sum())))
    pd.DataFrame(correlations).to_csv(OUT/'短观察期_因子相关性.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(result).drop(columns=['factors']).to_csv(OUT/'短观察期_候选清单.csv',index=False,encoding='utf-8-sig')
    data=dict(rules=rules,groups=result,excluded=excluded,scores=scores.to_dict('records'),curves=summary.to_dict('records'),correlations=correlations,
        audit=dict(own_cells_verified=len(check),dates=len(dates),atomic_rows=len(atomic),source_age=atomic.feature_source_age_seconds.describe().to_dict(),arrival_delay=atomic.arrival_delay_seconds.describe().to_dict()))
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        return None if isinstance(x,float) and not np.isfinite(x) else x
    data=clean(data);(OUT/'短观察期_看板数据.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print(pd.DataFrame(result)[['expression','candidate','w1_raw','w2_raw','w3_raw','w4_raw','w5_raw','arrival_score','coverage','day_share','incremental_2_5']].to_string(index=False),flush=True)
    print('VERIFIED own summary cells:',len(check),flush=True)

if __name__=='__main__':run()
