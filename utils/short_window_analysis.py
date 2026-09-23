"""Per-horizon history compression, with explicit sample and clock alignment."""
import json,hashlib,warnings
from datetime import datetime,timezone
from itertools import product
import numpy as np
import pandas as pd
from .ic_statistics import pair_matrix,STATUS

def summarize_vector(rank,pearson,n,sign):
    finite=np.isfinite(rank);days=finite.sum(axis=0)
    return dict(mean_rank_ic=np.nanmean(rank,axis=0),mean_ic=np.nanmean(pearson,axis=0),valid_days=days,
        mean_coverage=n.mean(axis=0)/50,agreeing_day_share=np.divide((rank*sign>0).sum(axis=0),days,out=np.full(days.shape,np.nan),where=days>0))

def strong(r,sign,rules):
    return (np.isfinite(r['mean_rank_ic']) and sign*r['mean_rank_ic']>=rules['ic_threshold']
        and r['valid_days']>=rules['minimum_valid_days'] and r['mean_coverage']>=rules['minimum_coverage']
        and r['agreeing_day_share']>=rules['minimum_agreeing_day_share'] and sign*r['mean_ic']>0)

def pair_comparison(short,long,short_ic,long_ic,n,sign,weights):
    """Same tasks were used upstream; restrict each horizon to paired valid days."""
    valid=np.isfinite(short)&np.isfinite(long);delta=np.where(valid,(short-long)*sign,np.nan)
    denom=weights@valid.astype(float);means=np.divide(weights@np.nan_to_num(delta),denom,out=np.full((len(weights),delta.shape[1]),np.nan),where=denom>0)
    lo,hi=np.nanquantile(means,[.025,.975],axis=0)
    return dict(raw_short_ic=np.nanmean(np.where(valid,short,np.nan),axis=0),raw_long_ic=np.nanmean(np.where(valid,long,np.nan),axis=0),
        raw_difference=np.nanmean(np.where(valid,short-long,np.nan),axis=0),aligned_difference=np.nanmean(delta,axis=0),ci_low=lo,ci_high=hi,
        paired_days=valid.sum(axis=0),coverage=n.mean(axis=0)/50,
        raw_short_pearson=np.nanmean(np.where(valid,short_ic,np.nan),axis=0),raw_long_pearson=np.nanmean(np.where(valid,long_ic,np.nan),axis=0))

def compression_supported(short_rows,long_rows,pairs,sign,rules):
    return all(strong(a,sign,rules) and strong(b,sign,rules) and p['paired_days']>=rules['minimum_valid_days']
        and p['coverage']>=rules['minimum_coverage'] and np.isfinite(p['ci_low']) and p['ci_low']>=-rules['compression_margin']
        for a,b,p in zip(short_rows,long_rows,pairs))

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    return None if isinstance(x,float) and not np.isfinite(x) else x

def analyze(groups,excluded,registry,rules,out,root):
    import pyarrow as pa
    import pyarrow.parquet as pq
    sha=lambda p:hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
    rules={**rules,'registered_at':datetime.now(timezone.utc).isoformat(),'registry_sha256':sha(out/'feature_registry.csv'),'already_seen_same_futures_data':True}
    (out/'短观察期筛选口径.json').write_text(json.dumps(rules,ensure_ascii=False,indent=2),encoding='utf-8')
    reg=registry.set_index('factor');names=list(dict.fromkeys(f for g in groups for f in g['factors']))
    labels={(a,k):[f'y_{a}_'+('incremental_' if k=='incremental' else '')+f'{u}s' for u in range(1,31)] for a,k in product(['decision','arrival'],['cumulative','incremental'])}
    counts={f:('R06_'+reg.loc[f,'dependency']+'_Count_h'+str(int(reg.loc[f,'history_seconds']))+'s') for f in names if reg.loc[f,'history_seconds']>0}
    spans={f:c.replace('_Count_','_Coverage_') for f,c in counts.items()}
    starts={f:c.replace('_Count_','_StartSeconds_') for f,c in counts.items()}
    cols=['trade_date','task_time','task_elapsed_seconds','observation_seconds','decision_elapsed_seconds','feature_source_age_seconds','arrival_delay_seconds']+names+sum(labels.values(),[])+list(set(counts.values())|set(spans.values())|set(starts.values()))
    atomic=pd.read_parquet(root/'sample_snapshot_原子执行总表.parquet',columns=cols)
    dates=sorted(atomic.trade_date.unique());assert len(dates)==38 and len(atomic)==9500
    # Only timestamps are required to count newly received snapshots, not to recompute features.
    raw=pq.read_table(root/'新窗口交接_20260921/20260720_20260911_IC2609.parquet',columns=['Datetime']).to_pandas()
    t=raw.Datetime;raw=raw.loc[(t.dt.hour==9)&(t.dt.minute.between(30,31))].copy();raw['date']=raw.Datetime.dt.strftime('%Y-%m-%d')
    raw_times={date:((g.Datetime-pd.Timestamp(date+' 09:30',tz='Asia/Shanghai')).dt.total_seconds().to_numpy()) for date,g in raw.groupby('date')}
    modes=['wait','fixed'];combos=list(product(range(1,6),['decision','arrival'],['cumulative','incremental'],['own','diagonal_common']))
    shape=(38,2,len(combos),len(groups),30)
    metrics={k:np.full(shape,np.nan) for k in ['ic','rank_ic']};metrics['n_pairs']=np.zeros(shape,np.int16);metrics['status']=np.ones(shape,np.int8)
    paircoords=list(product(range(1,5),['decision','arrival'],['cumulative','incremental']))
    pshape=(38,2,len(paircoords),len(groups),30)
    pm={k:np.full(pshape,np.nan) for k in ['short','long','short_ic','long_ic']};pm['n']=np.zeros(pshape,np.int16)
    input_rows=[];correlation=[]
    for di,date in enumerate(dates):
        day=atomic.loc[atomic.trade_date.eq(date)];frames=[day.loc[day.observation_seconds.eq(w)].sort_values('task_time') for w in range(1,6)]
        assert all(f.task_time.tolist()==frames[0].task_time.tolist() for f in frames)
        for mi,mode in enumerate(modes):
            fs=frames if mode=='wait' else [frames[-1]]*5
            x=np.stack([f[[g['factors'][wi] for g in groups]].to_numpy(float) for wi,f in enumerate(fs)])
            ys={key:np.stack([f[c].to_numpy(float) for f in fs]) for key,c in labels.items()}
            common={key:np.isfinite(x).all(axis=0)&np.isfinite(y).all(axis=(0,2))[:,None] for key,y in ys.items()}
            for ci,(w,a,k,p) in enumerate(combos):
                xx=x[w-1] if p=='own' else np.where(common[a,k],x[w-1],np.nan)
                n,ic,r,status,_=pair_matrix(xx,ys[a,k][w-1],20)
                for key,v in [('n_pairs',n),('ic',ic),('rank_ic',r),('status',status)]:metrics[key][di,mi,ci]=v
            for ci,(w,a,k) in enumerate(paircoords):
                valid=np.isfinite(x[w-1])&np.isfinite(x[-1])&(np.isfinite(ys[a,k][w-1]).all(axis=1)&np.isfinite(ys[a,k][-1]).all(axis=1))[:,None]
                ns,ics,rs,_,_=pair_matrix(np.where(valid,x[w-1],np.nan),ys[a,k][w-1],20)
                nl,icl,rl,_,_=pair_matrix(np.where(valid,x[-1],np.nan),ys[a,k][-1],20)
                assert np.array_equal(ns,nl)
                for key,v in [('short',rs),('long',rl),('short_ic',ics),('long_ic',icl),('n',ns)]:pm[key][di,mi,ci]=v
            for wi,f in enumerate(fs):
                for gi,g in enumerate(groups):
                    factor=g['factors'][wi];good=np.isfinite(x[wi,:,gi]);n=np.where(good,1 if factor in ['F01_QI','F07_QuotePosition','A05_QIState'] else 2,np.nan)
                    if g['mode']=='rolling':n=f[counts[factor]].to_numpy(float)
                    span=f[spans[factor]].to_numpy(float) if g['mode']=='rolling' else np.full(50,np.nan)
                    begin=f[starts[factor]].to_numpy(float) if g['mode']=='rolling' else np.full(50,np.nan)
                    ticks=raw_times[date];new=np.searchsorted(ticks,f.decision_elapsed_seconds.to_numpy(),side='right')-np.searchsorted(ticks,f.task_elapsed_seconds.to_numpy(),side='right')
                    for j in range(50):input_rows.append(dict(mode=mode,expression=g['expression'],w=wi+1,trade_date=date,task_time=f.task_time.iloc[j],factor_valid=bool(good[j]),
                        window_snapshots=n[j],used_snapshots=n[j] if good[j] else np.nan,covered_seconds=span[j],new_snapshots=int(new[j]),
                        history_before_task=max(0.,f.task_elapsed_seconds.iloc[j]-begin[j]) if np.isfinite(begin[j]) else np.nan))
            if mode=='wait':correlation.append(pair_matrix(x[0],x[0],20)[2])
        if di%10==0:print(f'Curves and paired samples: {di+1}/38 days',flush=True)
    inputs=pd.DataFrame(input_rows);inputs.to_parquet(out/'短观察期_快照数量逐任务.parquet',index=False)
    inputs_summary=[]
    for (mode,expression,w),g in inputs.groupby(['mode','expression','w'],sort=False):
        n=g.used_snapshots;valid=g.factor_valid
        inputs_summary.append(dict(mode=mode,expression=expression,w=int(w),valid_tasks=int(valid.sum()),scheduled_tasks=len(g),
            median_used=float(n.median()),min_used=float(n.min()),max_used=float(n.max()),median_available=float(g.window_snapshots.median()),
            median_new=float(g.new_snapshots.median()),median_span=float(g.loc[valid,'covered_seconds'].median()),max_history_before_task=float(g.history_before_task.max())))
    rng=np.random.default_rng(20260923);starts_draw=rng.integers(0,38,size=(2000,8));draws=((starts_draw[:,:,None]+np.arange(5))%38).reshape(2000,-1)[:,:38]
    weights=np.stack([np.bincount(r,minlength=38) for r in draws]).astype(float)
    curves=[];paired=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for mi,mode in enumerate(modes):
            for ci,(w,a,k,p) in enumerate(combos):
                for gi,g in enumerate(groups):
                    v=summarize_vector(metrics['rank_ic'][:,mi,ci,gi],metrics['ic'][:,mi,ci,gi],metrics['n_pairs'][:,mi,ci,gi],1 if g['prior']=='positive' else -1)
                    for u in range(30):curves.append(dict(phase=rules['phase'],mode=mode,expression=g['expression'],factor=g['factors'][w-1],w=w,anchor=a,label_type=k,pair_set=p,horizon=u+1,**{name:arr[u].item() for name,arr in v.items()}))
            for ci,(w,a,k) in enumerate(paircoords):
                for gi,g in enumerate(groups):
                    v=pair_comparison(pm['short'][:,mi,ci,gi],pm['long'][:,mi,ci,gi],pm['short_ic'][:,mi,ci,gi],pm['long_ic'][:,mi,ci,gi],pm['n'][:,mi,ci,gi],1 if g['prior']=='positive' else -1,weights)
                    for u in range(30):paired.append(dict(phase=rules['phase'],mode=mode,expression=g['expression'],w=w,anchor=a,label_type=k,horizon=u+1,**{name:arr[u].item() for name,arr in v.items()}))
    df=pd.DataFrame(curves);pdf=pd.DataFrame(paired)
    df.to_parquet(out/'短观察期_逐期限汇总.parquet',index=False);df.to_csv(out/'短观察期_五组对比.csv',index=False,encoding='utf-8-sig')
    pdf.to_csv(out/'短观察期_相对5秒逐期限差异.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(inputs_summary).to_csv(out/'短观察期_快照数量汇总.csv',index=False,encoding='utf-8-sig')
    # Daily file is written in bounded day-sized batches.
    metadata=pd.DataFrame([dict(mode=mode,w=w,anchor=a,label_type=k,pair_set=p,expression=g['expression'],factor=g['factors'][w-1],horizon=u)
        for mode in modes for w,a,k,p in combos for g in groups for u in range(1,31)])
    writer=None
    for di,date in enumerate(dates):
        d=metadata.copy();d['phase']=rules['phase'];d['trade_date']=date
        for key in ['ic','rank_ic','n_pairs']:d[key]=metrics[key][di].ravel()
        d['status']=STATUS[metrics['status'][di].ravel()];table=pa.Table.from_pandas(d,preserve_index=False)
        if writer is None:writer=pq.ParquetWriter(out/'短观察期_逐日IC.parquet',table.schema,compression='zstd')
        writer.write_table(table)
    writer.close()
    old=pd.read_parquet(out/'summary_ic.parquet',filters=[('target','==','signed'),('pair_set','==','own'),('factor','in',names)],columns=['factor','observation_seconds','anchor','label_type','horizon_seconds','mean_ic','mean_rank_ic','valid_days','mean_scheduled_coverage'])
    own=df.loc[df.pair_set.eq('own')].copy();own['observation_seconds']=np.where(own['mode'].eq('wait'),own.w,5)
    check=own.merge(old,left_on=['factor','observation_seconds','anchor','label_type','horizon'],right_on=['factor','observation_seconds','anchor','label_type','horizon_seconds'],suffixes=('','_old'),validate='many_to_one')
    assert len(check)==len(own)
    for col in ['mean_ic','mean_rank_ic','valid_days']:np.testing.assert_allclose(check[col],check[col+'_old'],atol=1e-10,rtol=1e-10,equal_nan=True)
    np.testing.assert_allclose(check.mean_coverage,check.mean_scheduled_coverage,atol=1e-10)
    lookup={(r['mode'],r['expression'],r['w'],r['horizon']):r for r in curves if r['anchor']=='decision' and r['label_type']=='cumulative' and r['pair_set']=='own'}
    plookup={(r['mode'],r['expression'],r['w'],r['horizon']):r for r in paired if r['anchor']=='decision' and r['label_type']=='cumulative'}
    selection=[]
    for mode,g in product(modes,groups):
        sign=1 if g['prior']=='positive' else -1;expr=g['expression'];supported=[];quality=[]
        for w in range(1,6):
            rs=[lookup[mode,expr,w,u] for u in rules['early_horizons']];quality.append(all(strong(r,sign,rules) for r in rs))
            if w<5 and g['mode']=='rolling':supported.append(compression_supported(rs,[lookup[mode,expr,5,u] for u in rules['early_horizons']],[plookup[mode,expr,w,u] for u in rules['early_horizons']],sign,rules))
        minimum=next((w for w,v in enumerate(supported,1) if v),None)
        firstgood=next((w for w,v in enumerate(quality,1) if v),None)
        if g['mode']=='snapshot':category='快照基准有效' if quality[0] else '快照基准暂弱'
        elif minimum is not None:category='短窗接近有效长窗'
        elif firstgood is not None and not quality[-1]:category='短窗有效，长窗较弱'
        elif firstgood is not None:category='有预测关系，压缩证据不足'
        elif strong(lookup[mode,expr,1,1],sign,rules):category='仅首秒脉冲候选'
        else:category='暂未通过关系门槛'
        selection.append(dict(mode=mode,expression=expr,category=category,minimum_supported_window=minimum,first_strong_window=firstgood,
            strong_windows=[w for w,v in enumerate(quality,1) if v],short_window_supported=[w for w,v in enumerate(supported,1) if v],
            short_effect_horizons=[u for u in range(1,31) if strong(lookup[mode,expr,1,u],sign,rules)],candidate=firstgood is not None or category=='仅首秒脉冲候选'))
    pd.DataFrame(selection).to_csv(out/'短观察期_候选清单.csv',index=False,encoding='utf-8-sig')
    cr=[];rs=np.stack(correlation)
    for i in range(len(groups)):
        for j in range(i+1,len(groups)):
            v=rs[:,i,j];cr.append(dict(a=groups[i]['expression'],b=groups[j]['expression'],mean_rank_correlation=float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,valid_days=int(np.isfinite(v).sum())))
    pd.DataFrame(cr).to_csv(out/'短观察期_因子相关性.csv',index=False,encoding='utf-8-sig')
    data=clean(dict(rules=rules,groups=groups,excluded=excluded,curves=curves,paired=paired,counts=inputs_summary,selections=selection,correlations=cr,
        audit=dict(dates=38,atomic_rows=9500,own_cells_verified=len(check),pairwise_task_masks_equal=True,primary_origin='each own observation end',
            label_columns=labels['decision','cumulative'],same_end_diagnostic_observation=5)))
    (out/'短观察期_看板数据.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print(pd.DataFrame(selection).to_string(index=False),flush=True)
    print('Own curves verified:',len(check),flush=True)
    return data
