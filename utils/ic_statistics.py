"""Pairwise-correct matrix correlations and bounded-memory daily IC export."""
from itertools import product
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

STATUS=np.array(['ok','NOT_ENOUGH_PAIRS','CONSTANT_FACTOR','CONSTANT_LABEL'])
GROUPS=['session','factor','role','history_seconds','observation_seconds','horizon_seconds',
        'anchor','label_type','target','pair_set']


def _mask_groups(mask):
    groups={}
    for col,row in enumerate(np.packbits(mask.T,axis=1)):
        groups.setdefault(row.tobytes(),[]).append(col)
    return list(groups.values())


def pair_matrix(x,y,minimum_pairs=20):
    """Each cell ranks only its joint finite sample, including true zero returns."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    fx=np.isfinite(x);fy=np.isfinite(y);shape=(x.shape[1],y.shape[1])
    counts=fx.astype(np.int16).T@fy.astype(np.int16)
    zero=fx.astype(np.int16).T@((y==0)&fy).astype(np.int16)
    ic=np.full(shape,np.nan);rank=ic.copy();status=np.ones(shape,np.int8)
    for xs in _mask_groups(fx):
        for ys in _mask_groups(fy):
            mask=fx[:,xs[0]]&fy[:,ys[0]];n=int(mask.sum());ix=np.ix_(xs,ys)
            if n<minimum_pairs:continue
            a=x[mask][:,xs];b=y[mask][:,ys]
            ax=a-a.mean(axis=0);by=b-b.mean(axis=0)
            nx=np.linalg.norm(ax,axis=0);ny=np.linalg.norm(by,axis=0)
            cx=np.ptp(a,axis=0)==0;cy=np.ptp(b,axis=0)==0
            state=np.zeros((len(xs),len(ys)),np.int8)
            state[:,cy]=3;state[cx,:]=2;status[ix]=state
            with np.errstate(divide='ignore',invalid='ignore'):
                coef=(ax.T@by)/(nx[:,None]*ny[None,:])
            coef[state!=0]=np.nan;ic[ix]=np.clip(coef,-1,1)
            ra=pd.DataFrame(a).rank(method='average').to_numpy();rb=pd.DataFrame(b).rank(method='average').to_numpy()
            ra=ra-ra.mean(axis=0);rb=rb-rb.mean(axis=0)
            with np.errstate(divide='ignore',invalid='ignore'):
                coef=(ra.T@rb)/(np.linalg.norm(ra,axis=0)[:,None]*np.linalg.norm(rb,axis=0)[None,:])
            coef[state!=0]=np.nan;rank[ix]=np.clip(coef,-1,1)
    return counts,ic,rank,status,np.divide(zero,counts,out=np.full(shape,np.nan),where=counts>0)


def summarize_daily(daily,*unused_dates):
    """All available dates, equally weighted; date partitions are intentionally gone."""
    daily=daily.copy()
    if 'label_type' not in daily:daily['label_type']='cumulative'
    summary=daily.groupby(GROUPS,dropna=False,as_index=False).agg(
        scheduled_days=('trade_date','size'),valid_days=('rank_ic','count'),
        mean_ic=('ic','mean'),median_ic=('ic','median'),std_ic=('ic','std'),
        mean_rank_ic=('rank_ic','mean'),median_rank_ic=('rank_ic','median'),std_rank_ic=('rank_ic','std'),
        mean_pairs=('n_pairs','mean'),total_pairs=('n_pairs','sum'),
        mean_scheduled_coverage=('scheduled_coverage','mean'),mean_zero_return_share=('zero_return_share','mean'),
        positive_rank_ic_days=('rank_ic',lambda x:int(x.gt(0).sum())),
        negative_rank_ic_days=('rank_ic',lambda x:int(x.lt(0).sum())))
    summary['positive_day_share']=summary.positive_rank_ic_days/summary.valid_days.replace(0,np.nan)
    summary['negative_day_share']=summary.negative_rank_ic_days/summary.valid_days.replace(0,np.nan)
    summary['rank_ic_ir']=summary.mean_rank_ic/summary.std_rank_ic.replace(0,np.nan)
    summary.insert(0,'phase','all_sample')
    return summary


def _prepare(atomic,registry,config):
    names=registry.factor.tolist();observations=list(config.schedule.observation_seconds)
    days=[]
    for (date,session),day in atomic.groupby(['trade_date','session'],sort=True):
        groups=[day.loc[day.observation_seconds.eq(o)].sort_values('task_time') for o in observations]
        tasks=groups[0].task_time.tolist()
        assert all(g.task_time.tolist()==tasks for g in groups)
        x=np.stack([g[names].to_numpy(float) for g in groups])
        ys={}
        for anchor,kind in product(('decision','arrival'),('cumulative','incremental')):
            prefix=f'y_{anchor}_'+('incremental_' if kind=='incremental' else '')
            columns=[prefix+f'{h}s' for h in config.horizons_seconds]
            ys[(anchor,kind)]=np.stack([g[columns].to_numpy(float) for g in groups])
        days.append((date,session,x,ys,len(tasks)))
    return days


def _coordinate_frame(registry,combos,horizons):
    c,f,u=np.indices((len(combos),len(registry),len(horizons)))
    c=c.ravel();f=f.ravel();u=u.ravel()
    names=['observation_seconds','anchor','label_type','target','pair_set']
    data={name:np.array([combo[j] for combo in combos])[c] for j,name in enumerate(names)}
    data['observation_seconds']=data['observation_seconds'].astype(float)
    for name in ('factor','family_id','role','history_seconds'):
        data[name]=registry[name].to_numpy()[f]
    data['horizon_seconds']=np.array(horizons)[u]
    data['session']='AM'
    return pd.DataFrame(data)


def evaluate_all(atomic,registry,config,output):
    """Write daily records per family/day; aggregate dates without pooling snapshots."""
    registry=registry.loc[registry.evaluate].reset_index(drop=True)
    days=_prepare(atomic,registry,config)
    observations=list(config.schedule.observation_seconds);horizons=list(config.horizons_seconds)
    combos=list(product(observations,('decision','arrival'),('cumulative','incremental'),
                        ('signed','absolute'),('own','common_observations_horizons')))
    writer=None;summaries=[];total=0
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    try:
        for family,family_registry in registry.groupby('family_id',sort=False):
            indices=family_registry.index.to_numpy();meta=_coordinate_frame(family_registry,combos,horizons)
            shape=(len(days),len(combos),len(indices),len(horizons))
            metrics={name:np.full(shape,np.nan) for name in ('ic','rank_ic','zero_return_share')}
            metrics.update({name:np.zeros(shape,np.int16) for name in ('n_pairs','n_label','n_candidate')})
            states=np.ones(shape,np.int8)
            task_counts=[]
            for day_index,(date,session,xall,ys,n_tasks) in enumerate(days):
                x=xall[:,:,indices];task_counts.append(n_tasks)
                common={key:np.isfinite(x).all(axis=0)&np.isfinite(y).all(axis=(0,2))[:,None]
                        for key,y in ys.items()}
                for combo_index,(obs,anchor,kind,target,pair_set) in enumerate(combos):
                    oi=observations.index(obs);y=ys[(anchor,kind)][oi]
                    if target=='absolute':y=abs(y)
                    valid_tasks=common[(anchor,kind)] if pair_set!='own' else np.ones(x[oi].shape,bool)
                    xx=np.where(valid_tasks,x[oi],np.nan)
                    n,ic,rank,status,zero=pair_matrix(xx,y,config.minimum_pairs)
                    for name,value in [('n_pairs',n),('ic',ic),('rank_ic',rank),('zero_return_share',zero)]:metrics[name][day_index,combo_index]=value
                    metrics['n_candidate'][day_index,combo_index]=valid_tasks.sum(axis=0)[:,None]
                    metrics['n_label'][day_index,combo_index]=valid_tasks.astype(np.int16).T@np.isfinite(y).astype(np.int16)
                    states[day_index,combo_index]=status
                daily=meta.copy();daily['trade_date']=date;daily['n_scheduled']=n_tasks
                for name,values in metrics.items():daily[name]=values[day_index].ravel()
                daily['status']=STATUS[states[day_index].ravel()]
                daily['coverage']=daily.n_pairs/daily.n_label.replace(0,np.nan)
                daily['scheduled_coverage']=daily.n_pairs/n_tasks
                table=pa.Table.from_pandas(daily,preserve_index=False)
                if writer is None:writer=pq.ParquetWriter(output/'daily_ic.parquet',table.schema,compression='zstd',use_dictionary=True)
                writer.write_table(table);total+=len(daily)
            summary=meta.copy();summary.insert(0,'phase','all_sample')
            def assign(name,value):summary[name]=np.asarray(value).ravel()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)  # Empty/constant groups are explicit below.
                for name in ('ic','rank_ic'):
                    values=metrics[name]
                    for label,fn in [('mean',np.nanmean),('median',np.nanmedian)]:assign(label+'_'+name,fn(values,axis=0))
                    assign('std_'+name,np.nanstd(values,axis=0,ddof=1))
                assign('mean_zero_return_share',np.nanmean(metrics['zero_return_share'],axis=0))
                assign('rank_ic_p10',np.nanquantile(metrics['rank_ic'],.1,axis=0))
            assign('valid_days',np.isfinite(metrics['rank_ic']).sum(axis=0));summary['scheduled_days']=len(days)
            assign('valid_ic_days',np.isfinite(metrics['ic']).sum(axis=0))
            assign('mean_pairs',metrics['n_pairs'].mean(axis=0));assign('total_pairs',metrics['n_pairs'].sum(axis=0))
            assign('mean_scheduled_coverage',(metrics['n_pairs']/np.array(task_counts)[:,None,None,None]).mean(axis=0))
            for method in ('ic','rank_ic'):
                for sign in ('positive','negative'):
                    n=(metrics[method]>0).sum(axis=0) if sign=='positive' else (metrics[method]<0).sum(axis=0)
                    assign(sign+'_'+method+'_days',n)
            summary['positive_day_share']=summary.positive_rank_ic_days/summary.valid_days.replace(0,np.nan)
            summary['negative_day_share']=summary.negative_rank_ic_days/summary.valid_days.replace(0,np.nan)
            summary['rank_ic_ir']=summary.mean_rank_ic/summary.std_rank_ic.replace(0,np.nan)
            for code,label in enumerate(STATUS):assign('days_'+label.lower(),(states==code).sum(axis=0))
            summaries.append(summary)
            print(f'IC {family}: {len(family_registry)} outputs; {total:,} daily rows written',flush=True)
    finally:
        if writer is not None:writer.close()
    summary=pd.concat(summaries,ignore_index=True)
    summary.to_parquet(output/'summary_ic.parquet',index=False,compression='zstd')
    summary.to_csv(output/'summary_ic.csv',index=False,encoding='utf-8-sig')
    return summary,total
