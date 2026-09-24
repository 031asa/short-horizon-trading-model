"""Fix initial C at 19 ticks; sweep only its T+3 limit, preserving old cohorts."""
from scripts.run_rebased_thresholds import *
from scripts.run_four_strategy_all_dates import full_path_safe
from concurrent.futures import ProcessPoolExecutor

DEST=OUT/'signal_offset_sweep'
OFFSETS=np.array(list(range(31))+[40,60],dtype=int)


def safe_sweep(d,t,side):
    """Independent vector scan on fully validated paths; old order has priority."""
    origin=d.source_index(t);p0=d.p[origin];old=p0-side*19
    first=int(np.searchsorted(d.times,t,'right'))+1
    arrival=int(np.searchsorted(d.times,t+3,'right'))+1
    deadline=int(np.searchsorted(d.times,t+10,'right'))+1
    def hit(ids,limits):
        quote=(d.a[ids,None]<=limits) if side==1 else (d.b[ids,None]>=limits)
        trade=(d.p[ids,None]<limits) if side==1 else (d.p[ids,None]>limits)
        return quote|((d.dv[ids,None]>0)&trade)
    ids=np.arange(first,arrival+1)
    oldhits=hit(ids,np.array([old]))[:,0]
    if oldhits.any():
        ix=np.full(len(OFFSETS),ids[np.flatnonzero(oldhits)[0]])
        return ix,np.full(len(OFFSETS),old),np.zeros(len(OFFSETS),int)
    limits=d.p[d.source_index(t+3)]-side*OFFSETS
    ids=np.arange(arrival,deadline+1);hits=hit(ids,limits)
    filled=hits.any(axis=0);ix=np.where(filled,ids[hits.argmax(axis=0)],deadline)
    prices=np.where(filled,limits,d.a[deadline] if side==1 else d.b[deadline])
    # Keeping exactly the old limit does not submit an update.
    stages=np.where(filled,np.where(limits==old,0,1),2)
    return ix,prices,stages


def worker(job):
    scope,date,frame,base,signals=job
    cfg=ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,task_range_seconds=3600,observation_seconds=(3.,)))
    d=SessionData(frame,pd.Timestamp(date+' 09:30',tz='Asia/Shanghai'),cfg)
    sig=signals.set_index('nominal_second').signal
    old=base[base.strategy.eq('C')].sort_values(['nominal_second','direction'])
    records=[];errors=[];verified=0;fast=0
    codes={('limit','initial'):0,('limit','signal'):1,('market','deadline'):2,('market','signal'):3}
    for n,r in enumerate(old.itertuples(index=False)):
        side=int(r.direction);signal=int(sig.loc[r.nominal_second]);t=float(r.task_seconds)
        if side*signal>0 or r.elapsed_seconds<=3:
            times=np.full(len(OFFSETS),r.fill_seconds);prices=np.full(len(OFFSETS),r.fill_ticks)
            stages=np.full(len(OFFSETS),codes[(r.fill_kind,r.fill_stage)])
        elif full_path_safe(d,t):
            ix,prices,stages=safe_sweep(d,t,side);times=d.times[ix];fast+=1
        else:
            times=np.full(len(OFFSETS),np.nan);prices=times.copy();stages=np.full(len(OFFSETS),-1)
        # Full event engine on unsafe paths and deterministic samples of all offsets.
        for j,k in enumerate(OFFSETS):
            if not np.isfinite(times[j]) or n%97==0:
                z=simulate(d,t,side,'C_limit_first',signal,c_limit_offset_ticks=19,c_signal_offset_ticks=int(k))
                if z['status']!='filled':
                    assert not np.isfinite(times[j])
                    errors.append(dict(trade_date=date,nominal_second=r.nominal_second,direction=side,k=int(k),reason=z['reason']))
                    continue
                ix2,p2=independent_fill(d,z['trace'],side)
                assert p2==z['fill_ticks'] and d.times[ix2]==z['fill_seconds']
                if np.isfinite(times[j]):
                    assert times[j]==z['fill_seconds'] and prices[j]==z['fill_ticks'] and stages[j]==codes[(z['fill_kind'],z['fill_stage'])]
                times[j]=z['fill_seconds'];prices[j]=z['fill_ticks'];stages[j]=codes[(z['fill_kind'],z['fill_stage'])];verified+=1
                for e in z['trace']:
                    if e['event']=='submit' and e['kind']=='limit':
                        expected=r.initial_ticks-side*19 if e['stage']=='initial' else d.p[d.source_index(t+3)]-side*k
                        assert e['limit_ticks']==expected
            records.append((date,r.nominal_second,side,int(k),times[j],prices[j],side*(prices[j]-r.initial_ticks)/r.initial_ticks*10000,times[j]-t,stages[j]))
    rows=pd.DataFrame(records,columns=['trade_date','nominal_second','direction','k','fill_seconds','fill_ticks','cost_bp','elapsed_seconds','method'])
    complete=rows.groupby('nominal_second').cost_bp.agg(['count','size'])
    good=complete.index[(complete['count']==2*len(OFFSETS))&(complete['size']==2*len(OFFSETS))]
    rows=rows[rows.nominal_second.isin(good)]
    assert not rows.duplicated(['nominal_second','direction','k']).any()
    # Exact full-cohort regression against both independent existing baselines.
    control=pd.read_parquet(CORRECT/scope/'四策略逐任务配对.parquet',filters=[('trade_date','==',date),('strategy','==','C')]).set_index(KEY)
    for k,reference in [(0,control),(19,old.set_index(KEY))]:
        a=rows[rows.k.eq(k)].set_index(KEY).sort_index();b=reference.loc[a.index]
        np.testing.assert_array_equal(a[['fill_seconds','fill_ticks','cost_bp','elapsed_seconds']],b[['fill_seconds','fill_ticks','cost_bp','elapsed_seconds']])
    folder=DEST/scope; (folder/'逐日逐单').mkdir(parents=True,exist_ok=True)
    rows.to_parquet(folder/'逐日逐单'/f'{date}.parquet',index=False)
    refs=base[base.strategy.isin(['M','A','B'])&base.nominal_second.isin(good)].copy()
    refs['k']=refs.strategy.map({'M':-1,'A':-2,'B':-3});refs['method']=[codes.get((a,b),4) for a,b in zip(refs.fill_kind,refs.fill_stage)]
    allrows=pd.concat([rows,refs[rows.columns]],ignore_index=True)
    for code,name in enumerate(['initial_limit','signal_limit','deadline_market','signal_market','immediate_market']):allrows[name]=allrows.method.eq(code)
    metrics=['cost_bp','elapsed_seconds','initial_limit','signal_limit','deadline_market','signal_market','immediate_market']
    daily=[]
    for minutes in [1,19,30,60]:
        for grid in ['minute','second']:
            g=allrows[allrows.nominal_second.lt(minutes*60)&((allrows.nominal_second%60==0) if grid=='minute' else True)]
            for side,label in [(0,'both'),(1,'buy'),(-1,'sell')]:
                z=g if side==0 else g[g.direction.eq(side)]
                if z.empty:continue
                a=z.groupby('k')[metrics].mean().reset_index();a['orders']=z.groupby('k').size().values
                a['trade_date']=date;a['minutes']=minutes;a['grid']=grid;a['side']=label;daily.append(a)
    pd.concat(daily,ignore_index=True).to_csv(folder/f'.daily_{date}.csv',index=False)
    receipt=dict(scope=scope,trade_date=date,tasks_before=len(old)//2,tasks_after=len(good),excluded=sorted(set(old.nominal_second)-set(good)),errors=errors,
                 vector_orders=fast,event_engine_checks=verified,exact_k0_k19=True)
    save_json(folder/f'.check_{date}.json',receipt)
    return receipt


def finish(scope):
    folder=DEST/scope
    daily=pd.concat([pd.read_csv(p) for p in sorted(folder.glob('.daily_*.csv'))],ignore_index=True)
    daily.to_csv(folder/'逐日结果.csv',index=False,encoding='utf-8-sig')
    output=[]
    for (minutes,grid,side),g in daily.groupby(['minutes','grid','side']):
        p=g.pivot(index='trade_date',columns='k',values='cost_bp')
        for k,a in g.groupby('k'):
            r=dict(scope=scope,minutes=int(minutes),grid=grid,side=side,k=int(k),days=len(a),orders=int(a.orders.sum()))
            for name in ['cost_bp','elapsed_seconds','initial_limit','signal_limit','deadline_market','signal_market','immediate_market']:r[name]=float(a[name].mean())
            assert abs(sum(r[name] for name in ['initial_limit','signal_limit','deadline_market','signal_market','immediate_market'])-1)<1e-10
            for ref,label in [(-1,'M'),(0,'new_C'),(19,'old_C')]:
                diff=p[ref]-p[k];r['saving_vs_'+label]=float(diff.mean());lo,hi=date_block_interval(diff)
                r['low_vs_'+label]=float(lo);r['high_vs_'+label]=float(hi)
            output.append(r)
    summary=pd.DataFrame(output);summary.to_csv(folder/'全部档位汇总.csv',index=False,encoding='utf-8-sig')
    checks=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('.check_*.json'))]
    primary=summary[summary.minutes.eq(60)&summary.grid.eq('second')&summary.side.eq('both')&summary.k.ge(0)]
    best=primary.sort_values(['cost_bp','k']).iloc[0].to_dict()
    save_json(folder/'验收与最低点.json',dict(checks=checks,primary_best=best,selection='descriptive minimum, not independent validation'))
    print(scope,'BEST',json.dumps(best,ensure_ascii=False),flush=True)


def main():
    DEST.mkdir(exist_ok=True)
    inputs=[RAW,MODELS,*SIGNALS.values(),*[p/'四策略逐任务配对.parquet' for p in SOURCES.values()],*[CORRECT/s/'四策略逐任务配对.parquet' for s in SOURCES]]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    registration=dict(offsets=OFFSETS.tolist(),initial_offset=19,signal='unchanged adverse market, otherwise limit with scanned offset',deadline=10,delay=2,
        primary='first60min eachsecond, date equal buy/sell half; descriptive minimum only',input_sha256=hashes)
    reg=DEST/'实验口径.json'
    if reg.exists():assert json.loads(reg.read_text(encoding='utf-8'))==registration
    else:save_json(reg,registration)
    if '--finish-only' not in sys.argv:
        raw=pd.read_parquet(RAW);old={s:pd.read_parquet(p/'四策略逐任务配对.parquet') for s,p in SOURCES.items()}
        signals={s:pd.read_parquet(p) for s,p in SIGNALS.items()};jobs=[]
        for date,frame in raw.groupby(raw.Date.astype(str),sort=True):
            start=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai')
            f=frame[(frame.Datetime>=start)&(frame.Datetime<=start+pd.Timedelta(seconds=3620))]
            f=f[[c for c in ['Datetime','LastPrice','BidPrice1','AskPrice1','BidVolume1','AskVolume1','Volume','source_row'] if c in f]].copy()
            for scope,b in old.items():
                g=b[b.trade_date.eq(date)]
                if not g.empty and not (DEST/scope/f'.check_{date}.json').exists():jobs.append((scope,date,f,g,signals[scope][signals[scope].trade_date.eq(date)]))
        with ProcessPoolExecutor(max_workers=2) as pool:
            for start in range(0,len(jobs),4):
                for r in pool.map(worker,jobs[start:start+4]):print(r['scope'],r['trade_date'],r['tasks_after'],'tasks; checked',r['event_engine_checks'],flush=True)
    for scope in SOURCES:finish(scope)
    for p in inputs:assert sha(p)==hashes[str(p.relative_to(ROOT))]
    save_json(DEST/'输入保持不变.json',dict(all_hashes_verified=True))


if __name__=='__main__':main()
