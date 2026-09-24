"""Paired survivors: independently replay every T+3 limit, regardless of argmax."""
from scripts.run_signal_offset_sweep import *
from scripts.run_signal_offset_sweep import safe_sweep

DEST=OUT/'probability_offset_relation'
GRID=np.arange(61,dtype=int)


def worker(job):
    scope,date,frame,info=job
    cfg=ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,task_range_seconds=3600,observation_seconds=(3.,)))
    d=SessionData(frame,pd.Timestamp(date+' 09:30',tz='Asia/Shanghai'),cfg)
    rows=[];errors=[];checked=0
    for n,r in enumerate(info.itertuples(index=False)):
        t=float(r.task_seconds);side=int(r.direction)
        if full_path_safe(d,t):
            ix,prices,stages=safe_sweep(d,t,side,GRID);times=d.times[ix]
        else:
            times=np.full(len(GRID),np.nan);prices=times.copy();stages=np.full(len(GRID),-1)
        for j,k in enumerate(GRID):
            if not np.isfinite(times[j]) or n%113==0:
                # Counterfactual limit action: synthetic neutral signal only routes
                # the engine; the actual frozen model probabilities remain in info.
                z=simulate(d,t,side,'C_limit_first',0,c_limit_offset_ticks=19,c_signal_offset_ticks=int(k))
                if z['status']!='filled':
                    assert not np.isfinite(times[j]);errors.append(dict(trade_date=date,nominal_second=r.nominal_second,direction=side,k=int(k),reason=z['reason']));continue
                ii,pp=independent_fill(d,z['trace'],side);assert pp==z['fill_ticks'] and d.times[ii]==z['fill_seconds']
                code={('limit','initial'):0,('limit','signal'):1,('market','deadline'):2}[(z['fill_kind'],z['fill_stage'])]
                if np.isfinite(times[j]):assert times[j]==z['fill_seconds'] and prices[j]==pp and stages[j]==code
                times[j]=z['fill_seconds'];prices[j]=pp;stages[j]=code;checked+=1
            rows.append((date,r.nominal_second,side,int(k),times[j],prices[j],side*(prices[j]-r.initial_ticks)/r.initial_ticks*10000,times[j]-t,stages[j]))
    data=pd.DataFrame(rows,columns=KEY+['k','fill_seconds','fill_ticks','cost_bp','elapsed_seconds','method'])
    counts=data.groupby(TASK).cost_bp.count();valid=counts[counts.eq(2*len(GRID))].index
    data=data.set_index(TASK).loc[valid].reset_index()
    assert not data.duplicated(KEY+['k']).any()
    previous=OUT/'rebased_three_action_threshold'/scope
    refs=pd.read_parquet(previous/'阈值与基准_逐单结果.parquet',filters=[('trade_date','==',date),('policy','in',['R_lastprice','R_passive'])])
    for k,policy in [(0,'R_lastprice'),(19,'R_passive')]:
        a=data[data.k.eq(k)].set_index(KEY);b=refs[refs.policy.eq(policy)].set_index(KEY).loc[a.index]
        np.testing.assert_allclose(a[['fill_seconds','fill_ticks','cost_bp','elapsed_seconds']],b[['fill_seconds','fill_ticks','cost_bp','elapsed_seconds']],rtol=0,atol=1e-10)
    folder=DEST/scope/'逐日逐单';folder.mkdir(exist_ok=True,parents=True)
    data.to_parquet(folder/f'{date}.parquet',index=False)
    result=dict(scope=scope,date=date,before_tasks=len(info)//2,after_tasks=len(data)//(2*len(GRID)),errors=errors,event_checks=checked,k0_k19_verified=True)
    save_json(DEST/scope/f'.check_{date}.json',result)
    return result


def main():
    DEST.mkdir(exist_ok=True)
    inputs=[RAW,MODELS,*SIGNALS.values()]
    for scope in SOURCES:inputs.extend([OUT/'rebased_three_action_threshold'/scope/'任务与信号.parquet',OUT/'rebased_three_action_threshold'/scope/'阈值与基准_逐单结果.parquet',SOURCES[scope]/'四策略逐任务配对.parquet'])
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    registration=dict(initial_offset=19,signal_offsets=GRID.tolist(),cohort='same tasks with both sides surviving at T+3; no argmax action filter',action='force limit for all k versus same-task R_market; initial19 persists during delay; T+10 fallback',model='unchanged frozen probabilities',input_sha256=hashes)
    reg=DEST/'实验口径.json'
    if reg.exists():assert json.loads(reg.read_text(encoding='utf-8'))==registration
    else:save_json(reg,registration)
    raw=pd.read_parquet(RAW);infos={}
    for scope in SOURCES:
        p=OUT/'rebased_three_action_threshold'/scope
        info=pd.read_parquet(p/'任务与信号.parquet');alive=info.groupby(TASK).survivor3.all()
        info=info.set_index(TASK).loc[alive[alive].index].reset_index()
        old=pd.read_parquet(SOURCES[scope]/'四策略逐任务配对.parquet',filters=[('strategy','==','C')])
        info=info.merge(old[KEY+['task_seconds','initial_ticks']],on=KEY,validate='one_to_one')
        market=pd.read_parquet(p/'阈值与基准_逐单结果.parquet',filters=[('policy','==','R_market')])
        info=info.merge(market[KEY+['cost_bp','elapsed_seconds']].rename(columns={'cost_bp':'market3_cost_bp','elapsed_seconds':'market3_elapsed_seconds'}),on=KEY,validate='one_to_one')
        assert info.groupby(TASK).size().eq(2).all()
        (DEST/scope).mkdir(exist_ok=True);info.to_parquet(DEST/scope/'共同任务与概率.parquet',index=False);infos[scope]=info
    jobs=[]
    for date,frame in raw.groupby(raw.Date.astype(str),sort=True):
        start=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai');f=frame[(frame.Datetime>=start)&(frame.Datetime<=start+pd.Timedelta(seconds=3620))]
        f=f[[c for c in ['Datetime','LastPrice','BidPrice1','AskPrice1','BidVolume1','AskVolume1','Volume','source_row'] if c in f]].copy()
        for scope,info in infos.items():
            g=info[info.trade_date.eq(date)]
            if len(g) and not (DEST/scope/f'.check_{date}.json').exists():jobs.append((scope,date,f,g))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for start in range(0,len(jobs),4):
            for r in pool.map(worker,jobs[start:start+4]):print(r['scope'],r['date'],r['after_tasks'],'paired tasks',flush=True)
    checks=[]
    for scope in SOURCES:
        checks.extend(json.loads(p.read_text(encoding='utf-8')) for p in sorted((DEST/scope).glob('.check_*.json')))
    for p in inputs:assert sha(p)==hashes[str(p.relative_to(ROOT))]
    save_json(DEST/'撮合验收.json',dict(checks=checks,inputs_unchanged=True))


if __name__=='__main__':main()
