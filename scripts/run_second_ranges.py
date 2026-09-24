"""One task each second in the first hour; matched minute-grid comparison."""
from pathlib import Path
import os,sys,json
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ.setdefault(name,'1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_ic import SessionData,ICConfig
from utils.opening_schedule import ScheduleConfig
from utils.opening_task_window import bounded_features
from utils.opening_anchored import predict
from utils.opening_two_stage import POLICIES,simulate,date_block_interval
from scripts.run_two_stage_execution import RAW,MODELS,sha
from scripts.run_opening_prediction import save_json

OUT=ROOT/'result/opening_execution/opening_ranges'
SECONDS=OUT/'首小时逐秒全部执行.parquet'
MINUTES=ROOT/'result/opening_execution/every_minute/market_comparison/与立即市价配对.parquet'


def independent_fill(d,trace,side):
    commands=[e for e in trace if e['event']=='submit']
    for n,c in enumerate(commands):
        start=c['arrival_index']
        if c['kind']=='market':return start,float(d.a[start] if side==1 else d.b[start])
        end=commands[n+1]['arrival_index'] if n+1<len(commands) else d.n-1
        ids=np.arange(start,min(end+1,d.n));limit=c['limit_ticks']
        quote=d.a[ids]<=limit if side==1 else d.b[ids]>=limit
        cross=d.p[ids]<limit if side==1 else d.p[ids]>limit
        hit=quote|((d.dv[ids]>0)&cross)
        if hit.any():return int(ids[np.flatnonzero(hit)[0]]),float(limit)
    raise AssertionError('No independent fill')


def worker(job):
    date,g,m=job
    cfg=ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,task_range_seconds=3600,observation_seconds=(3.,)))
    d=SessionData(g,pd.Timestamp(date+' 09:30',tz='Asia/Shanghai'),cfg)
    assert max(m['train_dates'])<date
    records=[];signals=[];missing=[];checked=0;causal=0
    for s in range(3600):
        i=int(np.searchsorted(d.times,s,'left'))
        if i>=d.n or d.times[i]-s>.5+1e-10:
            missing.append(dict(trade_date=date,nominal_second=s,reason='no_snapshot_within_half_second'))
            continue
        t=float(d.times[i]);f=bounded_features(d,t,3)
        x=np.array([f[k] for k in m['features']])
        p=predict(m,x[None,:])[0] if np.isfinite(x).all() else np.full(3,np.nan)
        signal=int(p.argmax()-1) if np.isfinite(p).all() else None
        signals.append(dict(trade_date=date,nominal_second=s,task_seconds=t,signal=signal,
                            p_down=p[0],p_flat=p[1],p_up=p[2]))
        if s%120==0:
            clip=SessionData(d.frame[(d.times>=t)&(d.times<=t+3)],d.open_time,cfg)
            sf=bounded_features(clip,t,3)
            np.testing.assert_allclose(x,[sf[k] for k in m['features']],atol=0,rtol=0,equal_nan=True)
            causal+=1
        j=int(np.searchsorted(d.times,t,'right'))+1
        market_ok=(j<d.n and all(d.valid[k][i:j+1].all() for k in ['price','book','queue','volume'])
                   and d.time_edge[i+1:j+1].all() and d.edge['volume'][i+1:j+1].all())
        for side in [1,-1]:
            market_ticks=side*((d.a[j] if side==1 else d.b[j])-d.p[i]) if market_ok else np.nan
            market_bp=market_ticks/d.p[i]*10000 if market_ok else np.nan
            for policy in POLICIES:
                r=simulate(d,t,side,policy,signal);trace=r.pop('trace')
                if r['status']=='filled':
                    ii,price=independent_fill(d,trace,side)
                    assert d.times[ii]==r['fill_seconds'] and price==r['fill_ticks']
                    checked+=1
                records.append(dict(trade_date=date,nominal_second=s,fold=m['fold'],signal=signal,
                    market_valid=bool(market_ok),market_cost_bp=float(market_bp),market_cost_ticks=float(market_ticks),**r))
    return records,signals,missing,dict(trade_date=date,verified_fills=checked,boundary_checks=causal)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [RAW,MODELS,MINUTES]}
    models=[m for m in json.loads(MODELS.read_text(encoding='utf-8')) if m['window']==3 and m['method']=='selected']
    mapping={date:m for m in models for date in m['test_dates']}
    save_json(OUT/'逐秒实验口径.json',dict(input_sha256=hashes,dates=sorted(mapping),
        grid='each nominal second 0..3599; first snapshot at or after second, offset <=0.5s',
        model='frozen opening W3 selected models; no refit',observation_seconds=3,deadline_seconds=10,
        delay_snapshots=2,ranges_minutes=[1,19,30,60],quantity_per_side=1))
    raw=pd.read_parquet(RAW);jobs=[]
    for date,g in raw.groupby(raw.Date.astype(str),sort=True):
        if date not in mapping:continue
        start=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai')
        g=g[(g.Datetime>=start)&(g.Datetime<=start+pd.Timedelta(seconds=3650))]
        jobs.append((date,g,mapping[date]))
    records=[];signals=[];missing=[];checks=[]
    for job in jobs:
        r,s,m,c=worker(job);records.extend(r);signals.extend(s);missing.extend(m);checks.append(c)
        print(c,flush=True)
    f=pd.DataFrame(records).sort_values(['trade_date','nominal_second','direction','policy']).reset_index(drop=True)
    key=['trade_date','nominal_second']
    valid=f.assign(valid=f.status.eq('filled')&f.market_valid).groupby(key).valid.all()
    f=f.merge(valid.rename('common_valid'),on=key,validate='many_to_one')
    assert not f.duplicated(key+['direction','policy']).any()
    assert f.groupby(key).size().eq(6).all()
    f.to_parquet(SECONDS,index=False)
    pd.DataFrame(signals).sort_values(key).to_parquet(OUT/'首小时逐秒信号.parquet',index=False)
    pd.DataFrame(missing,columns=key+['reason']).sort_values(key).to_csv(OUT/'逐秒无行情任务.csv',index=False,encoding='utf-8-sig')
    f[~f.common_valid].to_parquet(OUT/'逐秒无效任务.parquet',index=False)
    common=f[f.common_valid]
    # Every minute task must be a strict subset of the new second grid, with
    # identical actual origin, signals, fills and costs (including exclusions).
    old=pd.read_parquet(MINUTES);old=old[(old.part=='AM')&(old.task_seconds<3600)].copy()
    old['nominal_second']=(old.clock.str[:2].astype(int)*60+old.clock.str[3:].astype(int)-570)*60
    selected=common[common.nominal_second%60==0]
    match=selected.merge(old,on=key+['direction','policy'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(match)==len(selected)==len(old)
    for col in ['task_seconds','fill_ticks','fill_seconds','cost_bp','market_cost_bp']:
        np.testing.assert_allclose(match[col+'_new'],match[col+'_old'],rtol=0,atol=0)
    for p in [RAW,MODELS,MINUTES]:assert sha(p)==hashes[str(p.relative_to(ROOT))]
    save_json(OUT/'逐秒验收.json',dict(nominal_tasks=18*3600,observed_tasks=len(valid),
        common_tasks=int(valid.sum()),no_quote_tasks=len(missing),invalid_tasks=int((~valid).sum()),
        verified_fills=sum(c['verified_fills'] for c in checks),
        boundary_checks=sum(c['boundary_checks'] for c in checks),minute_rows_exactly_replayed=len(match),
        original_inputs_unchanged=True,per_day=checks))
    report()


def report():
    second=pd.read_parquet(SECONDS);second=second[second.common_valid]
    minute=pd.read_parquet(MINUTES);minute=minute[(minute.part=='AM')&(minute.task_seconds<3600)].copy()
    minute['nominal_second']=(minute.clock.str[:2].astype(int)*60+minute.clock.str[3:].astype(int)-570)*60
    rows=[];detail=[];daily_rows=[]
    for length in [1,19,30,60]:
        for grid,frame,step in [('minute',minute,60),('second',second,1)]:
            g=frame[frame.nominal_second<length*60]
            for side,z in [('买卖各半',g),('买入',g[g.direction==1]),('卖出',g[g.direction==-1])]:
                costs=z.groupby(['trade_date','policy']).cost_bp.mean().unstack()
                market=z[z.policy=='A_benchmark'].groupby('trade_date').market_cost_bp.mean()
                a=costs.A_benchmark;b=costs.B_observe_first;c=costs.C_limit_first
                row=dict(minutes=length,grid=grid,side=side,days=len(costs),planned_tasks=18*length*60//step,
                    valid_tasks=len(g[['trade_date','nominal_second']].drop_duplicates()),
                    market=float(market.mean()),A=float(a.mean()),B=float(b.mean()),C=float(c.mean()),
                    B_vs_A=float((a-b).mean()),C_vs_A=float((a-c).mean()),
                    B_vs_market=float((market-b).mean()),C_vs_market=float((market-c).mean()))
                rows.append(row)
                for policy,sub in z.groupby('policy'):
                    sub=sub.copy();delta=sub.market_cost_ticks-sub.cost_ticks
                    sub['win']=delta>0;sub['loss']=delta<0;sub['tie']=delta==0
                    daily=sub.groupby('trade_date')[['cost_bp','market_cost_bp','win','loss','tie']].mean().sort_index()
                    saving=daily.market_cost_bp-daily.cost_bp;lo,hi=date_block_interval(saving)
                    saving_A=a-daily.cost_bp;alo,ahi=date_block_interval(saving_A)
                    detail.append(dict(minutes=length,grid=grid,side=side,policy=policy,
                        market_win=float(daily.win.mean()),market_loss=float(daily.loss.mean()),market_tie=float(daily.tie.mean()),
                        saving_vs_market=float(saving.mean()),market_low=float(lo),market_high=float(hi),
                        saving_vs_A=float(saving_A.mean()),A_low=float(alo),A_high=float(ahi)))
                    for date,r in daily.iterrows():daily_rows.append(dict(minutes=length,grid=grid,side=side,policy=policy,trade_date=date,**r.to_dict()))
                np.testing.assert_allclose(row['B_vs_A'],row['A']-row['B'],atol=1e-12)
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'时段成本对照.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(detail).to_csv(OUT/'胜率与区间.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(daily_rows).to_csv(OUT/'逐日时段对照.csv',index=False,encoding='utf-8-sig')
    lines=['# 四个时间段 × 两种任务频率','',
        '每分钟和每秒两种任务频率，均取对应整点／整秒后第一条快照为实际起点T，最大偏移0.5秒。不是将未来报价倒填到整点。',
        '09:30整点任务为每分钟抽样的第一行；首分钟内每秒任务计划18×60=1080个。前19分钟为[09:30,09:49)，前30分钟到10:00之前，前60分钟到10:30之前。',
        '每组均使用相同18个检验日、冻结W=3模型、观察3秒、10秒兜底、两条未来快照的下单延迟。立即市价在T提交并按到达对手一价成交。',
        '日期等权、买卖各半，成本越低越好；表中节约为正表示改善。各区间相互嵌套，不是独立样本。','',
        '|时间段|任务频率|计划／有效任务|市价bp|A bp|B bp|C bp|B较A节约bp|B较市价节约bp|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary[summary.side=='买卖各半'].itertuples():
        lines.append(f'|前{r.minutes}分钟|每'+('分钟' if r.grid=='minute' else '秒')+f'|{r.planned_tasks}/{r.valid_tasks}|{r.market:.4f}|{r.A:.4f}|{r.B:.4f}|{r.C:.4f}|{r.B_vs_A:+.4f}|{r.B_vs_market:+.4f}|')
    lines+=['','新增逐秒行与每分钟行的整分钟任务逐笔核对一致，完整原子数据、信号、排除任务及验收文件保存在同目录。',
        '历史截图1062任务是在第1–59整秒按当时最新报价发起，口径不同，保留在原首分钟逐秒市价对照.parquet。原来0.340bp结论不改写；新增表不是简单填补18行，也不能用旧数字代替新口径计算。',
        '未计排队、冲击、手续费或部分成交。分钟级与秒级的样本数不等于独立交易日数，区间按日期块计算。']
    (OUT/'时段比较说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary[summary.side=='买卖各半'].to_string(index=False),flush=True)


if __name__=='__main__':run()
