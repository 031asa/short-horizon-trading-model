"""Paired one-tick passive C experiment, using frozen signals and saved cohorts."""
from scripts.run_second_ranges import *

DEST=OUT/'c_passive_offset'


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [RAW,SECONDS,MODELS]}
    old=pd.read_parquet(SECONDS)
    old=old[old.common_valid].copy()
    raw=pd.read_parquet(RAW)
    records=[]; verified=0; replayed=0
    cfg=ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,task_range_seconds=3600,observation_seconds=(3.,)))
    for date,g in raw.groupby(raw.Date.astype(str),sort=True):
        baseline=old[(old.trade_date==date)&(old.policy=='C_limit_first')]
        if baseline.empty:continue
        start=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai')
        g=g[(g.Datetime>=start)&(g.Datetime<=start+pd.Timedelta(seconds=3650))]
        d=SessionData(g,start,cfg)
        for r in baseline.itertuples():
            signal=None if pd.isna(r.signal) else int(r.signal)
            # Replay zero offset exactly to validate engine backwards compatibility.
            zero=simulate(d,r.task_seconds,int(r.direction),'C_limit_first',signal)
            assert zero['status']==r.status
            for col in ['fill_ticks','fill_seconds','cost_bp']:assert zero[col]==getattr(r,col)
            replayed+=1
            new=simulate(d,r.task_seconds,int(r.direction),'C_limit_first',signal,c_limit_offset_ticks=1)
            trace=new.pop('trace')
            for e in trace:
                if e['event']=='submit' and e['kind']=='limit':
                    idx=d.source_index(e['time'])
                    assert e['limit_ticks']==d.p[idx]-r.direction
            if new['status']=='filled':
                idx,price=independent_fill(d,trace,int(r.direction))
                assert new['fill_ticks']==price and new['fill_seconds']==d.times[idx]
                verified+=1
            records.append(dict(trade_date=date,nominal_second=r.nominal_second,**new))
        print(date,len(baseline),flush=True)
    new=pd.DataFrame(records)
    key=['trade_date','nominal_second']
    valid=new.assign(ok=new.status.eq('filled')).groupby(key).ok.all()
    new=new.merge(valid.rename('paired_valid'),on=key)
    new.to_parquet(DEST/'C偏移1tick逐任务.parquet',index=False)
    new[~new.paired_valid].to_csv(DEST/'额外排除任务.csv',index=False,encoding='utf-8-sig')
    good=new[new.paired_valid]
    base=old.merge(good[key].drop_duplicates(),on=key,validate='many_to_one')
    rows=[]; dailyrows=[]
    for length in [1,19,30,60]:
        for grid in ['minute','second']:
            select=lambda x:x[(x.nominal_second<length*60)&((x.nominal_second%60==0) if grid=='minute' else True)]
            b=select(base); n=select(good)
            for side in ['买卖各半','买入','卖出']:
                bb=b if side=='买卖各半' else b[b.direction==(1 if side=='买入' else -1)]
                nn=n if side=='买卖各半' else n[n.direction==(1 if side=='买入' else -1)]
                daily=bb.groupby(['trade_date','policy']).cost_bp.mean().unstack()
                daily['market']=bb[bb.policy=='A_benchmark'].groupby('trade_date').market_cost_bp.mean()
                daily['C_offset']=nn.groupby('trade_date').cost_bp.mean()
                delta=daily.C_limit_first-daily.C_offset
                lo,hi=date_block_interval(delta)
                c=bb[bb.policy=='C_limit_first']
                rate=lambda x,col,val:float(x.assign(flag=x[col].eq(val)).groupby('trade_date').flag.mean().mean())
                row=dict(minutes=length,grid=grid,side=side,valid_tasks=len(n[key].drop_duplicates()),
                    market=daily.market.mean(),A=daily.A_benchmark.mean(),B=daily.B_observe_first.mean(),
                    C=daily.C_limit_first.mean(),C_offset=daily.C_offset.mean(),saving_vs_C=delta.mean(),
                    ci_low=lo,ci_high=hi,saving_vs_A=(daily.A_benchmark-daily.C_offset).mean(),
                    saving_vs_market=(daily.market-daily.C_offset).mean(),
                    old_limit_rate=rate(c,'fill_kind','limit'),new_limit_rate=rate(nn,'fill_kind','limit'),
                    old_deadline_rate=rate(c,'fill_stage','deadline'),new_deadline_rate=rate(nn,'fill_stage','deadline'),
                    old_elapsed=c.groupby('trade_date').elapsed_seconds.mean().mean(),
                    new_elapsed=nn.groupby('trade_date').elapsed_seconds.mean().mean())
                rows.append(row)
                for date,r in daily.iterrows():dailyrows.append(dict(minutes=length,grid=grid,side=side,trade_date=date,**r.to_dict()))
    summary=pd.DataFrame(rows)
    summary.to_csv(DEST/'限价偏移对照.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(dailyrows).to_csv(DEST/'逐日成本.csv',index=False,encoding='utf-8-sig')
    for p in [RAW,SECONDS,MODELS]:assert sha(p)==hashes[str(p.relative_to(ROOT))]
    save_json(DEST/'验收与口径.json',dict(offset_ticks=1,price_offset=.2,
        rule='C only: initial and T+3 limit = LastPrice - direction * 1 tick; unchanged signal, market and deadline',
        zero_offset_replayed=replayed,independent_fills_verified=verified,
        original_tasks=len(valid),common_tasks=int(valid.sum()),lost_tasks=int((~valid).sum()),
        input_sha256=hashes,all_inputs_unchanged=True))
    lines=['# C限价向被动方向偏移1 tick','',
        '买价=LastPrice−0.2，卖价=LastPrice+0.2；初始及第3秒限价均如此。冻结原模型和信号，保留执行延迟与10秒市价兜底。',
        '18日期等权、买卖各半；与原A/B/C/市价使用共同有效任务。每个任务独立模拟；成本单位bp。',
        f'额外排除{int((~valid).sum())}个任务，见额外排除任务.csv；同时从全部对照策略剔除。',
        '不是最优档位搜索，尚不包含排队、冲击及手续费。','',
        '|分钟范围|频率|有效任务|市价|A|B|原C|偏移C|较原C节约|95%日期块区间|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in summary[summary.side=='买卖各半'].itertuples():
        lines.append(f'|{r.minutes}|{r.grid}|{r.valid_tasks}|{r.market:.4f}|{r.A:.4f}|{r.B:.4f}|{r.C:.4f}|{r.C_offset:.4f}|{r.saving_vs_C:+.4f}|[{r.ci_low:.4f},{r.ci_high:.4f}]|')
    (DEST/'限价偏移说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary[summary.side=='买卖各半'].to_string(index=False),flush=True)


if __name__=='__main__':main()
