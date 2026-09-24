"""Dense early-period refinements after the broad C-offset scan."""
from scripts.run_c_passive_offset import *


def main():
    dest=DEST/'offset_search'
    path=dest/'各档逐任务.parquet';before=sha(path)
    cached=pd.read_parquet(path).drop(columns='sweep_valid')
    have=set(cached.offset_ticks.unique())
    specs={1:list(range(101)),19:sorted(have|set(range(29)))}
    old=pd.read_parquet(SECONDS);old=old[old.common_valid]
    raw=pd.read_parquet(RAW);records=[];verified=0
    cfg=ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,task_range_seconds=3600,observation_seconds=(3.,)))
    for date,g in raw.groupby(raw.Date.astype(str),sort=True):
        tasks=old[(old.trade_date==date)&(old.policy=='C_limit_first')&(old.nominal_second<1140)]
        if tasks.empty:continue
        start=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai')
        d=SessionData(g[(g.Datetime>=start)&(g.Datetime<=start+pd.Timedelta(seconds=1160))],start,cfg)
        for r in tasks.itertuples():
            offsets=sorted(set(specs[1 if r.nominal_second<60 else 19])-have)
            signal=None if pd.isna(r.signal) else int(r.signal)
            for offset in offsets:
                n=simulate(d,r.task_seconds,int(r.direction),'C_limit_first',signal,c_limit_offset_ticks=offset)
                trace=n.pop('trace')
                for e in trace:
                    if e['event']=='submit' and e['kind']=='limit':assert e['limit_ticks']==d.p[d.source_index(e['time'])]-r.direction*offset
                if n['status']=='filled':
                    idx,price=independent_fill(d,trace,int(r.direction))
                    assert n['fill_ticks']==price and n['fill_seconds']==d.times[idx];verified+=1
                records.append(dict(trade_date=date,nominal_second=r.nominal_second,offset_ticks=offset,**n))
        print(date,'early refinement complete',flush=True)
    new=pd.DataFrame(records);new.to_parquet(dest/'早段补算逐任务.parquet',index=False)
    full=pd.concat([cached[cached.nominal_second<1140],new],ignore_index=True)
    summary=pd.read_csv(dest/'多档对照.csv');days=pd.read_csv(dest/'逐日结果.csv')
    rows=summary[~summary.minutes.isin(specs)].to_dict('records')
    dayrows=days[~days.minutes.isin(specs)].to_dict('records');audits={}
    key=['trade_date','nominal_second']
    for minutes,offsets in specs.items():
        f=full[(full.nominal_second<minutes*60)&full.offset_ticks.isin(offsets)]
        assert f.groupby(key).size().eq(len(offsets)*2).all()
        assert not f.duplicated(key+['direction','offset_ticks']).any()
        valid=f.assign(ok=f.status.eq('filled')).groupby(key).ok.all()
        f=f.merge(valid.rename('refine_valid'),on=key)
        f[~f.refine_valid].to_parquet(dest/f'前{minutes}分钟排除明细.parquet',index=False)
        f=f[f.refine_valid];base=old.merge(f[key].drop_duplicates(),on=key)
        audits[minutes]=dict(offsets=offsets,common_tasks=int(valid.sum()),lost_tasks=int((~valid).sum()))
        for grid in ['minute','second']:
            b=base if grid=='second' else base[base.nominal_second%60==0]
            g=f if grid=='second' else f[f.nominal_second%60==0]
            for side in ['买卖各半','买入','卖出']:
                bb=b if side=='买卖各半' else b[b.direction==(1 if side=='买入' else -1)]
                gg=g if side=='买卖各半' else g[g.direction==(1 if side=='买入' else -1)]
                costs=bb.groupby(['trade_date','policy']).cost_bp.mean().unstack()
                market=bb[bb.policy=='A_benchmark'].groupby('trade_date').market_cost_bp.mean()
                for offset,n in gg.groupby('offset_ticks'):
                    daily=n.assign(limit=n.fill_kind.eq('limit'),deadline=n.fill_stage.eq('deadline')).groupby('trade_date')[['cost_bp','limit','deadline','elapsed_seconds']].mean()
                    delta=costs.C_limit_first-daily.cost_bp;lo,hi=date_block_interval(delta)
                    rows.append(dict(minutes=minutes,grid=grid,side=side,offset_ticks=int(offset),valid_tasks=len(n[key].drop_duplicates()),
                        market=market.mean(),A=costs.A_benchmark.mean(),B=costs.B_observe_first.mean(),cost=daily.cost_bp.mean(),
                        saving_vs_C=delta.mean(),ci_low=lo,ci_high=hi,saving_vs_A=(costs.A_benchmark-daily.cost_bp).mean(),
                        saving_vs_B=(costs.B_observe_first-daily.cost_bp).mean(),saving_vs_market=(market-daily.cost_bp).mean(),
                        limit_rate=daily.limit.mean(),deadline_rate=daily.deadline.mean(),elapsed=daily.elapsed_seconds.mean()))
                    for date,r in daily.iterrows():dayrows.append(dict(minutes=minutes,grid=grid,side=side,offset_ticks=int(offset),trade_date=date,**r.to_dict(),saving_vs_C=delta.loc[date]))
    pd.DataFrame(rows).to_csv(dest/'细查后多档对照.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(dayrows).to_csv(dest/'细查后逐日结果.csv',index=False,encoding='utf-8-sig')
    assert sha(path)==before
    save_json(dest/'早段细查验收.json',dict(periods=audits,verified_fills=verified,cached_input_sha256=before,source_unchanged=True))


if __name__=='__main__':main()
