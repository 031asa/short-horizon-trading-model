"""Compare frozen minute strategies with immediate, equally delayed market orders."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_ic import SessionData,ICConfig
from utils.opening_schedule import ScheduleConfig
from utils.opening_two_stage import date_block_interval
from scripts.run_two_stage_execution import RAW,sha,NAMES
from scripts.run_opening_prediction import save_json

SOURCE=ROOT/'result/opening_execution/every_minute/共同任务成交.parquet'
OUT=SOURCE.parent/'market_comparison'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,RAW]}
    original=pd.read_parquet(SOURCE)
    keys=['trade_date','clock','direction']
    tasks=original.drop_duplicates(keys)
    assert len(tasks)==8392 and tasks.trade_date.nunique()==18
    raw=pd.read_parquet(RAW);raw=raw[raw.Date.astype(str).isin(tasks.trade_date.unique())]
    cfg=ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,task_range_seconds=7200,observation_seconds=(3.,)))
    rows=[]
    for date,g in raw.groupby(raw.Date.astype(str),sort=True):
        for part,start,end in [('AM','09:30','11:30'),('PM','13:00','15:00')]:
            opening=pd.Timestamp(date+' '+start,tz='Asia/Shanghai');closing=pd.Timestamp(date+' '+end,tz='Asia/Shanghai')
            d=SessionData(g[(g.Datetime>=opening)&(g.Datetime<=closing)],opening,cfg)
            for r in tasks[(tasks.trade_date==date)&(tasks.part==part)].itertuples():
                t=r.task_seconds;i=d.source_index(t)
                j=int(np.searchsorted(d.times,t,side='right'))+1
                valid=(i>=0 and j<d.n and t-d.times[i]<=.5+1e-10)
                if valid:
                    valid=all(d.valid[k][i:j+1].all() for k in ['price','book','queue','volume'])
                    valid=bool(valid and d.time_edge[i+1:j+1].all() and d.edge['volume'][i+1:j+1].all())
                if not valid:raise ValueError(f'Market control invalid on shared task: {date}, {r.clock}')
                price=float(d.a[j] if r.direction==1 else d.b[j])
                qty=d.qa[j] if r.direction==1 else d.qb[j]
                assert qty>=1 and d.p[i]==r.initial_ticks
                cost_ticks=r.direction*(price-d.p[i])
                rows.append(dict(trade_date=date,clock=r.clock,direction=r.direction,part=part,
                    task_seconds=t,initial_ticks=float(d.p[i]),market_fill_seconds=float(d.times[j]),
                    market_fill_ticks=price,market_cost_ticks=float(cost_ticks),
                    market_cost_bp=float(cost_ticks/d.p[i]*10000),
                    market_elapsed_seconds=float(d.times[j]-t),arrival_spread_ticks=float(d.a[j]-d.b[j])))
    market=pd.DataFrame(rows)
    assert len(market)==len(tasks) and not market.duplicated(keys).any()
    # For paired buy/sell with the same P0 and arrival, mean signed market cost
    # is exactly half the arrival spread divided by P0, regardless of price drift.
    pairs=market.groupby(['trade_date','clock']).agg(cost=('market_cost_bp','mean'),
        spread=('arrival_spread_ticks','first'),p0=('initial_ticks','first'),n=('direction','size'))
    assert pairs.n.eq(2).all()
    np.testing.assert_allclose(pairs.cost,pairs.spread/2/pairs.p0*10000,atol=1e-12,rtol=1e-12)
    paired=original.merge(market,on=keys+['part','task_seconds','initial_ticks'],validate='many_to_one')
    assert len(paired)==len(original)
    paired['market_saving_bp']=paired.market_cost_bp-paired.cost_bp
    delta=paired.market_cost_ticks-paired.cost_ticks
    paired['market_win']=delta>0;paired['market_loss']=delta<0;paired['market_tie']=delta==0
    summary=[]
    for scope,g in [('全天',paired),('上午',paired[paired.part=='AM']),('下午',paired[paired.part=='PM']),
                    ('09:30–09:59',paired[paired.clock.between('09:30','09:59')])]:
        for side,sub in [('买卖各半',g),('买入',g[g.direction==1]),('卖出',g[g.direction==-1])]:
            for policy,z in sub.groupby('policy'):
                daily=z.groupby('trade_date')[['cost_bp','market_cost_bp','market_saving_bp','market_win','market_loss','market_tie','elapsed_seconds','market_elapsed_seconds']].mean().sort_index()
                lo,hi=date_block_interval(daily.market_saving_bp)
                summary.append(dict(scope=scope,side=side,policy=policy,days=len(daily),n=len(z),
                    **{c:float(daily[c].mean()) for c in daily},low=float(lo),high=float(hi)))
    summary=pd.DataFrame(summary)
    clock=[]
    for (minute,policy),g in paired.groupby(['clock','policy']):
        day=g.groupby('trade_date')[['market_saving_bp','market_win','market_tie']].mean().sort_index()
        lo,hi=date_block_interval(day.market_saving_bp)
        clock.append(dict(clock=minute,policy=policy,n=len(g),days=len(day),
                          **{c:float(day[c].mean()) for c in day},low=float(lo),high=float(hi)))
    market.to_csv(OUT/'逐任务立即市价.csv',index=False,encoding='utf-8-sig')
    paired.to_parquet(OUT/'与立即市价配对.parquet',index=False)
    summary.to_csv(OUT/'市价对照总表.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(clock).to_csv(OUT/'每分钟市价对照.csv',index=False,encoding='utf-8-sig')
    paired.groupby(['trade_date','policy'])[['market_saving_bp','market_win','market_loss','market_tie']].mean().reset_index().to_csv(OUT/'逐日市价对照.csv',index=False,encoding='utf-8-sig')
    for p in [SOURCE,RAW]:assert sha(p)==hashes[str(p.relative_to(ROOT))]
    save_json(OUT/'验收与口径.json',dict(input_sha256=hashes,common_tasks=len(pairs),
        market_orders=len(market),strategy_comparisons=len(paired),no_extra_exclusions=True,
        price='buy ask1, sell bid1 at arrival; one contract; best quantity >=1',
        delay='second snapshot strictly after task origin, same delay as A/B/C',
        all_pair_spread_identities_verified=True,original_files_unchanged=True))
    lines=['# 与立即市价比较','',
        'M：任务开始即提交市价单，仍延迟至严格晚于决策的第二条快照成交，买入按卖一、卖出按买一。不是零延迟或按LastPrice成交。',
        '沿用全天每分钟实验的4196个共同任务、18个日期，买卖各1手。新增市价对照全部有效，无额外删除任务。',
        '成本统一相对任务起点LastPrice；节约=立即市价成本−策略成本，正数更好。胜率为严格优于立即市价，持平不算胜。按日期等权、买卖各半。','',
        '|范围|策略|立即市价成本bp|策略成本bp|较市价节约bp|胜率|持平率|95%日期块节约区间|',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in summary[summary.side=='买卖各半'].itertuples():
        lines.append(f'|{r.scope}|{NAMES[r.policy]}|{r.market_cost_bp:.4f}|{r.cost_bp:.4f}|{r.market_saving_bp:+.4f}|{r.market_win:.2%}|{r.market_tie:.2%}|[{r.low:.4f}, {r.high:.4f}]|')
    lines+=['','买卖来自同一行情；区间按5日日期块重采样，不将每笔当独立日期。',
        '原三策略与基准A的比较保持不变。本表新增M作为比较对象；预测准确率不等于执行胜率。',
        '仍是冻结开盘模型在已有日期上的探索；L1一手撮合代理不含手续费、排队、市场冲击和部分成交。']
    (OUT/'市价对照报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary[summary.side=='买卖各半'].to_string(index=False))


if __name__=='__main__':main()
