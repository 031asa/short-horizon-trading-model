"""Fixed nested opening ranges with identical minute sampling, plus legacy second-grid control."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_ic import SessionData,ICConfig
from utils.opening_two_stage import date_block_interval
from scripts.run_two_stage_execution import RAW,sha
from scripts.run_opening_prediction import save_json

SOURCE=ROOT/'result/opening_execution/every_minute/market_comparison/与立即市价配对.parquet'
SECOND=ROOT/'result/opening_execution/two_stage_ablation/共同任务成交明细.parquet'
OUT=ROOT/'result/opening_execution/opening_ranges'


def second_grid():
    frame=pd.read_parquet(SECOND)
    raw=pd.read_parquet(RAW);raw=raw[raw.Date.astype(str).isin(frame.trade_date.unique())]
    rows=[]
    for date,g in raw.groupby(raw.Date.astype(str),sort=True):
        d=SessionData(g,pd.Timestamp(date+' 09:30',tz='Asia/Shanghai'),ICConfig())
        tasks=frame[frame.trade_date==date].drop_duplicates(['task_seconds','direction'])
        for r in tasks.itertuples():
            i=d.source_index(r.task_seconds);j=int(np.searchsorted(d.times,r.task_seconds,'right'))+1
            assert i>=0 and j<d.n and d.p[i]==r.initial_ticks
            assert d.time_edge[i+1:j+1].all() and d.edge['volume'][i+1:j+1].all()
            assert all(d.valid[k][i:j+1].all() for k in ['price','book','queue','volume'])
            price=d.a[j] if r.direction==1 else d.b[j]
            assert (d.qa[j] if r.direction==1 else d.qb[j])>=1
            ticks=r.direction*(price-d.p[i])
            rows.append(dict(trade_date=date,task_seconds=r.task_seconds,direction=r.direction,
                             market_cost_bp=float(ticks/d.p[i]*10000),market_cost_ticks=float(ticks)))
    frame=frame.merge(pd.DataFrame(rows),on=['trade_date','task_seconds','direction'],validate='many_to_one')
    frame['market_saving_bp']=frame.market_cost_bp-frame.cost_bp
    delta=frame.market_cost_ticks-frame.cost_ticks
    frame['market_win']=delta>0;frame['market_loss']=delta<0;frame['market_tie']=delta==0
    return frame


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,SECOND,RAW]}
    frame=pd.read_parquet(SOURCE)
    am=frame[frame.part=='AM'].copy()
    am['minute_index']=am.clock.str[:2].astype(int)*60+am.clock.str[3:].astype(int)-(9*60+30)
    legacy=second_grid()
    sets=[(f'前{n}分钟',n,am[am.minute_index.between(0,n-1)],'minute') for n in [1,19,30,60]]
    sets.append(('首分钟每秒发单（原截图）',1,legacy,'second'))
    rows=[];detail=[];daily_rows=[]
    for title,n,g,grid in sets:
        task_key=['trade_date','clock'] if grid=='minute' else ['trade_date','task_seconds']
        assert g.groupby(task_key).size().eq(6).all()
        for side,z in [('买卖各半',g),('买入',g[g.direction==1]),('卖出',g[g.direction==-1])]:
            costs=z.groupby(['trade_date','policy']).cost_bp.mean().unstack('policy')
            market=z[z.policy=='A_benchmark'].groupby('trade_date').market_cost_bp.mean()
            assert len(costs)==18 and market.index.equals(costs.index)
            a=costs.A_benchmark;b=costs.B_observe_first;c=costs.C_limit_first
            row=dict(range=title,grid=grid,minutes=n,side=side,days=len(costs),tasks=len(g.drop_duplicates(task_key)),
                directional_orders=len(z)//3,market=float(market.mean()),A=float(a.mean()),B=float(b.mean()),C=float(c.mean()),
                B_vs_A=float((a-b).mean()),C_vs_A=float((a-c).mean()),B_vs_market=float((market-b).mean()),C_vs_market=float((market-c).mean()))
            rows.append(row)
            for policy,sub in z.groupby('policy'):
                daily=sub.groupby('trade_date')[['cost_bp','market_cost_bp','market_saving_bp','market_win','market_loss','market_tie']].mean().sort_index()
                lo,hi=date_block_interval(daily.market_saving_bp)
                bsave=a-daily.cost_bp
                blo,bhi=date_block_interval(bsave)
                detail.append(dict(range=title,grid=grid,side=side,policy=policy,n=len(sub),
                    **{k:float(daily[k].mean()) for k in daily},market_low=float(lo),market_high=float(hi),
                    saving_vs_A=float(bsave.mean()),A_low=float(blo),A_high=float(bhi)))
                for date,r in daily.iterrows():
                    daily_rows.append(dict(range=title,grid=grid,side=side,policy=policy,trade_date=date,**r.to_dict()))
            np.testing.assert_allclose([row['B_vs_A'],row['B_vs_market']],[row['A']-row['B'],row['market']-row['B']],atol=1e-12)
    summary=pd.DataFrame(rows);detail=pd.DataFrame(detail)
    summary.to_csv(OUT/'时段成本对照.csv',index=False,encoding='utf-8-sig')
    detail.to_csv(OUT/'胜率与区间.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(daily_rows).to_csv(OUT/'逐日时段对照.csv',index=False,encoding='utf-8-sig')
    legacy.to_parquet(OUT/'原首分钟逐秒市价对照.parquet',index=False)
    np.testing.assert_allclose(summary[(summary.grid=='second')&(summary.side=='买卖各半')].B_vs_A.iloc[0],.3402213462146902,atol=1e-12)
    for p in [SOURCE,SECOND,RAW]:assert sha(p)==hashes[str(p.relative_to(ROOT))]
    save_json(OUT/'口径与复核.json',dict(input_sha256=hashes,minute_ranges=[1,19,30,60],
        windows='[09:30,09:31), [09:30,09:49), [09:30,10:00), [09:30,10:30)',
        main_sampling='one task per minute, first snapshot after minute <=0.5s',
        supplementary_sampling='original 09:30:01 through 09:30:59, one task per second',
        market='submitted at same T with same two future snapshot delay; opposite best quote',
        dates=18,day_equal_weight=True,all_shared_tasks_preserved=True,
        legacy_034bp_reproduced=True,original_files_unchanged=True))
    lines=['# 开盘前1、19、30、60分钟比较','',
        '主表统一为每分钟发起一个任务，取整点后首条快照（偏移≤0.5秒）作为T。18个检验日；买卖各半、日期等权。',
        '前19分钟严格指[09:30,09:49)，并非前20分钟；前半小时到10:00之前，前1小时到10:30之前。',
        '每个任务观察3秒、T+10兜底；立即市价也保留两条快照延迟。所有成本都以同一任务初始LastPrice计算。',
        '这四个区间相互包含，不是四个独立样本。持平不算胜；成本越低越好，节约为正越好。','',
        '|范围|抽样|任务数|立即市价bp|A成本bp|B成本bp|C成本bp|B较A节约bp|B较市价节约bp|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary[summary.side=='买卖各半'].itertuples():
        lines.append(f'|{r.range}|{r.grid}|{r.tasks}|{r.market:.4f}|{r.A:.4f}|{r.B:.4f}|{r.C:.4f}|{r.B_vs_A:+.4f}|{r.B_vs_market:+.4f}|')
    lines+=['','原截图的首分钟每秒任务单独补充，不与主表的每分钟任务混合。旧结论B较A节约0.340221bp复算一致，并为这组原任务补齐了立即市价对照。',
        '完整买卖方向、胜率、持平率及5日日期块95%区间见胜率与区间.csv。原数据和旧表未改动。',
        '这是既有日期上冻结开盘模型的探索结果；L1撮合代理不含手续费、排队、冲击和部分成交。']
    (OUT/'时段比较说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary[summary.side=='买卖各半'].to_string(index=False))
    print(detail[(detail.side=='买卖各半')&(detail.policy=='B_observe_first')].to_string(index=False))


if __name__=='__main__':main()
