"""Describe the cost minimum and date sensitivity of the passive-offset sweep."""
from scripts.run_c_passive_offset import *


def analyze():
    dest=DEST/'offset_search'
    refined=(dest/'细查后多档对照.csv').exists()
    summary=pd.read_csv(dest/('细查后多档对照.csv' if refined else '多档对照.csv'))
    daily=pd.read_csv(dest/('细查后逐日结果.csv' if refined else '逐日结果.csv'))
    selected=[];stability=[];comparisons=[]
    for (minutes,grid,side),g in summary.groupby(['minutes','grid','side'],sort=False):
        g=g.sort_values('offset_ticks');best=g.loc[g.cost.idxmin()]
        h=daily[(daily.minutes==minutes)&(daily.grid==grid)&(daily.side==side)]
        matrix=h.pivot(index='trade_date',columns='offset_ticks',values='cost_bp').sort_index()
        near=g[g.cost<=best.cost+.01].offset_ticks.astype(int).tolist()
        loo=[]
        for day in matrix.index:
            k=int(matrix.drop(index=day).mean().idxmin());loo.append(k)
            stability.append(dict(minutes=minutes,grid=grid,side=side,kind='leave_one_date_out',date=str(day),best_offset=k))
        for fold,(start,end) in enumerate([(0,5),(5,10),(10,15),(15,18)],1):
            segment=matrix.iloc[start:end]
            stability.append(dict(minutes=minutes,grid=grid,side=side,kind='date_block',date=f'block{fold}',best_offset=int(segment.mean().idxmin())))
        k=int(best.offset_ticks)
        for alternative in matrix.columns:
            delta=matrix[alternative]-matrix[k];lo,hi=date_block_interval(delta)
            comparisons.append(dict(minutes=minutes,grid=grid,side=side,best_offset=k,alternative=int(alternative),saving=delta.mean(),low=lo,high=hi,
                note='descriptive paired interval after in-sample selection; not selection-adjusted'))
        selected.append(dict(minutes=minutes,grid=grid,side=side,offset=k,price_offset=k*.2,cost=best.cost,
            market=best.market,A=best.A,B=best.B,saving_vs_B=best.saving_vs_B,saving_vs_market=best.saving_vs_market,
            valid_tasks=int(best.valid_tasks),limit_rate=best.limit_rate,deadline_rate=best.deadline_rate,elapsed=best.elapsed,
            within_001bp=','.join(map(str,near)),leave_one_out_min=min(loo),leave_one_out_max=max(loo),
            at_search_boundary=k==int(g.offset_ticks.max())))
    chosen=pd.DataFrame(selected).sort_values(['minutes','grid','side'])
    chosen.to_csv(dest/'档位选择与稳定性.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(stability).to_csv(dest/'日期敏感性.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(comparisons).to_csv(dest/'最低点配对差异.csv',index=False,encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8.5))
    for ax,length in zip(axes.flat,[1,19,30,60]):
        g=summary[(summary.minutes==length)&(summary.grid=='second')&(summary.side=='买卖各半')].sort_values('offset_ticks')
        best=g.loc[g.cost.idxmin()]
        ax.plot(g.offset_ticks,g.cost,'o-',label='C：限价被动偏移',markersize=3)
        ax.axhline(best.B,color='#d58d21',ls='--',label='B：先观察3秒')
        ax.axhline(best.market,color='#25856a',ls=':',label='立即市价')
        ax.scatter([best.offset_ticks],[best.cost],color='crimson',zorder=5)
        ax.annotate(f'{int(best.offset_ticks)}档 / {best.cost:.3f} bp',(best.offset_ticks,best.cost),xytext=(5,12),textcoords='offset points')
        ax.set(title=f'前{length}分钟 · 每秒任务',xlabel='偏移档数（1档=0.2价格点）',ylabel='平均执行成本（bp）')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('C策略挂单偏移：共同任务、18日等权、买卖各半',fontsize=15)
    fig.tight_layout();fig.savefig(dest/'档位与成本曲线.png',dpi=160);plt.close(fig)
    lines=['# C档位成本最低点与稳定性','',
        '目标为平均执行成本最低，不额外给等待时间定价。全部档位共同任务，冻结模型，不改执行延迟和10秒期限。',
        '0.01bp范围仅描述曲线平坦程度，不是统计显著阈值。留一日与分日期块结果是敏感性检查，不是独立验证。配对区间未作事后选择校正。','',
        '|分钟范围|频率|历史最低档|成本bp|较B节约bp|较市价节约bp|距最低≤0.01bp档位|留一日最低档范围|',
        '|---|---|---:|---:|---:|---:|---|---|']
    for r in chosen[chosen.side=='买卖各半'].itertuples():
        lines.append(f'|{r.minutes}|{r.grid}|{r.offset}|{r.cost:.4f}|{r.saving_vs_B:+.4f}|{r.saving_vs_market:+.4f}|{r.within_001bp}|{r.leave_one_out_min}–{r.leave_one_out_max}|')
    (dest/'档位选择说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    f=pd.read_parquet(dest/'各档逐任务.parquet');f=f[f.sweep_valid]
    market_daily=f[f.offset_ticks==0].groupby('trade_date').market_cost_bp.mean()
    pair=f[f.offset_ticks.isin([14,19])].groupby(['trade_date','offset_ticks']).cost_bp.mean().unstack()
    intervals=[]
    for k in [14,19]:
        for name,delta in [('market',market_daily-pair[k]),('C19',pair[19]-pair[k])]:
            lo,hi=date_block_interval(delta)
            intervals.append(dict(offset=k,reference=name,saving=float(delta.mean()),low=float(lo),high=float(hi),
                note='descriptive; selected after seeing historical results; no selection correction'))
    save_json(dest/'首小时候选配对区间.json',intervals)
    print(chosen[chosen.side=='买卖各半'].to_string(index=False),flush=True)
    print(pd.DataFrame(stability).query("side=='买卖各半' and grid=='second' and kind=='date_block'").to_string(index=False),flush=True)


if __name__=='__main__':analyze()
