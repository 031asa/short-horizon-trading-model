"""Fixed C19, paired M/A/B/C charts over the established eight cohorts."""
from scripts.run_second_ranges import ROOT, OUT, SECONDS, RAW, MODELS, sha, save_json
from utils.opening_ic import SessionData, ICConfig
from utils.opening_schedule import ScheduleConfig
from utils.opening_two_stage import date_block_interval
import numpy as np
import pandas as pd

DEST = OUT/'four_strategy_c19'
SEARCH = OUT/'c_passive_offset/offset_search'
KEY = ['trade_date', 'nominal_second', 'direction']
TASK = KEY[:2]
ORDER = ['M', 'A', 'B', 'C']
NAMES = {'M': 'M 立即市价', 'A': 'A 初始限价', 'B': 'B 先观察3秒', 'C': 'C 被动19档'}
COLORS = {'M': '#177C70', 'A': '#4575B4', 'B': '#DB902C', 'C': '#865BB4'}
EXPECTED = {'minute': {1: 18, 19: 342, 30: 539, 60: 1060},
            'second': {1: 1080, 19: 20477, 30: 32171, 60: 62982}}
METHODS = ['immediate_market', 'initial_limit', 'signal_limit', 'signal_market', 'deadline_market']
METHOD_NAMES = ['立即市价', '初始限价成交', '3秒更新限价成交', '3秒市价成交', '10秒兜底市价成交']
METHOD_COLORS = ['#56616D', '#4D93C8', '#79C6C0', '#EEB05E', '#C77C99']
OUTCOMES = ['win', 'tie', 'loss']


def build_orders():
    source = SEARCH/'各档逐任务.parquet'
    reference = SEARCH/'细查后多档对照.csv'
    inputs = [RAW, SECONDS, source, reference, MODELS]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    c = pd.read_parquet(source, filters=[('offset_ticks', '==', 19), ('sweep_valid', '==', True)])
    cohort = c[TASK].drop_duplicates()
    assert len(cohort) == 62982 and c.trade_date.nunique() == 18
    assert not c.duplicated(KEY).any() and c.groupby(TASK).direction.nunique().eq(2).all()
    old = pd.read_parquet(SECONDS)
    old = old.merge(cohort, on=TASK, validate='many_to_one')
    assert old.common_valid.all() and old.status.eq('filled').all()
    fields = KEY+['task_seconds', 'initial_ticks', 'fill_seconds', 'fill_ticks', 'cost_ticks',
                  'cost_bp', 'elapsed_seconds', 'fill_kind', 'fill_stage', 'pending_cancelled']
    a = old[old.policy=='A_benchmark'][fields].copy(); a['strategy'] = 'A'
    b = old[old.policy=='B_observe_first'][fields].copy(); b['strategy'] = 'B'
    c = c[fields].copy(); c['strategy'] = 'C'
    tasks = old[old.policy=='A_benchmark']
    raw = pd.read_parquet(RAW); market_rows = []
    cfg = ICConfig(schedule=ScheduleConfig(cold_start_seconds=0, task_range_seconds=3600, observation_seconds=(3.,)))
    for date, frame in raw.groupby(raw.Date.astype(str), sort=True):
        tgroup = tasks[tasks.trade_date==date]
        if tgroup.empty:
            continue
        opening = pd.Timestamp(date+' 09:30', tz='Asia/Shanghai')
        frame = frame[(frame.Datetime>=opening)&(frame.Datetime<=opening+pd.Timedelta(seconds=3620))]
        d = SessionData(frame, opening, cfg)
        for r in tgroup.itertuples():
            t = r.task_seconds; i = d.source_index(t)
            j = int(np.searchsorted(d.times, t, side='right'))+1
            assert i>=0 and j<d.n and t-d.times[i]<=.5+1e-10
            assert d.times[j]>t and all(d.valid[k][i:j+1].all() for k in ['price','book','queue','volume'])
            assert d.time_edge[i+1:j+1].all() and d.edge['volume'][i+1:j+1].all()
            price, size = (d.a[j], d.qa[j]) if r.direction==1 else (d.b[j], d.qb[j])
            assert size>=1 and r.initial_ticks==d.p[i]
            ticks = r.direction*(price-d.p[i]); bp = ticks/d.p[i]*10000
            assert ticks==r.market_cost_ticks and bp==r.market_cost_bp
            market_rows.append(dict(trade_date=date, nominal_second=r.nominal_second, direction=r.direction,
                strategy='M', task_seconds=t, initial_ticks=float(d.p[i]), fill_seconds=float(d.times[j]),
                fill_ticks=float(price), cost_ticks=float(ticks), cost_bp=float(bp),
                elapsed_seconds=float(d.times[j]-t), fill_kind='market', fill_stage='immediate',
                pending_cancelled=False, arrival_spread_ticks=float(d.a[j]-d.b[j])))
        print(date, 'market price/time verified', len(tgroup), flush=True)
    m = pd.DataFrame(market_rows)
    pairs = m.groupby(TASK).agg(cost=('cost_bp','mean'), spread=('arrival_spread_ticks','first'), p0=('initial_ticks','first'))
    np.testing.assert_allclose(pairs.cost, pairs.spread/2/pairs.p0*10000, atol=1e-12, rtol=1e-12)
    orders = pd.concat([m,a,b,c], ignore_index=True)
    assert not orders.duplicated(KEY+['strategy']).any()
    assert orders.groupby(TASK).size().eq(8).all()
    assert orders.groupby(KEY).strategy.nunique().eq(4).all()
    for col in ['task_seconds','initial_ticks']:
        assert orders.groupby(KEY)[col].nunique().eq(1).all()
    control = m[KEY+['cost_ticks','cost_bp']].rename(columns={'cost_ticks':'market_cost_ticks','cost_bp':'market_cost_bp'})
    orders = orders.merge(control, on=KEY, validate='many_to_one')
    orders['saving_vs_market_bp'] = orders.market_cost_bp-orders.cost_bp
    delta = orders.market_cost_ticks-orders.cost_ticks
    orders['win'], orders['tie'], orders['loss'] = delta>0, delta==0, delta<0
    mapping = {('market','immediate'):'immediate_market', ('limit','initial'):'initial_limit',
               ('limit','signal'):'signal_limit', ('market','signal'):'signal_market', ('market','deadline'):'deadline_market'}
    orders['method'] = [mapping[(k,s)] for k,s in zip(orders.fill_kind,orders.fill_stage)]
    for method in METHODS:
        orders[method] = orders.method.eq(method)
    assert orders[METHODS].sum(axis=1).eq(1).all() and orders[OUTCOMES].sum(axis=1).eq(1).all()
    assert orders[orders.strategy=='M'].tie.all()
    assert orders[orders.strategy=='M'].immediate_market.all()
    assert not orders[orders.strategy=='A'][['signal_limit','signal_market']].any().any()
    assert not orders[orders.strategy=='B'].initial_limit.any()
    np.testing.assert_allclose(orders.elapsed_seconds,orders.fill_seconds-orders.task_seconds,atol=1e-12)
    assert (orders.elapsed_seconds>0).all()
    orders.sort_values(KEY+['strategy']).to_parquet(DEST/'四策略逐任务配对.parquet',index=False)
    for p in inputs:
        assert sha(p)==hashes[str(p.relative_to(ROOT))]
    return orders, pd.read_csv(reference), dict(input_sha256=hashes, original_inputs_unchanged=True,
        tasks=len(cohort), market_orders_verified=len(m), strategy_orders=len(orders), market_half_spread_identity_passed=True)


def aggregate(orders, reference):
    metrics = ['cost_bp','saving_vs_market_bp','elapsed_seconds']+METHODS+OUTCOMES
    rows=[]; days=[]; wide=[]
    for minutes in [1,19,30,60]:
        for grid in ['minute','second']:
            group=orders[orders.nominal_second<minutes*60]
            if grid=='minute':group=group[group.nominal_second%60==0]
            n=group[TASK].drop_duplicates().shape[0]
            assert n==EXPECTED[grid][minutes]
            for side,sub in [('买卖各半',group),('买入',group[group.direction==1]),('卖出',group[group.direction==-1])]:
                for strategy,z in sub.groupby('strategy'):
                    daily=z.groupby('trade_date')[metrics].mean().sort_index()
                    assert len(daily)==18
                    lo,hi=date_block_interval(daily.saving_vs_market_bp)
                    row=dict(minutes=minutes,grid=grid,side=side,strategy=strategy,days=18,tasks=n,orders=len(z),
                             **daily.mean().to_dict(),saving_ci_low=float(lo),saving_ci_high=float(hi))
                    rows.append(row)
                    for date,r in daily.iterrows():
                        days.append(dict(minutes=minutes,grid=grid,side=side,strategy=strategy,trade_date=date,**r.to_dict()))
                    ref=reference[(reference.minutes==minutes)&(reference.grid==grid)&(reference.side==side)&(reference.offset_ticks==19)].iloc[0]
                    expected=ref[{'M':'market','A':'A','B':'B','C':'cost'}[strategy]]
                    np.testing.assert_allclose(row['cost_bp'],expected,atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(sum(row[k] for k in METHODS),1,atol=1e-12)
                    np.testing.assert_allclose(sum(row[k] for k in OUTCOMES),1,atol=1e-12)
                    if strategy=='C':
                        np.testing.assert_allclose([row['elapsed_seconds'],row['initial_limit']+row['signal_limit'],row['deadline_market']],
                            [ref.elapsed,ref.limit_rate,ref.deadline_rate],atol=1e-12,rtol=1e-12)
            r=dict(时间段=f'前{minutes}分钟',任务频率='每分钟' if grid=='minute' else '每秒',有效任务数=n,日期数=18)
            for x in [v for v in rows if v['minutes']==minutes and v['grid']==grid and v['side']=='买卖各半']:
                for label,k in [('成本bp','cost_bp'),('较市价节约bp','saving_vs_market_bp'),('完成秒数','elapsed_seconds'),
                                ('成本胜率','win'),('平局率','tie'),('成本负率','loss'),('10秒市价成交比例','deadline_market')]:
                    r[f'{x["strategy"]}_{label}']=x[k]
            wide.append(r)
    summary=pd.DataFrame(rows);daily=pd.DataFrame(days)
    summary.to_csv(DEST/'四策略_分方向汇总.csv',index=False,encoding='utf-8-sig')
    daily.to_csv(DEST/'四策略_逐日结果.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(wide).to_csv(DEST/'四策略_8组总表.csv',index=False,encoding='utf-8-sig')
    return summary


def draw(summary, scope_label='18日期等权', expected=None, title_prefix='', show_days=False,
         c_description='固定被动 19 档', c_footer='C买价=LastPrice−3.8、卖价=LastPrice+3.8'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei'],'axes.unicode_minus':False,
        'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
    s=summary[summary.side=='买卖各半']
    expected=EXPECTED if expected is None else expected
    periods=[1,19,30,60];x=np.arange(4);width=.19
    handles=[Patch(color=COLORS[p],label=NAMES[p]) for p in ORDER]
    footer=scope_label+' · 买卖各半 · n为任务时点数，每时点双向各1手 · '+c_footer+' · 保留执行延迟'

    def data(grid,strategy):
        return s[(s.grid==grid)&(s.strategy==strategy)].set_index('minutes').loc[periods]

    def period_ticks(ax,grid):
        labels=[f'前{v}分钟\nn={expected[grid][v]:,}'+
                (f' · {int(data(grid,"M").loc[v,"days"])}日' if show_days else '') for v in periods]
        ax.set_xticks(x,labels)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)

    def save(fig,name):
        fig.savefig(DEST/f'{name}.png',dpi=200,facecolor='white')
        fig.savefig(DEST/f'{name}.svg',facecolor='white')
        plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(15,9.5))
    fig.suptitle(title_prefix+'四策略执行成本对比｜C '+c_description,fontsize=18,y=.975)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.941),ncol=4,frameon=False)
    for row,grid in enumerate(['minute','second']):
        label='每分钟任务' if grid=='minute' else '每秒任务'
        ax=axes[row,0];bx=axes[row,1]
        for j,p in enumerate(ORDER):
            d=data(grid,p);pos=x+(j-1.5)*width
            bars=ax.bar(pos,d.cost_bp,width*.94,color=COLORS[p])
            ax.bar_label(bars,labels=[f'{v:.3f}' for v in d.cost_bp],fontsize=8,padding=3)
            # The percentile interval need not contain its point estimate.
            bx.vlines(pos,d.saving_ci_low,d.saving_ci_high,color=COLORS[p],lw=1.8)
            bx.plot(pos,d.saving_vs_market_bp,'o',color=COLORS[p],markersize=5)
            bx.hlines(d.saving_ci_low,pos-.035,pos+.035,color=COLORS[p],lw=1)
            bx.hlines(d.saving_ci_high,pos-.035,pos+.035,color=COLORS[p],lw=1)
        ax.set_title(label+' · 平均成本（越低越好）',pad=13);ax.set_ylabel('执行成本 / bp')
        ax.set_ylim(0,max(data(grid,p).cost_bp.max() for p in ORDER)*1.22)
        bx.set_title(label+' · 较立即市价节约（正值更省）',pad=13);bx.set_ylabel('成本节约 / bp')
        bx.axhline(0,color='#333333',lw=.8,ls='--')
        period_ticks(ax,grid);period_ticks(bx,grid)
    fig.text(.5,.056,'误差线：95% 日期块重采样区间（5日块、5,000次）；已有日期上的描述性分析，未经档位选择校正。',ha='center',fontsize=9,color='#444444')
    fig.text(.5,.028,footer,ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.065,right=.975,bottom=.14,top=.84,hspace=.42,wspace=.23)
    save(fig,'01_执行成本与节约')

    fig,axes=plt.subplots(1,2,figsize=(15,5.7))
    fig.suptitle(title_prefix+'四策略完成时间对比｜从实际任务起点到成交',fontsize=18,y=.975)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.919),ncol=4,frameon=False)
    ymax=s.elapsed_seconds.max()*1.19
    for ax,grid in zip(axes,['minute','second']):
        for j,p in enumerate(ORDER):
            d=data(grid,p);bars=ax.bar(x+(j-1.5)*width,d.elapsed_seconds,width*.94,color=COLORS[p])
            ax.bar_label(bars,labels=[f'{v:.2f}' for v in d.elapsed_seconds],fontsize=8,padding=3)
        ax.set_title('每分钟任务' if grid=='minute' else '每秒任务',pad=12)
        ax.set_ylabel('平均完成时间 / 秒');ax.set_ylim(0,ymax);period_ticks(ax,grid)
    fig.text(.5,.06,footer,ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.065,right=.975,bottom=.23,top=.78,wspace=.21)
    save(fig,'02_平均完成时间')

    def stacked(name,title,columns,labels,colors,note):
        fig,axes=plt.subplots(2,4,figsize=(16,9))
        fig.suptitle(title_prefix+title,fontsize=18,y=.975)
        for i,grid in enumerate(['minute','second']):
            for j,minutes in enumerate(periods):
                ax=axes[i,j]
                d=s[(s.grid==grid)&(s.minutes==minutes)].set_index('strategy').loc[ORDER]
                bottom=np.zeros(4)
                for col,label,color in zip(columns,labels,colors):
                    vals=d[col].to_numpy()*100
                    bars=ax.bar(x,vals,bottom=bottom,color=color,width=.68,label=label)
                    for bar,v,b in zip(bars,vals,bottom):
                        if v>=5:
                            ax.text(bar.get_x()+bar.get_width()/2,b+v/2,f'{v:.1f}%',ha='center',va='center',fontsize=8,
                                    color='white' if color=='#56616D' else '#172333')
                    bottom+=vals
                ax.set_title(f'前{minutes}分钟 · '+('每分钟' if grid=='minute' else '每秒')+f'\nn={expected[grid][minutes]:,}'+
                             (f' · {int(d.loc["M","days"])}日' if show_days else ''),fontsize=11,pad=10)
                ax.set_xticks(x,ORDER);ax.set_ylim(0,102);ax.set_yticks([0,25,50,75,100])
                if j==0:ax.set_ylabel('成交比例 / %' if columns==METHODS else '任务比例 / %')
                for tick,p in zip(ax.get_xticklabels(),ORDER):tick.set_color(COLORS[p]);tick.set_fontweight('bold')
                ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
        fig.legend([Patch(color=c,label=l) for c,l in zip(colors,labels)],labels,
            loc='lower center',bbox_to_anchor=(.5,.074),ncol=min(5,len(labels)),frameon=False,fontsize=10)
        fig.text(.5,.049,note,ha='center',fontsize=9,color='#444444')
        fig.text(.5,.022,'M 立即市价  |  A 初始限价  |  B 先观察3秒  |  C '+c_description+'；'+scope_label+'、买卖各半。',ha='center',fontsize=9,color='#444444')
        fig.subplots_adjust(left=.055,right=.98,bottom=.20,top=.875,hspace=.40,wspace=.23)
        save(fig,name)
    stacked('03_成交方式构成','四策略成交方式构成｜按最终实际成交方式分类',METHODS,METHOD_NAMES,METHOD_COLORS,
            '兜底市价到达前若旧限价已成交，仍归入限价成交；小于5%的分项见完整表。')
    stacked('04_相对市价胜平负','四策略相对立即市价的执行成本胜／平／负',OUTCOMES,
            ['胜：成本更低','平：成本相同','负：成本更高'],['#83C8B4','#D8DBDF','#EBA6A1'],
            '执行成本胜率，不是涨跌预测准确率；M与自身比较，平局100%。小于5%的分项见完整表。')


def report(summary,audit):
    lines=['# 四策略完整对比：C固定被动19档','',
        'M立即市价；A初始LastPrice限价；B先观察3秒再按原信号下单；C初始被动19档限价、3秒按原信号处理。A/B/C均在10秒未成交时提交兜底市价。',
        'C的初始和第3秒限价：买价=LastPrice−3.8、卖价=LastPrice+3.8。买入遇上涨、卖出遇下跌则按既有规则转市价；平局或有利方向挂限价。',
        '全部下单保留两条严格未来快照的执行延迟，旧单在替换到达前仍可成交。10秒为兜底市价提交时点，实际成交可以稍晚。',
        '四个区间分别为[09:30,09:31)、[09:30,09:49)、[09:30,10:00)、[09:30,10:30)。实际T取整秒后首条快照、偏移不超过0.5秒。',
        '沿用档位搜索最终共同样本，保留其排除范围，即使只比较C19也不重新扩大样本；不按本次结果选择档位、不展示各时段最低点。',
        '18日期等权、买卖各半，各任务独立模拟买卖各1手。成本=direction×(成交价−起点LastPrice)/起点LastPrice×10000；节约=市价成本−策略成本。',
        '胜平负按逐任务成本的整数tick差判定，先逐日计算比例再日期等权。成交构成按实际成交方式分组，不按最后发出的指令分组。','',
        '|时间段|频率|任务数|M成本bp|A成本bp|B成本bp|C19成本bp|C较M节约bp|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    s=summary[summary.side=='买卖各半']
    for minutes in [1,19,30,60]:
        for grid in ['minute','second']:
            g=s[(s.minutes==minutes)&(s.grid==grid)].set_index('strategy')
            vals='|'.join(f'{g.loc[p,"cost_bp"]:.4f}' for p in ORDER)
            lines.append(f'|前{minutes}分钟|'+('每分钟' if grid=='minute' else '每秒')+f'|{int(g.loc["M","tasks"]):,}|{vals}|{g.loc["C","saving_vs_market_bp"]:+.4f}|')
    lines+=['','四张图分别提供PNG和SVG；8组总表为合并方向，分方向汇总含买入、卖出、买卖各半；逐日表及逐任务Parquet支持复算。',
        '误差线使用5日循环日期块、5,000次重采样、随机种子20260923。区间为已有日期上的描述性分析，未校正此前档位搜索；任务数量不是独立样本数量。',
        '本实验为L1代理撮合，不包含真实队列、部分成交、冲击和手续费；执行成本节约不等同于实盘盈利。']
    (DEST/'四策略对比说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit.update(expected_tasks=EXPECTED,strategies=ORDER,c_offset_ticks=19,tick_size=.2,
        frozen_models=True,no_strategy_retraining=True,old_summary_matched=True,
        outcome_and_fill_shares_sum_to_one=True,summary_rows=len(summary),
        date_bootstrap=dict(block=5,repeats=5000,seed=20260923),plot_files=sorted(p.name for p in DEST.glob('*.png')))
    save_json(DEST/'验收与口径.json',audit)


def bundle():
    import zipfile
    import hashlib
    target=ROOT/'result/四策略完整对比_C19.zip'
    files=sorted(p for p in DEST.iterdir() if p.is_file())
    digests={p.name:sha(p) for p in files}
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,arcname='四策略完整对比_C19/'+p.name)
    with zipfile.ZipFile(target) as z:
        assert len(z.namelist())==len(files)
        for name,digest in digests.items():
            assert hashlib.sha256(z.read('四策略完整对比_C19/'+name)).hexdigest()==digest
    print('Four-strategy archive verified:',len(files),'files',flush=True)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    orders,reference,audit=build_orders()
    summary=aggregate(orders,reference)
    draw(summary);report(summary,audit);bundle()
    print(summary[(summary.side=='买卖各半')&(summary.strategy=='C')][['minutes','grid','tasks','cost_bp','saving_vs_market_bp','elapsed_seconds','win','tie','loss']].to_string(index=False),flush=True)


if __name__=='__main__':main()
