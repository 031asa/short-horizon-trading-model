"""Verify and deliver the two initial-offset selection criteria as static figures."""
from scripts.run_initial_offset_selection import OUT, ROOT, SOURCE, sha, save_json
from utils.opening_initial_offset import METHODS
from utils.opening_two_stage import date_block_interval
import hashlib
import json
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'svg.fonttype': 'path', 'font.size': 11})
BLUE = '#376c9f'
ORANGE = '#d1872e'
GREEN = '#238777'
RED = '#ba605c'
SCOPE = {60: '开盘后首小时', 1: '开盘首分钟'}
SIDE = {'both': '买卖各半', 'buy': '买入', 'sell': '卖出'}
METHOD = {**METHODS, 'baseline19': '初始19档基准'}


def figsave(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT/f'{name}.{ext}', dpi=210, facecolor='white')
    plt.close(fig)


def decorate(ax, ylabel):
    ax.set(xlabel='初始被动档位 k（每档0.2价格点）', ylabel=ylabel, xlim=(-1, 61))
    ax.set_xticks(np.arange(0, 61, 5))
    ax.grid(alpha=.17)


def references(ax, early, total):
    ax.axvline(early, color=ORANGE, ls=':', lw=1.3)
    ax.axvline(total, color=GREEN, ls=':', lw=1.3)
    if 19 != total:
        ax.axvline(19, color='#888888', ls=':', lw=1)


def scope_figure(summary, choices, minutes):
    g = summary[summary.minutes.eq(minutes) & summary.side.eq('both')].set_index('k').sort_index()
    c = choices[choices.minutes.eq(minutes) & choices.side.eq('both')].set_index('method')
    ke, kt = int(c.loc['early', 'k']), int(c.loc['total', 'k'])
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), layout='constrained')
    ax = axes[0, 0]
    ax.plot(g.index, 100*g.early_fill, color=BLUE, lw=2)
    for k, label, color in [(ke, '提前贡献选择', ORANGE), (kt, '完整成本选择', GREEN)]:
        ax.scatter(k, 100*g.loc[k, 'early_fill'], color=color, s=60, zorder=5)
        ax.plot([], [], 'o', color=color, label=f'{label}：{k}档，{g.loc[k,"early_fill"]:.2%}')
    references(ax, ke, kt)
    decorate(ax, 'T+3秒内成交率（%）')
    ax.set_title('第一阶段：挂得越深，3秒内成交越少', fontsize=13)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, loc='upper right')
    ax = axes[0, 1]
    ax.plot(g.index, g.early_contribution_bp, color=ORANGE, lw=2)
    ax.fill_between(g.index, g.early_contribution_bp_low, g.early_contribution_bp_high, color=ORANGE, alpha=.14)
    ax.scatter(ke, g.loc[ke, 'early_contribution_bp'], color=ORANGE, s=70, zorder=5)
    ax.axhline(0, color='#888', lw=1)
    references(ax, ke, kt)
    decorate(ax, '提前成交贡献（bp，越高越好）')
    ax.set_title(f'方案一：提前贡献最高在 {ke} 档', fontsize=13)
    ax.text(.97, .94, f'{ke}档：{g.loc[ke,"early_contribution_bp"]:+.4f} bp\n19档：{g.loc[19,"early_contribution_bp"]:+.4f} bp', transform=ax.transAxes, ha='right', va='top', fontsize=11)
    ax = axes[1, 0]
    ax.plot(g.index, g.cost_bp, color=BLUE, lw=2, label='完整策略成本')
    ax.axhline(g.market_cost_bp.iloc[0], color='#777', ls='--', label=f'立即市价：{g.market_cost_bp.iloc[0]:.4f} bp')
    for k, label, color, marker in [(ke, '提前贡献选择', ORANGE, 'o'), (kt, '完整成本选择', GREEN, '*')]:
        ax.scatter(k, g.loc[k, 'cost_bp'], color=color, marker=marker, s=110, zorder=6,
                   label=f'{label} {k}档：{g.loc[k,"cost_bp"]:.4f} bp')
    references(ax, ke, kt)
    decorate(ax, '平均执行成本（bp，越低越好）')
    ax.set_title(f'方案二：完整成本最低在 {kt} 档', fontsize=13)
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    ax.plot(g.index, g.saving_vs_19_bp, color=GREEN, lw=2)
    ax.fill_between(g.index, g.saving_vs_19_low, g.saving_vs_19_high, color=GREEN, alpha=.17)
    ax.axhline(0, color='#777', lw=1)
    references(ax, ke, kt)
    decorate(ax, '相对初始19档的节约（bp，越高越好）')
    ax.set_title('配对比较：阴影为95%日期块区间', fontsize=13)
    ax.text(.97, .05, '参照19档也使用观察3秒／预测1秒模型', transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
    fig.suptitle(f'{SCOPE[minutes]}：初始挂多少档？两种选择方法对照\n观察3秒、预测1秒；第3秒限价固定19档，10秒兜底', fontsize=17, y=.99)
    fig.get_layout_engine().set(rect=(0, .09, 1, .81))
    timerange = '09:30–10:30' if minutes == 60 else '09:30–09:31'
    fig.text(.5, .025, f'{timerange} 每秒发起任务；18日、{int(g.tasks.iloc[0]):,}个共同任务，日期等权、买卖各半。全部初始档位0–60逐档回放，包含执行延迟。\n提前贡献=3秒内成交率×这些成交相对起点立即市价的平均节约。最优点为历史探索；区间逐档计算，未经选档校正。', ha='center', fontsize=10)
    figsave(fig, f'{SCOPE[minutes]}_两种初始档位选法')


def decomposition_figure(summary, choices):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), layout='constrained')
    for row, minutes in enumerate([60, 1]):
        g = summary[summary.minutes.eq(minutes) & summary.side.eq('both')].set_index('k')
        c = choices[choices.minutes.eq(minutes) & choices.side.eq('both')].set_index('method')
        ke, kt = int(c.loc['early','k']), int(c.loc['total','k'])
        ax = axes[row, 0]
        ax.plot(g.index, g.early_contribution_bp, color=ORANGE, lw=2, label='前3秒成交贡献')
        ax.plot(g.index, g.later_contribution_bp, color=RED, lw=2, label='其余订单后续贡献')
        ax.plot(g.index, g.saving_vs_market_bp, color=GREEN, lw=2.5, label='合计：相对立即市价节约')
        references(ax, ke, kt)
        ax.axhline(0, color='#777', lw=1)
        decorate(ax, '对全部任务的节约贡献（bp）')
        ax.set_title(f'{SCOPE[minutes]}：两部分相加才是最终效果', fontsize=13)
        ax.legend(fontsize=9, loc='lower right')
        ax = axes[row, 1]
        ks = list(dict.fromkeys([ke, 19, kt]))
        labels = [f'{k}档\n'+('提前贡献选择' if k==ke else '19档基准')+('／成本选择' if k==kt else '') for k in ks]
        if kt != 19 and kt != ke:
            labels[-1] = f'{kt}档\n完整成本选择'
        for j, side in enumerate(['buy', 'sell', 'both']):
            q = summary[summary.minutes.eq(minutes) & summary.side.eq(side)].set_index('k').loc[ks]
            xx = np.arange(len(ks))+(j-1)*.24
            ax.bar(xx, q.saving_vs_market_bp, width=.23, color=[BLUE, RED, GREEN][j], label=SIDE[side])
            for x, y in zip(xx, q.saving_vs_market_bp):
                ax.annotate(f'{y:+.3f}', (x, y), xytext=(0, 5 if y>=0 else -13), textcoords='offset points', ha='center', fontsize=9)
        ax.axhline(0, color='#777', lw=1)
        ax.set(xticks=np.arange(len(ks)), xticklabels=labels, ylabel='相对同方向立即市价的节约（bp）', title=f'{SCOPE[minutes]}：买卖使用同一个初始档位')
        ax.grid(axis='y', alpha=.17)
        ax.margins(y=.28)
        ax.legend(fontsize=9)
    fig.suptitle('为什么提前成交贡献最高，最终却未必最省？\n完整节约 = 前3秒成交贡献 + 未在3秒内成交订单的后续贡献', fontsize=17, y=.99)
    fig.get_layout_engine().set(rect=(0, .08, 1, .82))
    fig.text(.5, .025, '所有贡献都以全部共同任务为分母，正数节约、负数增加成本；后续贡献包括第3秒后原限价成交。\n买入与卖出没有分别挑选档位；模型、后续19档、执行延迟和10秒兜底全部固定。', ha='center', fontsize=10)
    figsave(fig, '提前与后续贡献分解')


def daily_figure(daily, choices):
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), layout='constrained')
    for ax, minutes in zip(axes, [60, 1]):
        q = daily[daily.minutes.eq(minutes) & daily.side.eq('both')]
        cost = q.pivot(index='trade_date', columns='k', values='cost_bp').sort_index()
        delta = cost.apply(lambda col: cost[19]-col)
        lim = max(float(np.abs(delta.to_numpy()).max()), .001)
        im = ax.imshow(delta, aspect='auto', cmap='RdBu', norm=TwoSlopeNorm(0, -lim, lim), interpolation='nearest')
        c = choices[choices.minutes.eq(minutes) & choices.side.eq('both')].set_index('method')
        ke, kt = int(c.loc['early','k']), int(c.loc['total','k'])
        ax.axvline(ke, color=ORANGE, lw=2, ls='--', label=f'全18日提前贡献选择：{ke}档')
        ax.axvline(kt, color='#191919', lw=2, ls=':', label=f'全18日完整成本选择：{kt}档')
        ax.set(xticks=np.arange(0,61,5), yticks=np.arange(len(cost)), yticklabels=[d[5:] for d in cost.index],
               xlabel='初始被动档位 k', ylabel='日期（2026年）')
        ax.set_title(f'{SCOPE[minutes]}：相对同日初始19档的节约\n橙色虚线：提前贡献选择 {ke} 档；黑色点线：完整成本选择 {kt} 档',fontsize=12)
        ax.tick_params(axis='y', labelsize=9)
        fig.colorbar(im, ax=ax, label='节约 bp：蓝色更省，红色更贵', pad=.015)
    fig.suptitle('逐日检查：不同日期是否支持相同档位？', fontsize=18, y=.99)
    fig.get_layout_engine().set(rect=(0, .07, 1, .88))
    fig.text(.5,.02,'买卖各半；上、下图分别设定色标范围，以免首分钟波动掩盖首小时差异。\n虚线档位由全部18日事后选出，不能将本图当作选档后的独立验证。',ha='center',fontsize=10)
    figsave(fig, '逐日初始档位节约热力图')


def verify(daily, summary, choices):
    """Recalculate from per-order prices, independently of the sweep aggregation."""
    checks = []
    for path in sorted((OUT/'逐任务').glob('*.parquet')):
        r = pd.read_parquet(path)
        date = path.stem
        assert r.trade_date.eq(date).all()
        assert not r.duplicated(['nominal_second', 'direction', 'k']).any()
        assert r.groupby('nominal_second').size().eq(122).all()
        cost = r.direction*(r.fill_ticks-r.initial_ticks)/r.initial_ticks*10000
        np.testing.assert_allclose(cost, r.cost_bp, atol=1e-12)
        np.testing.assert_allclose(r.fill_seconds-r.task_seconds, r.elapsed_seconds, atol=1e-12)
        early = r.fill_seconds.le(r.task_seconds+3+1e-10)
        gain = r.market_cost_bp-cost
        r['check_cost'] = cost
        r['check_gain'] = gain
        r['check_early'] = np.where(early,gain,0)
        r['check_late'] = np.where(early,0,gain)
        r['check_rate'] = early.astype(float)
        for minutes in [1,60]:
            for direction,side in [(0,'both'),(1,'buy'),(-1,'sell')]:
                sub = r[r.nominal_second.lt(minutes*60)]
                if direction: sub = sub[sub.direction.eq(direction)]
                recomputed = sub.groupby('k')[['check_cost','check_gain','check_early','check_late','check_rate']].mean()
                saved = daily[daily.trade_date.eq(date) & daily.minutes.eq(minutes) & daily.side.eq(side)].set_index('k').sort_index()
                np.testing.assert_allclose(recomputed, saved[['cost_bp','saving_vs_market_bp','early_contribution_bp','later_contribution_bp','early_fill']], atol=1e-12)
        checks.append(dict(trade_date=date,rows=len(r),tasks=r.nominal_second.nunique()))
    for (minutes,side),g in summary.groupby(['minutes','side']):
        for metric in ['cost_bp','market_cost_bp','early_contribution_bp','later_contribution_bp','early_fill']:
            recomputed = daily[daily.minutes.eq(minutes) & daily.side.eq(side)].groupby('k')[metric].mean()
            np.testing.assert_allclose(g.set_index('k').sort_index()[metric],recomputed,atol=1e-12)
        chosen = choices[choices.minutes.eq(minutes) & choices.side.eq(side)].set_index('method')
        assert int(chosen.loc['early','k']) == int(g.loc[g.early_contribution_bp.idxmax(),'k'])
        assert int(chosen.loc['total','k']) == int(g.loc[g.cost_bp.idxmin(),'k'])
    ci_checks = 0
    for r in choices.itertuples():
        g = daily[daily.minutes.eq(r.minutes) & daily.side.eq(r.side)]
        pivot = g.pivot(index='trade_date',columns='k',values='cost_bp').sort_index()
        delta = pivot[19]-pivot[r.k]
        np.testing.assert_allclose(delta.mean(),r.saving_vs_19_bp,atol=1e-12)
        np.testing.assert_allclose(date_block_interval(delta,seed=20260928),[r.saving_vs_19_low,r.saving_vs_19_high],atol=1e-12)
        ci_checks += 1
    save_json(OUT/'独立汇总验收.json',dict(per_day=checks, rows=sum(r['rows'] for r in checks),
        price_cost_reconstruction=True, independent_daily_cells=len(daily), selected_interval_checks=ci_checks,
        summaries_and_choices_reproduced=True))


def report(summary, choices, forward, records):
    audit = json.loads((OUT/'验收.json').read_text(encoding='utf-8'))
    tests = json.loads((OUT/'unit_test_verification.json').read_text(encoding='utf-8'))
    def main_table(minutes):
        g = choices[choices.minutes.eq(minutes) & choices.side.eq('both')].set_index('method')
        market = g.market_cost_bp.iloc[0]
        lines=[f'|立即市价|—|{market:.6f}|—|—|—|—|—|']
        for method in ['early','total','baseline19']:
            r=g.loc[method]
            lines.append(f'|{METHOD[method]}|{int(r.k)}|{r.cost_bp:.6f}|{r.early_fill:.2%}|{r.early_contribution_bp:+.6f}|{r.later_contribution_bp:+.6f}|{r.saving_vs_market_bp:+.6f}|{r.saving_vs_19_bp:+.6f}|')
        return '\n'.join(lines)
    table_header='|方案|初始档位|成本bp|3秒内成交率|提前贡献bp|后续贡献bp|相对市价节约bp|相对19档节约bp|\n|---|---:|---:|---:|---:|---:|---:|---:|'
    intervals=[]
    for r in choices[choices.side.eq('both')].itertuples():
        if r.method=='baseline19':continue
        intervals.append(f'|{SCOPE[r.minutes]}|{METHOD[r.method]}：{int(r.k)}档|{r.saving_vs_market_bp:+.6f} [{r.saving_vs_market_bp_low:+.6f}, {r.saving_vs_market_bp_high:+.6f}]|{r.saving_vs_19_bp:+.6f} [{r.saving_vs_19_low:+.6f}, {r.saving_vs_19_high:+.6f}]|')
    sides=[]
    for minutes in [60,1]:
        ks = choices[choices.minutes.eq(minutes) & choices.side.eq('both')].k.astype(int).unique()
        for k in ks:
            g = summary[summary.minutes.eq(minutes) & summary.k.eq(k)].set_index('side')
            sides.append(f'|{SCOPE[minutes]}|{k}|{g.loc["buy","saving_vs_market_bp"]:+.6f}|{g.loc["sell","saving_vs_market_bp"]:+.6f}|{g.loc["both","saving_vs_market_bp"]:+.6f}|')
    fw=[]
    for r in forward[forward.side.eq('both')].itertuples():
        ks=records[records.minutes.eq(r.minutes)&records.method.eq(r.method)].selected_k.astype(int).tolist()
        fw.append(f'|{SCOPE[r.minutes]}|{METHOD[r.method]}|{ks[0]} / {ks[1]}|{r.saving_vs_market_bp:+.6f} [{r.saving_vs_market_bp_low:+.6f}, {r.saving_vs_market_bp_high:+.6f}]|{r.saving_vs_19_bp:+.6f} [{r.saving_vs_19_bp_low:+.6f}, {r.saving_vs_19_bp_high:+.6f}]|')
    neighbors=summary[summary.minutes.eq(60)&summary.side.eq('both')&summary.k.isin([16,17,18,19,20,21,22])]
    near='\n'.join(f'|{int(r.k)}|{r.cost_bp:.6f}|{r.saving_vs_19_bp:+.6f}|[{r.saving_vs_19_low:+.6f}, {r.saving_vs_19_high:+.6f}]|' for r in neighbors.itertuples())
    text=f'''# 固定观察3秒：初始档位的两种选法

完成时间：2026-09-28。首小时指开盘后09:30–10:30，不是开盘前。先报告完整策略，再解释提前成交贡献。

结论：首小时“提前贡献最大”选6档，但最终比立即市价贵0.040947bp；“完整成本最低”仍选19档，相对市价省0.057082bp。开盘首分钟两种指标分别选13档和49档，二者仍未击败市价。不能用首小时结果替代首分钟。

## 本轮只改变初始档位

- 18个既有日期，首小时逐秒62,899个共同任务；首分钟1,080个。每个任务买卖各一笔。61个初始档位×两方向共{audit['rows']:,}条回放结果，不是{audit['rows']:,}个独立样本。
- 观察期=回看期3秒，冻结上一轮通过切换门槛的五因子未来1秒模型（四折参数与信号不变）。输入为最新OFI、报价位移、末价盘口位置、OFI方向、涨跌次数失衡。没有重训、没有按置信度筛任务。
- 初始买价=起点LastPrice−k×0.2，卖价=起点LastPrice+k×0.2，k扫描0–60的全部整数。第3秒如未成交：买遇预测涨、卖遇预测跌转市价，其余按第3秒LastPrice被动19档挂单；10秒提交兜底市价。
- 两条严格未来快照的延迟、旧限价在替换到达前仍有效、同快照旧单成交优先、同价不重发均保留。限价成交按限价结算，市价按到达时对手一价。成本基准为任务起点LastPrice，节约基准为同任务起点立即市价的真实模拟成交成本。
- **本轮“19档基准”已经使用观察3秒、预测1秒模型，不是预测3秒的原旧C。** 后续19档只是固定条件，本轮不判断后续19档是否最优，也未引入概率调档。

## 两个指标的定义

令I=1表示在T+3秒当时或之前已经成交，S=同任务立即市价成本−当前方案最终成本。

1. 你的选法：最大化 E[I×S] = P(I=1)×E[S|I=1]。前3秒未成交记该项贡献0，提前成交但比市价贵的负节约照实保留，不截断为0。
2. 完整成本选法：最小化 E[cost]，等价于最大化 E[S]。
3. E[S] = E[I×S] + E[(1−I)×S]，后项就是未在3秒内成交订单的后续贡献。

E采用先日期等权、再同日买卖各半的统一权重。不能将各日条件节约简单平均后再乘平均成交率；汇总条件节约=汇总提前贡献/汇总提前成交率。没有提前成交时条件节约缺失、提前贡献为0。

最优只指该历史数据与0–60档网格。完全相同指标（1e-12）优先接近19档，再取较小档；本轮最优点未命中60上界。没有向外自动扩展搜索。

## 首小时主结果

{table_header}
{main_table(60)}

6档的前3秒节约贡献约+0.353872bp，但未在3秒内成交订单后续贡献约−0.394819bp，合计−0.040947bp。19档前3秒贡献较少，但后续损失也显著较少，合计+0.057082bp。因此你的指标确实找到了更强的前段贡献，但固定目前后续规则后，它未能转化成更低的最终成本。这两种选择改变了留到后半段的订单群体，不能只拿早成交率评价全部执行。

![首小时](开盘后首小时_两种初始档位选法.png)

## 开盘首分钟单独结果

{table_header}
{main_table(1)}

49档虽然比同范围初始19档省0.165483bp，但比立即市价仍贵0.156236bp。49档只是本次历史最低点，附近较深档位也接近；提前成交率仅1.02%，策略已更接近先等待信号。不能据此宣布首分钟问题解决，或认为49档能稳定胜出。

![首分钟](开盘首分钟_两种初始档位选法.png)

## 配对区间与相邻档位

|范围|选择|相对市价节约与95%区间bp|相对初始19档节约与95%区间bp|
|---|---|---|---|
{chr(10).join(intervals)}

|首小时初始档位|完整成本bp|相对19档节约bp|95%配对区间bp|
|---|---:|---:|---|
{near}

19档是历史网格最低点；20档仅贵约0.000283bp，近邻多组差异区间跨0，不能证明19是唯一最优。以上为5日循环日期块、5,000次配对重采样，种子20260928；逐档区间未经61档选择和此前研究选择校正。

## 两侧分别看

以下仍是买卖共同选出的同一档位，没有分别挑选买入或卖出最优档。

|范围|初始档位|买入相对市价节约bp|卖出相对市价节约bp|买卖各半bp|
|---|---:|---:|---:|---:|
{chr(10).join(sides)}

首小时19档主要改善来自买入；卖出平均仍略贵于市价。分方向各自历史最低档另存`两种方法选档对照.csv`，仅供诊断，不作为不同方向分别优化后的策略成绩。

![贡献分解](提前与后续贡献分解.png)

## 只用较早日期选档的补充检查

前8日选档、随后5日看结果；前13日选档、随后5日看结果。两段共10个后续日期，每段仅用更早日期选k，买卖仍共同选择。

|范围|选法|两段所选档位|后10日相对市价节约与95%区间bp|后10日相对固定19档节约与95%区间bp|
|---|---|---|---|---|
{chr(10).join(fw)}

首小时完整成本规则选20/19档，后10日仍省市价，但比直接固定19档略贵；提前贡献规则选7/7档，后10日仍比固定19档贵。首分钟完整成本选59/49档，对固定19档的改善区间跨0、平均仍未击败市价。

预测期限1秒此前已经利用全部18日执行结果选择，所以这只是固定该期限后的前推敏感性检查，不是整套流程的全新样本外验证。

![逐日](逐日初始档位节约热力图.png)

## 验收、覆盖与文件

- 继承上轮62,899任务共同样本，本轮新增排除0；首分钟18×60=1,080。上轮相对原62,982底池排除83个五因子零分母任务，未填补。保留提前成交订单，不依未来标签有效性筛执行任务。
- 原事件撮合与最早成交独立扫描交叉核验{sum(r['event_engine_checks'] for r in audit['per_day']):,}条订单；初始19档{sum(r['k19_exact'] for r in audit['per_day']):,}笔逐笔复现上一轮1秒候选；立即市价从原行情相同延迟复核。
- 全部{audit['rows']:,}行由成交价格独立复算成本，逐日与汇总相互复现，买卖平均复现合并、成交构成加总100%、3秒成交率随档位单调不升、两段贡献精确相加；{tests['tests_run']}项测试通过。
- 行情、模型、信号、前轮结果与公共撮合实现SHA256不变。未重训、未修改模型默认配置、未上传远端、未新增看板。
- 输出含4张PNG/SVG、61档完整CSV、逐日表、分方向选择表、前推记录、逐日期逐任务Parquet、模型/信号来源及验收记录。`文件清单.csv`逐文件列大小和SHA256；独立压缩包位于`result/初始档位两种选法_观察3秒预测1秒.zip`。
- 原始L1撮合仍不含队列、冲击、费用与部分成交；报告描述历史执行成本，不等同实盘可实现收益。没有新增缺口或无法回放异常。

复现：先运行 `python -m scripts.run_initial_offset_selection`，再运行 `python scripts/test_opening.py result/opening_execution/initial_offset_selection/initial_offset_sweep`，最后运行 `python -m scripts.deliver_initial_offset_selection`。同目录原输入与口径须一致；不将旧实验缓存当作新配置。
'''
    (OUT/'初始档位两种选法报告.md').write_text(text,encoding='utf-8')


def package():
    context=OUT/'冻结信号来源'
    context.mkdir(exist_ok=True)
    for name in ['模型参数.json','模型切换判定.json']:
        shutil.copyfile(SOURCE/name,context/name)
    signals=pd.read_parquet(SOURCE/'三模型逐任务信号.parquet')
    signals=signals[signals.strategy.eq('fixed_1s') & signals.common]
    assert len(signals)==62899
    signals.to_parquet(context/'本次冻结信号.parquet',index=False)
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and '.checks' not in p.parts and p.name!='文件清单.csv')
    manifest=pd.DataFrame([dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    manifest.to_csv(OUT/'文件清单.csv',index=False,encoding='utf-8-sig')
    archive=ROOT/'result/初始档位两种选法_观察3秒预测1秒.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [*files,OUT/'文件清单.csv']:
            z.write(p,arcname=p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert len(z.infolist())==len(files)+1
        for r in manifest.itertuples():
            assert z.getinfo(r.path).file_size==r.bytes
            with z.open(r.path) as f:
                h=hashlib.file_digest(f,'sha256').hexdigest()
            assert h==r.sha256
    save_json(archive.with_suffix('.verification.json'),dict(file=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,
        sha256=sha(archive),entries=len(files)+1,all_entry_sizes_and_sha256_verified=True))
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,entries=len(files)+1),ensure_ascii=False),flush=True)


def main():
    summary=pd.read_csv(OUT/'全部档位汇总.csv')
    daily=pd.read_csv(OUT/'逐日档位结果.csv')
    choices=pd.read_csv(OUT/'两种方法选档对照.csv')
    forward=pd.read_csv(OUT/'前推表现汇总.csv')
    records=pd.read_csv(OUT/'前推选档记录.csv')
    verify(daily,summary,choices)
    for minutes in [60,1]: scope_figure(summary,choices,minutes)
    decomposition_figure(summary,choices)
    daily_figure(daily,choices)
    report(summary,choices,forward,records)
    package()


if __name__=='__main__':main()
