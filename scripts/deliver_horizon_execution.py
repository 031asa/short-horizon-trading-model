"""Static figures and analysis for the horizon comparison, without an offset sweep."""
from scripts.run_horizon_execution import OUT, ROOT, sha, save_json, PCOLS
from utils.opening_horizon import NAMES, MODELS, STRATEGIES, TASK
import json, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'svg.fonttype': 'path', 'font.size': 11})
COLORS = {'M': '#888888', 'legacy_3s': '#426b9d', 'fixed_1s': '#d08238', 'fixed_3s': '#298979'}
SHORT = {'M': '立即市价', 'legacy_3s': '原模型·3秒', 'fixed_1s': '五因子·1秒', 'fixed_3s': '五因子·3秒'}


def figsave(fig, name):
    for ext in ['png', 'svg']: fig.savefig(OUT/f'{name}.{ext}', dpi=210, facecolor='white')
    plt.close(fig)


def main():
    summary = pd.read_csv(OUT/'成本与节约汇总.csv')
    daily = pd.read_csv(OUT/'逐日成本.csv')
    pairs = pd.read_csv(OUT/'配对差异.csv')
    coverage = pd.read_csv(OUT/'逐日覆盖.csv')
    choice = json.loads((OUT/'模型切换判定.json').read_text(encoding='utf-8'))
    orders = pd.read_parquet(OUT/'四组逐任务成交.parquet')
    signals = pd.read_parquet(OUT/'三模型逐任务信号.parquet'); signals = signals[signals.common]
    main = summary[summary.minutes.eq(60) & summary.side.eq('both')].set_index('strategy').loc[STRATEGIES]
    primary = daily[daily.minutes.eq(60) & daily.side.eq('both')].pivot(index='trade_date', columns='strategy', values='cost_bp').sort_index()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    ax = axes[0, 0]; x = np.arange(4)
    ax.bar(x, main.cost_bp, color=[COLORS[s] for s in STRATEGIES], width=.6)
    for n, s in enumerate(STRATEGIES):
        ax.text(n, main.loc[s, 'cost_bp'], f'{main.loc[s,"cost_bp"]:.5f}', ha='center', va='bottom', fontsize=11)
    ax.set(xticks=x, xticklabels=[SHORT[s] for s in STRATEGIES], ylabel='平均执行成本（bp，越低越好）', title='同一批任务：完整执行成本')
    ax.set_ylim(0, max(main.cost_bp.max(), .01)*1.18); ax.grid(axis='y', alpha=.18)
    ax = axes[0, 1]
    show = pairs[pairs.minutes.eq(60) & pairs.side.eq('both')].reset_index(drop=True)
    for n, r in show.iterrows():
        color = COLORS[r.candidate]
        ax.plot([r.low_bp, r.high_bp], [n, n], color=color, lw=2)
        ax.scatter([r.gain_bp], [n], color=color, s=55)
        ax.annotate(f'{r.gain_bp:+.5f} [{r.low_bp:+.5f}, {r.high_bp:+.5f}]',
                    (r.gain_bp, n), xytext=(0, 14), textcoords='offset points', ha='center', fontsize=9)
    ax.axvline(0, color='#999', lw=1)
    ax.set(yticks=range(len(show)), yticklabels=[f'{SHORT[r.candidate]}\n相对{SHORT[r.reference]}' for r in show.itertuples()],
           xlabel='节约（bp，正数更好）；95%日期块区间', title='改善是否有区间支持？', ylim=(-.65, 2.8))
    ax.margins(x=.30); ax.grid(axis='x', alpha=.18)
    ax = axes[1, 0]
    for n, strategy in enumerate(MODELS):
        g = summary[summary.minutes.eq(60) & summary.strategy.eq(strategy)].set_index('side').loc[['buy', 'sell', 'both']]
        positions = np.arange(3)+(n-1)*.23
        ax.bar(positions, g.saving_vs_market_bp, width=.22, color=COLORS[strategy], label=SHORT[strategy])
        for xx, yy in zip(positions, g.saving_vs_market_bp):
            ax.annotate(f'{yy:+.4f}', (xx, yy), xytext=(0, 5 if yy >= 0 else -12), textcoords='offset points', ha='center', fontsize=8)
    ax.axhline(0, color='#888', lw=1); ax.set(xticks=range(3), xticklabels=['买入', '卖出', '买卖各半'],
        ylabel='相对立即市价的节约（bp）', title='买入、卖出分别是否改善？')
    ax.margins(y=.25); ax.legend(fontsize=9); ax.grid(axis='y', alpha=.18)
    ax = axes[1, 1]; delta = primary.legacy_3s - primary.fixed_1s
    ax.bar(np.arange(len(delta)), delta, color=['#298979' if v > 0 else '#bd6257' for v in delta])
    ax.axhline(0, color='#888', lw=1); ax.axhline(delta.mean(), color=COLORS['fixed_1s'], ls='--', label=f'日均 {delta.mean():+.5f} bp')
    ax.set(xticks=np.arange(len(delta)), xticklabels=[s[5:] for s in delta.index], ylabel='1秒候选相对原旧C的节约（bp）',
           title=f'逐日配对：{int((delta>1e-12).sum())}日改善／{int((delta< -1e-12).sum())}日变差')
    ax.tick_params(axis='x', rotation=60, labelsize=8); ax.legend(fontsize=9); ax.grid(axis='y', alpha=.18)
    decision = '切换为1秒候选' if choice['passes'] else '保留原旧C的3秒预测'
    fig.suptitle(f'观察3秒后，预测1秒是否改善旧C？\n18日首小时逐秒 · 初始19档／后续19档 · 判定：{decision}', fontsize=17, y=.99)
    fig.get_layout_engine().set(rect=(0, .08, 1, .82))
    fig.text(.5, .022, f'每组 {choice["common_tasks"]:,} 个共同任务，买卖各半、日期等权；底池 {choice["original_tasks"]:,} 个，覆盖 {choice["coverage"]:.2%}。\n两条后续快照延迟、10秒兜底不变；区间为既有日期探索，未校正历史筛选。没有扫描初始档位。', ha='center', fontsize=10)
    figsave(fig, '预测期限与旧C执行成本对比')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    ax = axes[0, 0]
    for n, s in enumerate(MODELS):
        g = summary[summary.side.eq('both') & summary.strategy.eq(s)].set_index('minutes').loc[[1, 60]]
        positions = np.array([0, 1]) + (n-1)*.23
        ax.bar(positions, g.saving_vs_market_bp, width=.22, color=COLORS[s], label=SHORT[s])
        for xx, yy in zip(positions, g.saving_vs_market_bp):
            ax.annotate(f'{yy:+.4f}', (xx, yy), xytext=(0, 5 if yy>=0 else -12), textcoords='offset points', ha='center', fontsize=9)
    ax.axhline(0, color='#888', lw=1)
    ax.set(xticks=[0, 1], xticklabels=['开盘首分钟', '首小时（主判定）'], ylabel='相对立即市价的节约（bp）', title='范围分开：首分钟尚未击败市价')
    ax.margins(y=.25); ax.legend(fontsize=9, loc='lower right'); ax.grid(axis='y', alpha=.18)
    ax = axes[0, 1]
    for n, s in enumerate(STRATEGIES):
        vals = summary[summary.strategy.eq(s) & summary.side.eq('both')].set_index('minutes').loc[[1, 60]].elapsed_seconds
        ax.bar(np.array([0,1])+(n-1.5)*.19, vals, width=.18, color=COLORS[s], label=SHORT[s])
    ax.set(xticks=[0,1], xticklabels=['开盘首分钟', '首小时'], ylabel='实际任务起点至成交（秒）', title='平均完成时间（包含执行延迟）')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=.18)
    ax = axes[1, 0]; bottom = np.zeros(4)
    kinds = [('immediate_market', '立即市价', '#888888'), ('initial_limit', '初始限价', '#426b9d'),
             ('signal_limit', '3秒更新限价', '#298979'), ('signal_market', '3秒信号市价', '#d08238'), ('deadline_market', '10秒兜底市价', '#bd6257')]
    for col, label, color in kinds:
        v = main[col].to_numpy()*100
        ax.bar(range(4), v, bottom=bottom, color=color, label=label)
        for i, y in enumerate(v):
            if y >= 5: ax.text(i, bottom[i]+y/2, f'{y:.1f}%', ha='center', va='center', fontsize=9, color='white')
        bottom += v
    np.testing.assert_allclose(bottom, 100, atol=1e-10)
    ax.set(xticks=range(4), xticklabels=[SHORT[s] for s in STRATEGIES], ylabel='成交构成（%）', title='首小时：最终由哪种订单成交？', ylim=(0, 110))
    ax.legend(fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(.5, -.10))
    ax = axes[1, 1]
    ax.bar(np.arange(len(coverage)), coverage.original_tasks, color='#e5e9ee', label='原底池任务')
    ax.bar(np.arange(len(coverage)), coverage.common_tasks, color='#426b9d', label='三组共同任务')
    ax.set(xticks=np.arange(len(coverage)), xticklabels=coverage.trade_date.str[5:], ylabel='任务数（不重复计算买卖）', title='逐日覆盖：缺失不填补')
    ax.tick_params(axis='x', rotation=60, labelsize=8); ax.legend(fontsize=9)
    fig.suptitle('预测期限比较：时段、成交方式与覆盖诊断', fontsize=18, y=.99)
    fig.get_layout_engine().set(rect=(0, .08, 1, .84))
    fig.text(.5, .018, '初始限价包括第3秒后、改单或兜底到达前由原单完成的成交；主结论按全部共同任务计算。\n五因子3秒对照沿用1秒候选选定的正则参数，只用于标签对照，不代表重新优化过的3秒模型。', ha='center', fontsize=10)
    figsave(fig, '预测期限执行诊断')

    signalwide = signals.pivot(index=TASK, columns='strategy', values='signal')
    action_rows = []
    for (date, side), g in orders[orders.strategy.eq('legacy_3s')].groupby(['trade_date', 'direction']):
        keys = pd.MultiIndex.from_frame(g[TASK]); s = signalwide.loc[keys]
        used = g.signal_used.to_numpy(bool)
        for other in ['fixed_1s', 'fixed_3s']:
            diff = (s.legacy_3s.to_numpy()*side > 0) != (s[other].to_numpy()*side > 0)
            action_rows.append(dict(trade_date=date, direction=side, candidate=other, tasks=len(g),
                signal_disagreement=float((s.legacy_3s != s[other]).mean()), surviving_tasks=int(used.sum()),
                action_changes=int((diff & used).sum()), change_rate_among_survivors=float(diff[used].mean()) if used.any() else np.nan))
    pd.DataFrame(action_rows).to_csv(OUT/'信号与动作差异.csv', index=False, encoding='utf-8-sig')
    accuracy = pd.read_csv(OUT/'预测诊断逐日.csv')
    acc = accuracy.groupby(['minutes', 'strategy', 'horizon'], as_index=False).agg(accuracy=('accuracy','mean'), balanced_accuracy=('balanced_accuracy','mean'), n=('n','sum'))
    acc.to_csv(OUT/'预测准确率诊断.csv', index=False, encoding='utf-8-sig')
    table = []
    for s, r in main.iterrows():
        table.append(f'|{NAMES[s]}|{r.cost_bp:.6f}|{r.saving_vs_market_bp:+.6f}|{r.saving_vs_legacy_bp:+.6f}|{r.elapsed_seconds:.3f}|')
    rows_minute = summary[summary.minutes.eq(1) & summary.side.eq('both')].set_index('strategy').loc[STRATEGIES]
    first_table = '\n'.join(f'|{NAMES[s]}|{r.cost_bp:.6f}|{r.saving_vs_market_bp:+.6f}|' for s,r in rows_minute.iterrows())
    paired_table = '\n'.join(f'|{SHORT[r.candidate]} 相对 {SHORT[r.reference]}|{r.gain_bp:+.6f}|[{r.low_bp:+.6f}, {r.high_bp:+.6f}]|{r.positive_days}/{r.negative_days}/{r.zero_days}|' for r in show.itertuples())
    first_gain = pairs[pairs.minutes.eq(1) & pairs.side.eq('both') & pairs.reference.eq('legacy_3s') & pairs.candidate.eq('fixed_1s')].iloc[0]
    folds = pd.read_csv(OUT/'四折表现.csv')
    foldcost = folds[folds.minutes.eq(60) & folds.side.eq('both')].pivot(index='fold', columns='strategy', values='cost_bp')
    fold_table = '\n'.join(f'|{idx}|{r.legacy_3s:.6f}|{r.fixed_1s:.6f}|{r.fixed_3s:.6f}|{r.legacy_3s-r.fixed_1s:+.6f}|' for idx,r in foldcost.iterrows())
    exclusions = pd.read_csv(OUT/'排除原因汇总.csv')
    reason_text = '\n'.join(f'- `{r.reason}`：{r.tasks}个任务。' for r in exclusions.itertuples()) or '- 无新增排除。'
    report = f'''# 观察3秒，预测1秒能否改善旧C

结论：**{decision}**。1秒候选相对原旧C节约{choice['mean_saving_bp']:+.6f}bp，95%日期块区间[{choice['low_bp']:+.6f}, {choice['high_bp']:+.6f}]bp。只依据首小时买卖合并配对结果判定；同因子3秒对照不参与择优。没有扫描初始档位，没有改变执行模型默认配置。

## 策略与训练

- 三组执行都为初始被动19档，第3秒买遇预测涨、卖遇预测跌则市价，其余按当时LastPrice被动19档；第10秒兜底提交市价。单位订单、两条严格未来快照延迟；限价报价触及或有成交量的LastPrice严格穿价都按限价结算。旧单在替换到达前有效，旧单成交优先。
- 原旧C复用原四折W3 selected模型，预测未来3秒，各折因子可能不同；1秒候选复用观察期实验W3固定五因子模型，预测未来1秒。均用argmax三分类信号，真实不变保留，不按置信度筛选。
- 同因子3秒对照与1秒候选使用完全相同训练任务、五因子、日期权重、标准化、每折C；仅更换标签重新拟合。C依次为1、0.1、0.1、10，沿用此前为1秒选出的值，不能把对照称为重新优化的最佳3秒模型。
- 前20/25/30/35日训练，后5/5/5/3日检验；所有模型仅以开盘首分钟任务训练，迁移到首小时，不用首小时成本拟合或调参。候选训练样本保留上一轮跨1–5秒窗口的共同任务，与旧模型训练样本不同；候选与原旧C是整套模型比较，同因子对照用于分辨标签变化。

## 同任务主结果：18日首小时逐秒

|策略|执行成本bp|相对市价节约bp|相对原旧C节约bp|平均完成秒|
|---|---:|---:|---:|---:|
{chr(10).join(table)}

|比较|平均节约bp|95%日期块区间bp|改善/变差/相同日期|
|---|---:|---|---:|
{paired_table}

每组{choice['common_tasks']:,}个任务、{choice['common_tasks']*2:,}个买卖订单，日期等权、买卖各半。原底池62,982任务，新增排除{choice['original_tasks']-choice['common_tasks']:,}，覆盖{choice['coverage']:.2%}。原底池成本已独立复现：旧C约0.663587bp、市价约0.691888bp；本表因共同样本变化重新计算，不混用旧汇总。

|折|原旧C成本bp|1秒候选成本bp|同因子3秒成本bp|1秒相对原旧C节约bp|
|---|---:|---:|---:|---:|
{fold_table}

## 开盘首分钟（次要结果，不改变主判定）

|策略|成本bp|相对市价节约bp|
|---|---:|---:|
{first_table}

共同任务{int(rows_minute.tasks.iloc[0])}个。原观察期预测报告的1,035任务是跨观察窗口的预测共同样本，本轮是三模型执行共同样本，不能直接混比。

仅看开盘首分钟，1秒候选相对原旧C节约{first_gain.gain_bp:+.6f}bp，区间[{first_gain.low_bp:+.6f}, {first_gain.high_bp:+.6f}]bp，未体现改善；三组C在该范围的平均成本均高于立即市价。首小时结论不能直接替代首分钟结论。

## 覆盖与解释边界

{reason_text}

- 取三组信号可计算且两方向全策略可完成回放的交集。第3秒前已成交订单也保留在共同样本中；因信号缺失被排除的任务另列，不把它们伪造为不变信号。未按1秒或3秒未来标签有效性筛执行任务。
- 两条后续快照延迟与1秒期限接近，信号目标起点保持观察结束，没有偷偷移到订单生效后。完整执行成本已包含延迟；高1秒准确率不必然改善执行成本。
- 区间使用5日循环日期块、5,000次配对重采样，种子20260928；不是每笔独立抽样，不代表未来盈利概率。所有日期此前已用于研究，区间未校正历史筛选。
- L1模拟保留无排队、无冲击、无费用、无部分成交限制。此结论是历史执行成本探索，不是实盘保证。

## 验收与交付

原旧C与市价逐笔复算原底池；三组初始限价相同，决策前成交一致。独立成交复核、观察区间外扰动、开盘五因子复现、原始1秒参数复现、训练检验日期隔离及文件SHA256保护均有验收记录。

图表、汇总与逐日表、预测诊断、逐任务信号/特征/成交、控制模型参数、训练任务、切换判定、排除清单均保存本目录。后续挂档实验未执行。
'''
    (OUT/'预测期限比较报告.md').write_text(report, encoding='utf-8')
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != '文件清单.csv')
    manifest = pd.DataFrame([dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in files])
    manifest.to_csv(OUT/'文件清单.csv', index=False, encoding='utf-8-sig')
    archive = ROOT/'result/旧C预测期限比较_观察3秒.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in [*files, OUT/'文件清单.csv']: z.write(p, arcname=p.name)
    print(report[:3200], flush=True)


if __name__ == '__main__': main()
