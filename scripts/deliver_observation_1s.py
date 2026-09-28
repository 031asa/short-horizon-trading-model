"""Static plots and a plain-language report for the observation-window study."""
from scripts.run_observation_1s import ROOT, OUT, OLD, FEATURES, PCOLS, sha, save_json
import json, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'svg.fonttype': 'path', 'font.size': 11})


def savefig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=200, facecolor='white')
    plt.close(fig)


def main():
    summary = pd.read_csv(OUT / '观察期汇总.csv')
    daily = pd.read_csv(OUT / '逐日表现.csv')
    coverage = pd.read_csv(OUT / '覆盖率.csv')
    paired = pd.read_csv(OUT / '相对3秒配对差异.csv')
    decisions = json.loads((OUT / '窗口选择.json').read_text(encoding='utf-8'))
    audit = json.loads((OUT / '验收.json').read_text(encoding='utf-8'))
    extension = json.loads((OUT / '延长窗口判定.json').read_text(encoding='utf-8'))
    p = pd.read_parquet(OUT / '逐任务预测.parquet')
    common = summary[(summary['sample'] == 'common_windows') & summary.method.eq('model') & summary.horizon.eq(1)].sort_values('window')
    native = summary[(summary['sample'] == 'native') & summary.method.eq('model') & summary.horizon.eq(1)].sort_values('window')
    prior = summary[(summary['sample'] == 'common_windows') & summary.method.eq('training_majority') & summary.horizon.eq(1)].sort_values('window')
    final = decisions[-1]['choice']; w = common.window.to_numpy(); days = int(common.days.iloc[0]); n = int(common.n.iloc[0])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    ax = axes[0, 0]
    ax.fill_between(w, common.accuracy_low * 100, common.accuracy_high * 100, color='#1d7885', alpha=.14, label='95%日期块区间')
    ax.plot(w, common.accuracy * 100, '-o', color='#1d7885', lw=2.4, label=f'共同任务（每窗 {n:,} 笔）')
    ax.plot(w, native.accuracy * 100, '--s', color='#999999', lw=1.4, label='各窗全部有效任务')
    for x, y in zip(w, common.accuracy * 100):
        ax.annotate(f'{y:.2f}%', (x, y), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10)
    ax.set(title='预测未来 1 秒：平均准确率', xlabel='观察期＝回看期（秒）', ylabel='日期等权准确率（%）', xticks=w)
    ax.set_ylim(min(common.accuracy_low.min(), native.accuracy.min()) * 100 - 1.5, common.accuracy_high.max() * 100 + 2.5)
    ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=.2)
    ax = axes[0, 1]
    ax.axhline(0, color='#777777', lw=1)
    ax.errorbar(paired.window, paired.gain * 100,
                yerr=np.vstack([(paired.gain - paired.low) * 100, (paired.high - paired.gain) * 100]),
                fmt='o', color='#8c5a87', capsize=5, markersize=7)
    ax.set(title='相对 3 秒观察：同任务配对差异', xlabel='观察期（秒）', ylabel='准确率差（百分点）', xticks=w)
    ax.grid(alpha=.2)
    ax = axes[1, 0]
    ax.plot(w, common.accuracy * 100, '-o', color='#1d7885', label='五因子模型')
    ax.plot(w, prior.accuracy * 100, '--o', color='#b08545', label='训练日期选出的多数类基准')
    ax.plot(w, common.balanced_accuracy * 100, '-^', color='#657aa1', label='日均类别平衡准确率')
    ax.set(title='检查提升是否只是类别占比变化', xlabel='观察期（秒）', ylabel='百分比（%）', xticks=w)
    ax.legend(fontsize=9); ax.grid(alpha=.2)
    ax = axes[1, 1]
    for field, label, color in [('down_recall', '真实下跌召回率', '#357a8e'), ('up_recall', '真实上涨召回率', '#d68a47'),
                                ('flat_recall', '真实不变召回率', '#8b719c')]:
        ax.plot(w, common[field] * 100, '-o', label=label, color=color)
    ax.plot(w, common.flat_share * 100, '--', label='真实不变占比', color='#777777')
    ax.set(title='真实不变保留；本轮不变类召回为 0', xlabel='观察期（秒）', ylabel='百分比（%）', xticks=w, ylim=(-3, 85))
    ax.legend(fontsize=9); ax.grid(alpha=.2)
    fig.suptitle(f'观察多久，再预测未来 1 秒？\n开盘首分钟 · 固定五因子与逻辑回归 · 四折向前检验，共 {days} 日', fontsize=17)
    fig.supxlabel('任务起点统一为整秒后首条快照（偏移≤0.5秒）；无冷启动；共同样本占计划检验任务95.83%。\n每个期限从各自观察结束起算；各窗预测的是不同时间段。日期块区间未校正历史选因子与选窗。\n类别平衡准确率按日内有样本类别平均召回；本图衡量预测，不是执行胜率。两快照延迟约1秒，已覆盖整个预测区间。', fontsize=9)
    savefig(fig, '观察期预测表现曲线')

    hist = daily[(daily['sample'] == 'common_windows') & daily.method.eq('model') & daily.horizon.eq(1)]
    table = hist.pivot(index='trade_date', columns='window', values='accuracy').sort_index()
    counts = hist.groupby('trade_date').n.first()
    val = pd.DataFrame([dict(fold=d['fold'], **{str(o['window']): o['accuracy'] for o in d['options']}) for d in decisions]).set_index('fold')
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), gridspec_kw={'width_ratios': [1.3, 1]}, layout='constrained')
    ax = axes[0]
    image = ax.imshow(table.to_numpy() * 100, cmap='YlGnBu', vmin=45, vmax=80, aspect='auto')
    ax.set_xticks(range(len(w)), [f'{i}秒' for i in w]); ax.set_yticks(range(len(table)), [f'{d[5:]}  n={counts.loc[d]}' for d in table.index])
    ax.set(title='后续检验日期：逐日准确率', xlabel='观察期（仅展示，未用这些格子选择当折参数）')
    for i in range(len(table)):
        for j in range(len(w)):
            value = table.iloc[i, j] * 100
            ax.text(j, i, f'{value:.1f}', ha='center', va='center', color='white' if value >= 66 else '#203641', fontsize=10)
    ax = axes[1]
    ax.imshow(val.to_numpy(float) * 100, cmap='YlGnBu', vmin=45, vmax=80, aspect='auto')
    labels = []
    for i, d in enumerate(decisions):
        tag = f'第{d["fold"]}折' if d['fold'] != 'final' else '下一批候选'
        labels.append(f'{tag}\n验证 {d["train_dates"][-5][5:]}—{d["train_dates"][-1][5:]}')
        j = list(w).index(d['choice']['window'])
        ax.add_patch(Rectangle((j - .48, i - .48), .96, .96, fill=False, edgecolor='#db6c31', linewidth=3))
        for jj in range(len(w)):
            value = val.iloc[i, jj] * 100
            ax.text(jj, i, f'{value:.2f}', ha='center', va='center', color='white' if value >= 66 else '#203641', fontsize=10)
    ax.set_yticks(range(len(val)), labels); ax.set_xticks(range(len(w)), [f'{i}秒' for i in w])
    ax.set(title='此前内部验证：用于选窗', xlabel='橙框：距最高≤0.5百分点后优先短窗')
    fig.colorbar(image, ax=axes, shrink=.6, label='日期等权准确率（%）')
    fig.suptitle(f'历史均值最高为3秒；最近验证规则选出{final["window"]}秒\n各折选窗依次为 ' + ' / '.join(str(d['choice']['window']) + '秒' for d in decisions[:-1]), fontsize=16)
    fig.supxlabel('左右两图用途不同：左图检验此前选出的参数，右图选择当折参数。最近验证与部分历史检验日期重合，不能当作额外独立验证。\n38个日期此前均参与研究；此结果支持保留2秒与3秒候选，尚未证明唯一稳定的最优观察期。', fontsize=10)
    savefig(fig, '逐日表现与选窗热力图')

    # Record the clock/coverage anomaly without rewriting any previous result.
    old = pd.read_parquet(OLD / '固定三因子原子表.parquet', columns=['window', *FEATURES, 'cum_1'])
    anomaly = []
    for window in w:
        oldw = old[old.window.eq(window)]
        newn = int(coverage.loc[coverage.window.eq(window), 'all_dates_valid'].iloc[0])
        anomaly.append(dict(window=int(window), nominal_complete=len(oldw.dropna(subset=FEATURES + ['cum_1'])),
                            actual_start_complete=newn, scheduled=len(oldw)))
    pd.DataFrame(anomaly).to_csv(OUT / '旧名义时钟覆盖审计.csv', index=False, encoding='utf-8-sig')
    rows = ['|观察＝回看|共同任务准确率|各窗全部有效准确率|相对3秒（百分点）|95%配对区间|全部有效／共同任务|',
            '|---|---:|---:|---:|---|---:|']
    for _, r in common.iterrows():
        q = paired[paired.window.eq(r.window)].iloc[0]; nat = native[native.window.eq(r.window)].iloc[0]
        rows.append(f'|{int(r.window)}秒|{r.accuracy:.2%}|{nat.accuracy:.2%}|{q.gain*100:+.3f}|[{q.low*100:+.3f}, {q.high*100:+.3f}]|{int(nat.n)}／{int(r.n)}|')
    report = f'''# 观察期选择：固定五因子，预测未来1秒

结论：18个后续检验日的共同任务上，3秒观察的历史平均准确率最高；按预先固定规则，最近5日内部验证为下一批数据选出2秒。窗口选择并非完全稳定，保留2秒与3秒两个候选；不能把历史最高点称为已验证的唯一最优参数。

## 数据与方法

- 使用原38个有效开盘日期，任务位于09:30第0—59秒，每秒一次。训练和检验均限开盘首分钟任务，不是首小时执行样本。07-20、07-27首分钟整段缺失继续排除。
- 无冷启动，观察＝回看，仅使用本任务区间的数据。实际起点采用整秒后首条快照，偏移≤0.5秒，与后续执行实验一致；信号结束时点与未来标签随窗口移动。
- 固定特征：最新OFI、报价位移、末价盘口位置、OFI方向、涨跌次数失衡。没有重新筛因子或改变公式；80%跨度覆盖、缺口/陈旧报价检查和零分母缺失规则保留。
- 标签为sign(LastPrice(T+W+1)-LastPrice(T+W))。未来端点取目标时刻起首条快照，允许误差≤0.5秒，保留不变。1秒模型同一信号在2／3秒的表现只作诊断，不分别重训。
- 四折按日期向前：训练20／25／30／35日，随后检验5／5／5／3日。训练末5日作内部验证；每窗C=0.1／1／10按验证Log loss选择，标准化仅用训练数据。
- 比较窗口时，训练、内部验证和主检验均按共同任务键对齐，只要求五因子及1秒标签有效，不借2／3秒辅助标签筛掉主样本。各窗全部有效检验任务另表展示。
- 选窗规则：验证准确率距最高不超过0.5个百分点时，先短窗，再日准确率标准差、Log loss和更强正则。每折先选窗再看后续日期；真实不变和所有可计算任务均保留，无置信度筛选。
- 5秒边界延长规则在拟合前登记，只看四折内部验证；本轮未触发，未用检验结果决定扩大搜索。
- 主表先日内求准确率，再日期等权。95%区间使用5日循环日期块5000次；属于既有日期探索，未校正历史因子及窗口搜索。

## 主结果

{chr(10).join(rows)}

各窗主表均为{n}个共同任务／18日，占1080计划任务的{n/1080:.2%}；实际起点不同于最早名义整秒实验，不与旧表直接混比。

内部验证选窗依次：{[d['choice']['window'] for d in decisions[:-1]]}秒。按这些事前选择，在后续日期的日期等权准确率为{audit['adaptive_accuracy']:.2%}，区间[{audit['adaptive_ci'][0]:.2%}, {audit['adaptive_ci'][1]:.2%}]。

最后使用前33日拟合、最近5日内部验证时，选中{final['window']}秒（验证准确率{final['accuracy']:.2%}）；其模型随后用38日重拟合，保存于下一步候选配置.json。这个最终配置没有额外独立检验，未替换任何执行模型。

## 必须同时说明的发现

1. **旧时钟的短窗覆盖问题。** 旧原子表1／2秒固定五因子完整样本分别为668／690，共2280任务；实际起点重建后为2176／2255。80%门槛未放宽。旧实验有其他特征池和口径，不能据此说旧全部结果算错，也不能直接拿旧表训练本次固定五因子短窗模型。
2. **不变类表现弱。** 共同任务真实不变比例约8%—9%，没有被删除；五因子模型在所有共同窗口共{len(p[p.common]):,}条预测中，预测不变{int(p[p.common].signal.eq(0).sum())}次，不变召回率为0。总体准确率主要来自上涨／下跌识别。日均平衡准确率按该日有真实样本的类别平均召回，某类当天无样本时不补0。
3. **1秒期限与延迟重合。** 所有共同任务两条未来快照的延迟均约1秒，订单生效时1秒预测区间已结束。预测准确率不能直接解释为生效后预测能力或可兑现节约；本轮没有改延迟、没有跑执行策略，也没有把标签偷偷移到生效后。
4. **窗口并非唯一稳定。** 历史主曲线峰值在3秒，最近验证更支持2秒。下一轮若必须按预定流程锁定一个参数，使用2秒；3秒保留为历史表现较好的对照。3秒历史峰值属看过检验曲线后的描述，不能替代当折选择。
5. **信息量与时点同时改变。** 1秒和3秒观察预测的是不同时间段。当前实验评价实际等待方案；准确率差不能全部归因于多收集了信息。没有观察到可据此断言“长窗口必然加入噪音”的证据。

## 验收与文件

- 原子表{audit['rows']:,}行；{audit['causal_comparisons']}次真实数据截断/观察期前后扰动一致性检查；{audit['independently_checked_labels']:,}个标签独立复核。
- 起点与旧表相同的{audit['same_clock_old_rows_exact']:,}行，五因子与1秒标签逐值相同。训练/检验日期隔离，存储参数可复算预测；原行情、原模型和撮合文件SHA256未变。
- 观察期汇总.csv、逐日表现.csv、逐折表现.csv、相对3秒配对差异.csv、覆盖率.csv、缺失原因.csv、三分类混淆矩阵.csv、执行延迟诊断.csv均保留。
- 原子表、逐任务预测、模型参数、内部验证记录、窗口选择、下一步候选配置、实验口径和验收JSON同目录保存。
- 图1：观察期预测表现曲线；图2：逐日表现与选窗热力图。均提供PNG／SVG，无新看板。
'''
    (OUT / '观察期选择报告.md').write_text(report, encoding='utf-8')
    metadata = dict(images=2, primary_days=days, primary_tasks=n, windows=w.tolist(),
                    historical_peak_window=int(common.loc[common.accuracy.idxmax(), 'window']),
                    prospective_validation_choice=int(final['window']),
                    extension=extension, same_signal_other_horizons=True,
                    not_an_execution_backtest=True)
    save_json(OUT / '交付验收.json', metadata)
    files = sorted(p for p in OUT.iterdir() if p.is_file())
    manifest = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
    archive_path = ROOT / 'result/观察期选择_未来1秒.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, p.name)
        z.writestr('文件清单.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(archive_path) as z:
        import hashlib
        for row in manifest:
            assert hashlib.sha256(z.read(row['path'])).hexdigest() == row['sha256']
    print(report, flush=True)


if __name__ == '__main__':
    main()
