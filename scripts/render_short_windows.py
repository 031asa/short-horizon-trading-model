"""Render full curves, paired differences and snapshot counts; no horizon-average scores."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'result/opening_execution'

def render():
    raw=(OUT/'短观察期_看板数据.json').read_text(encoding='utf-8');d=json.loads(raw)
    template=(ROOT/'utils/short_window_dashboard.html').read_text(encoding='utf-8')
    (OUT/'短观察期因子初筛.html').write_text(template.replace('__DATA__',raw.replace('<',chr(92)+'u003c')),encoding='utf-8')
    selected=[s for s in d['selections'] if s['mode']=='wait'];n=sum(s['candidate'] for s in selected)
    fmt=lambda v:'—' if v is None else f'{v:+.3f}'
    lines=['# 短窗口因子逐期限比较','',
        '[打开完整曲线与快照数量看板](短观察期因子初筛.html)','',
        '主实验严格采用观察期＝回看窗口＝1／2／3／4／5 秒。任务时点 T，观察结束 t=T+W；所有未来 1–30 秒分别从本组观察结束起算，评价 LastPrice(t+u)−LastPrice(t)。显示各期限原始 IC，不再用跨期限平均分筛选。执行延迟单独查看，不参与主筛选。','',
        '附加诊断把所有回看固定在任务后第 5 秒结束观察，未来标签相同，只改变回看长度；用于隔离历史信息量，不冒充更早出信号。没有用尚未形成的长窗口因子评价此前价格。','',
        '## 当前结论','',
        f'28 种明确预期的可比表达中，{n} 种呈现早段关系或首秒脉冲；其中包含相关性很高的重复信息表达，并非 {n} 份独立信号。4 种特殊参数表达另列。',
        '滚动 OFI 类和双侧队列变薄差是“短窗有效、长窗较弱”；成交位置偏离是“长短窗均有关系，压缩证据不足”；最新 OFI、报价整体移动和成交盘口位置属于少量输入的快照基准。',
        '按本轮预先固定的前三个期限、允许 IC 损失 0.02 及配对区间条件，暂时没有滚动表达通过“短窗接近有效长窗”。这不等于短窗无效，而是尚不能排除损失超过门槛，也不能把长窗本身无效称为压缩成功。','',
        '| 表达 | 窗口 | 输入快照中位数 | 未来 1s IC | 未来 2s IC | 未来 3s IC | 未来 4s IC | 未来 5s IC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for e in ['F03_OFI','A06_PositionDeviation','B05_DepletionDifference','F03_OFI_latest','F07_QuotePosition']:
        for w in [1,5]:
            c=next(r for r in d['counts'] if r['mode']=='wait' and r['expression']==e and r['w']==w)
            curves=[r for r in d['curves'] if r['mode']=='wait' and r['expression']==e and r['w']==w and r['anchor']=='decision' and r['pair_set']=='own' and r['label_type']=='cumulative' and r['horizon']<=5]
            lines.append('| '+e+f" | {w}s | {c['median_used']:g} | "+' | '.join(fmt(r['mean_rank_ic']) for r in curves)+' |')
    lines+=['','以上是逐日 Spearman IC 的等权均值，每个格子对应一个明确未来期限。看板同时提供 1–30 秒全部格子、五条曲线和逐秒新增曲线；持续正累计 IC 不代表后面每秒都有新增信息。','',
        '## 快照成本','',
        '普通滚动窗口 1／2／3／4／5 秒的有效输入快照中位数为 3／5／7／9／11 条，包含历史边界；观察期间新收到的中位数为 2／4／6／8／10 条。最新 OFI、QI 变化和报价变化需要相邻两条；QI、盘口位置及当前 QI 状态只需一条。',
        '数量从原质量表和原始行情时间戳得到，不用固定频率推算，不依赖未来标签有效性。滚动输入区间可能稍早于任务开始；连续采集与冷启动不变。快照因子的观察期不会改变所需输入条数，不能将五条快照曲线视为窗口压缩证据。',
        '主实验 1 秒与 5 秒名义信号终点有 46／50 重叠；同一结束时点诊断中，快照基准五条线完全重合是输入相同导致的。','',
        '## 逐期限筛选和配对比较','',
        '先登记本轮规则再运行，但此前已经研究过相同的期货数据，因此结果仍属研究性判断。逐项检查未来第 1、2、3 秒：预期一致 Rank IC ≥0.05、有效日 ≥30、覆盖 ≥80%、符合预期日期 ≥60%、Pearson 同向。三期都通过称早段关系；只有第一期达到门槛则单列首秒脉冲。全 30 个期限各自保留，不用峰值概括整个因子。',
        '曲线可查看逐期限样本或五个 W × 30 期限共同任务。差异图另取所选 W 与 5 秒共同有效的任务及配对日期，既不直接相减不同样本的汇总，也不因 1 秒斜率缺失而禁止 2 秒与 5 秒比较。主实验的两组未来区间分别相对各自结束时点；诊断实验未来区间完全相同。',
        '相对 5 秒的差异按逻辑预期方向计算；所有短、长原始 IC 单独保留。按日期做 5 日循环块重采样 2000 次、固定种子，得到各期限描述性区间。只有短、长在前三期均有效，且三期差值区间下界都 ≥−0.02 才标记压缩支持。没有跨因子／窗口多重检验校正，不宣称严格等价或统计显著，也不据此估计交易收益。','',
        '## 不足与异常','',
        '- A04 失衡斜率和 B03 深度斜率差在 1 秒缺失：窗口通常 3 条快照，仅 2 个正权重历史时间点，达不到原定至少 3 个时间点。没有降低门槛。',
        '- A08 双段 OFI 变化总历史为 4／10 秒；B09 三种恢复差类固定 10 秒，仍保留在全量看板。',
        '- 两天整段缺行情继续排除；7 月 21 日的 10.5 秒缺口继续按既有规则置缺失。共同样本或配对区间不足均显示空值。',
        '- F07 等信号在执行延迟后可能反号；本轮按用户要求只作为附加敏感性视图，没有用其剔除主预测候选。','',
        '## 验证','',
        f"本轮 {d['audit']['own_cells_verified']:,} 个原口径逐期限汇总与全量实验逐项核对，包含 Pearson、Spearman、有效日和覆盖。所有配对比较的短、长两侧任务数一致。原始行情、原子表和原全量 IC 文件未改写。",
        '数据：短观察期_五组对比.csv（现为逐期限曲线，不是旧平均分）、短观察期_候选清单.csv、短观察期_相对5秒逐期限差异.csv、短观察期_快照数量汇总.csv、短观察期_快照数量逐任务.parquet、短观察期_逐日IC.parquet。','']
    (OUT/'短观察期因子初筛.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':render()
