"""Package aggregate arrays in a self-contained offline HTML and write research reports."""
from pathlib import Path
from itertools import product
import base64
import gzip
import json
import numpy as np
import pandas as pd
from utils.factor_catalog import FAMILIES

METRICS=['mean_ic','mean_rank_ic','median_ic','median_rank_ic','std_ic','std_rank_ic','valid_ic_days','valid_days',
         'mean_pairs','total_pairs','mean_scheduled_coverage','positive_ic_days','negative_ic_days',
         'positive_rank_ic_days','negative_rank_ic_days','days_not_enough_pairs','days_constant_factor','days_constant_label','mean_zero_return_share']

def dashboard_payload(summary,registry,coverage,metadata):
    names=registry.loc[registry.evaluate,'factor'].tolist()
    observations=metadata['config']['schedule']['observation_seconds']
    horizons=metadata['config']['horizons_seconds']
    combos=list(product(observations,('decision','arrival'),('cumulative','incremental'),('signed','absolute'),('own','common_observations_horizons')))
    keys=['observation_seconds','anchor','label_type','target','pair_set','factor','horizon_seconds']
    index=pd.MultiIndex.from_tuples([(*c,f,h) for c in combos for f in names for h in horizons],names=keys)
    ordered=summary.set_index(keys).reindex(index)
    assert len(ordered)==len(summary) and ordered.phase.eq('all_sample').all()
    arrays=ordered[METRICS].to_numpy(dtype='<f4')
    finite=np.isfinite(arrays)
    original=ordered[METRICS].to_numpy(float)
    np.testing.assert_allclose(arrays[finite],original[finite],atol=1e-6,rtol=1e-6)
    registry_records=coverage.fillna('').to_dict('records')
    for r in registry_records:r['status_counts']=json.loads(r['status_counts'])
    info=dict(families=FAMILIES,registry=registry_records,factors=names,metrics=METRICS,combos=combos,horizons=horizons,
        meta={k:metadata[k] for k in ('atomic_rows','included_dates','excluded_openings','opening_gaps','computed_outputs','alias_outputs','quality_outputs','registry_outputs','prior_version')})
    packed=gzip.compress(arrays.tobytes(order='C'),compresslevel=9,mtime=0)
    # Verify the exact embedded payload, including every row/metric, before rendering.
    np.testing.assert_array_equal(np.frombuffer(gzip.decompress(packed),dtype='<f4').reshape(arrays.shape),arrays)
    metadata['verification']['dashboard_aggregate_values_checked']=int(arrays.size)
    metadata['verification']['dashboard_float32_tolerance']='atol 1e-6, rtol 1e-6; full precision in data files'
    return json.dumps(info,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'),base64.b64encode(packed).decode()

def render_report(summary,registry,coverage,metadata,output):
    output=Path(output)
    from scripts.filter_positive_ic import filter_positive
    filter_positive(output)
    info,payload=dashboard_payload(summary,registry,coverage,metadata)
    template=(Path(__file__).resolve().parents[1]/'utils/ic_dashboard.html').read_text(encoding='utf-8')
    (output/'全部因子IC与衰减.html').write_text(template.replace('__INFO__',info).replace('__PAYLOAD__',payload),encoding='utf-8')
    missing=coverage.loc[coverage.evaluate & coverage.valid_rows.eq(0)]
    base=summary.loc[summary.observation_seconds.eq(1)&summary.anchor.eq('decision')&summary.label_type.eq('cumulative')&summary.target.eq('signed')&summary.pair_set.eq('own')]
    noic=base.groupby('factor').valid_days.max();noic=noic[noic.eq(0)].index.tolist()

    lines=['# 全量因子 IC 与 30 秒衰减分析','',
        f"原子表：**{len(metadata['included_dates'])} 日 × 50 个任务 × 5 个观察期 = {metadata['atomic_rows']:,} 行**。59 个家族展开为 {len(registry)} 个输出：{metadata['computed_outputs']} 个计算输出、{metadata['alias_outputs']} 个别名、{metadata['quality_outputs']} 个质量字段。",'',
        '直接打开 [全部因子 IC 与衰减网页](全部因子IC与衰减.html)。所有家族按编号展示；别名复用其目标的 IC，质量字段进入质量清单，不独立计算方向 IC。','',
        '## 数据与样本','',
        '- 全部 38 个有效上午开盘纳入，无开发／验证划分。phase 统一为 all_sample；逐日计算 Pearson / Spearman，再对有效日期等权汇总。没有把全部快照合并成一个相关系数。',
        '- 第 10–59 秒每秒产生任务，观察 1／2／3／4／5 秒。特征回看包含任务前历史；普通窗口 5／10 秒。相同观察结束时刻的特征完全相同。',
        '- decision：观察结束时刻及其不晚于该时刻的最新有效末价。arrival：观察结束后第二张快照，保留原执行延迟口径。两种标签各自以其起点 LastPrice 为基准；没有模拟执行收益。',
        '- 未来 1–30 秒逐秒取目标时刻不早于目标的首张快照，对齐误差 ≤0.5 秒；价格换算为 0.2 tick。未来标签允许越过开盘第一分钟。',
        '- 逐期限样本：因子与本期限标签均有效。共同任务样本：对单一因子、单一时间基准和响应类型，要求同一任务在全部 5 个观察期及全部 30 个期限有效。各因子独立求交集；不要求所有因子共同有效。',
        '- 每日最少 20 对；缺失、常数因子、常数标签均不计算相关系数。有效零价变保留。网页覆盖率为配对数 / 该观察期全部已排定任务数，逐日等权。',
        '- 网页的 IC 均值、中位数和标准差均针对逐日 IC 分布。标准差 ddof=1；正负日期比例以相应方法的有效日期为分母，零 IC 日期保留在分母。配对数与覆盖率包括因子常数等导致 IC 不可计算的日期。','',
        '## 累计、增量与逻辑预期','',
        '- 累计响应：P(t+u)-P(t)。逐秒新增响应：P(t+u)-P(t+u-1)。绝对价变对所选响应整体取绝对值，不能把累计绝对值相减作为新增响应。',
        '- 累计曲线保留前期已发生的价差；新增曲线观察后续新增价变关系。共同样本便于区分覆盖变化与期限变化。不强拟合统一半衰期，不事后挑最佳期限。',
        '- 增量两端任一累计标签不合法则增量缺失；从起点到终点仍检查连续性，缺口不会被端点相减掩盖。',
        '- 逻辑预期在新增 IC 前登记于 factor_hypotheses.csv。此前已看过部分期货结果，因此标为研究假设；后续 ETF 研究前可冻结为其事前假设。后续补充的质量诊断字段仅登记在 feature_registry.csv，全部为不适用；已计算因子的预期与原始登记逐项一致。',
        '- 网页每个期限显示原始正负 IC；未按结果改写逻辑预期，未自动翻负号。统计正负不等于统计显著，本轮不输出显著性星号。',
        '- 波动／活动输出同时计算绝对价变，其他计算输出也保留绝对价变对照。重叠任务及重复决策时刻不视为独立实验；IC 不是准确率、概率或交易收益。','',
        '## 固定参数与因果约束','',
        '- 普通历史 5／10 秒；前后两段各 2／5 秒；M04/M05 历史差分 2 秒；M06 为 5 对 10 秒；M07–M09 外层 5 秒，内层 5／10 秒；QI 中性带 0.2，均线中性带 0.5 tick。',
        '- B09：回看 10 秒、事件观察 2 秒、净减少 10%；同侧事件观察区间不重叠；同价成熟事件至少 3 个。成功耗时仅用成功事件，至少 1 个。未成熟、报价退出、未恢复、缺口事件分别保留支持数。',
        '- D09：同期 5／10 秒；滞后版历史 10 秒、滞后 2 秒、至少 8 对。历史斜率至少 3 个正权重且不同的时间点。',
        '- 历史派生状态使用历史当时的均线与统计，未用最终均线、极值或成熟事件回填过去。左删失年龄仅保留下界和标记，主 IC 的年龄用精确值。',
        '- R04：价差差值／历史分位；深度对数比／历史分位；成交速率与波动比值／历史分位，后两者分别保留 5／10 秒版本。R05 复用项明确映射，补充深度与波动变化。R06 仅质量输出。','',
        '## 全部因子覆盖清单','',
        '完整逐输出清单见 [factor_coverage.csv](factor_coverage.csv)；含定义、公式、参数、单位、预期、缺失原因、支持行数和日期数。下表覆盖全部 59 家族。','',
        '| 家族 | 定义 | 计算 | 别名 | 质量 | 至少一行有效／总输出 |','|---|---|---:|---:|---:|---:|']
    for row in pd.read_csv(output/'family_coverage.csv').itertuples(index=False):
        lines.append(f'| {row.family_id} | {row.family} | {row.computed} | {row.aliases} | {row.quality} | {row.available_outputs}/{row.outputs} |')
    lines+=['','## 支持不足与异常','',f'全程无有效因子值的计算输出：{len(missing)} 个；在观察 1 秒、decision、有方向累计响应、逐期限样本的全部期限均无有效日 IC 的输出：{len(noic)} 个。网页保留所有这些输出。',
        '完整原因见 [数据异常说明.md](数据异常说明.md)；未根据覆盖不足放宽 B09 或每日最小配对数。','',
        '## 原有口径复核与文件','',
        f"原有 30 列及原 1–5 秒标签完成 {metadata['verification'].get('legacy_feature_and_label_values_checked',0):,} 个数值比较；原逐期限样本累计 IC 完成 {metadata['verification'].get('legacy_own_ic_rows_checked',0):,} 行复核，容差 1e-10。共同样本现在交集到 30 秒，样本改变，因此不要求与旧的 1–5 秒共同样本 IC 相同。",
        '- 根目录 sample_snapshot_原子执行总表.parquet：任务、全部输出、每项状态、累计／增量标签及标签状态。',
        '- daily_ic.parquet：逐日完整明细；summary_ic.csv / summary_ic.parquet：全样本汇总。明细因规模改用 Parquet，未裁剪日期或因子。',
        '- feature_registry.csv / factor_hypotheses.csv：登记与计算前假设；factor_coverage.csv / family_coverage.csv：全部输出及 59 家族覆盖。',
        '- experiment_manifest.json：参数、原始数据／源码／产物指纹、发布时间状态与验收记录。',
        '- 网页只嵌入汇总和因子覆盖，无外部资源或服务。浏览器内数值以 float32 存储；完整数据保持原精度。每个嵌入汇总值已验证误差 ≤1e-6 + 1e-6×绝对值。',
        '- 原始行情与 PDF 保持不变；PDF 为公式来源，当前批准的实验计划优先于文档中的旧时间范围。ETF 五档导入与真正筛选留待后续。','']
    (output/'IC研究结果.md').write_text('\n'.join(lines),encoding='utf-8')
    anomaly=['# 数据异常说明','', '## 行情缺失','', '整段无开盘行情并排除：'+ '、'.join(x['trade_date'] for x in metadata['excluded_openings'])+'。','']
    for g in metadata['opening_gaps']:anomaly.append(f"- {g['start']} → {g['end']}，间隔 {g['seconds']:g} 秒。检查范围延伸至开盘后 96 秒。")
    anomaly+=['','部分缺口保留该日；相应历史连续片段重置，跨缺口未来标签缺失，当前源过旧不填零。', '当前源状态：'+json.dumps(metadata['feature_status_counts'],ensure_ascii=False),'',
        '## 支持不足（不是静默删除）','', '| 输出 | 有效行 | 有效日期 | 主要缺失状态 |','|---|---:|---:|---|']
    for row in coverage.loc[coverage.evaluate & (coverage.coverage.lt(.8)|coverage.factor.isin(noic))].itertuples(index=False):
        anomaly.append(f'| {row.factor} | {row.valid_rows} | {row.valid_dates} | {row.status_counts} |')
    anomaly+=['','B09 事件频度和同价要求可能导致每日少于 20 对；即使有少量因子值，也可能没有任何有效日 IC。未恢复计入恢复比例与成功率，不混入成功耗时。未成熟及报价退出另行记录。',
        '年龄无可观察起点时标记 LEFT_CENSORED，不能将年龄下界当精确年龄输入主 IC。常数值也不等于缺失。',
        '','## 本轮范围说明','',
        '没有开发／验证划分；共同样本因交集范围扩大至 30 个期限而可能明显缩减。网页逐因子、逐期限显示实际有效日期与覆盖。',
        'LocalTime 的业务含义尚未确认，本轮未把它解释为网络延迟；沿用 Datetime 与第二张后续快照的延迟口径。','']
    (output/'数据异常说明.md').write_text('\n'.join(anomaly),encoding='utf-8')
    print(f'Offline HTML written: {(output/"全部因子IC与衰减.html").stat().st_size/1e6:.1f} MB',flush=True)
