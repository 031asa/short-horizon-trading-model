"""Freeze a research review, evaluate only new factors, and stage offline outputs.

build -> validation scripts / browser QA -> publish. Original 275-factor results
are archived with fingerprints and copied without recalculation or sign changes.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from utils.opening_ic import ICConfig,feature_registry
from utils.directional_review import REVIEW_VERSION,REVIEW_STATUS,reviewed_registry,enrich_atomic,variants
from utils.ic_statistics import evaluate_all
from scripts.run_opening_ic import sha,feature_audit,evaluation_fingerprint,verify_previous_features,verify_previous_ic
from scripts.render_opening_ic import render_report

OUT=ROOT/'result/opening_execution'
BASE=ROOT/'.cache/directional_review_base'
WORK=OUT/'.review_pending'
ATOMIC=ROOT/'sample_snapshot_原子执行总表.parquet'
FILES=['atomic.parquet','feature_registry.csv','summary_ic.parquet','daily_ic.parquet',
       'factor_hypotheses.csv','experiment_manifest.json','evaluation_manifest.json']


def write_atomic(atomic,registry,review):
    schema=pq.read_schema(BASE/'atomic.parquet')
    embedded=json.loads(schema.metadata[b'atomic_ic'])
    embedded.update(prior_version=REVIEW_VERSION,directional_review=review,
        registry_outputs=len(registry),computed_outputs=int(registry.evaluate.sum()),
        feature_columns=registry.loc[registry.evaluate,'factor'].tolist())
    table=pa.Table.from_pandas(atomic,preserve_index=False)
    table=table.replace_schema_metadata({**(table.schema.metadata or {}),b'atomic_ic':json.dumps(embedded,ensure_ascii=False).encode()})
    pq.write_table(table,WORK/'atomic.parquet',compression='zstd')


def archive_base():
    BASE.mkdir(parents=True,exist_ok=True)
    manifest=BASE/'snapshot.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text(encoding='utf-8'))
        for name,digest in saved['sha256'].items():assert sha(BASE/name)==digest
        return saved
    # Archive exactly the original experiment, not a previously enriched run.
    r=pd.read_csv(OUT/'feature_registry.csv')
    assert 'review_origin' not in r and r.evaluate.sum()==275
    for name in FILES:shutil.copy2(ATOMIC if name=='atomic.parquet' else OUT/name,BASE/name)
    saved=dict(archived_at=datetime.now(timezone.utc).isoformat(),sha256={n:sha(BASE/n) for n in FILES})
    manifest.write_text(json.dumps(saved,indent=2),encoding='utf-8')
    return saved


def write_review_report(summary,registry,coverage,meta):
    names={'positive':'正向','negative':'负向','uncertain':'不确定','not_applicable':'不适用'}
    original=registry.loc[registry.evaluate & registry.review_origin.eq('original')]
    changed=original.loc[original.review_change.eq('new_hypothesis')]
    remain=original.loc[original.expected_sign_signed.eq('uncertain')]
    new=registry.loc[registry.review_origin.eq('derived')]
    export=registry.loc[registry.kind.ne('quality')].copy()
    export.to_csv(WORK/'逻辑预期逐项审阅.csv',index=False,encoding='utf-8-sig')
    new.to_csv(WORK/'新增方向因子定义.csv',index=False,encoding='utf-8-sig')
    base=summary.loc[summary.observation_seconds.eq(1)&summary.anchor.eq('decision')&
        summary.label_type.eq('cumulative')&summary.target.eq('signed')&summary.pair_set.eq('own')]
    new_summary=summary.loc[summary.factor.isin(new.factor)]
    new_summary.to_parquet(WORK/'新增方向因子完整IC.parquet',compression='zstd',index=False)
    counts=original.expected_sign_signed.value_counts().to_dict()
    lines=['# 有方向价变：机制审阅与新增因子实验','',
        f'原有 275 个计算输出中，{len(changed)} 个从不确定升级为明确的正／负研究假设；原来正向 47、负向 7、不确定 221，现在正向 {counts.get("positive",0)}、负向 {counts.get("negative",0)}、不确定 {len(remain)}。另新增 {len(new)} 个带方向的交互输出。质量字段与别名不计入这些数量。','',
        '这里的“明确”是明确要检验哪个方向，不是确认未来一定同向，也不是统计显著。该批期货全部数据和部分 IC 已经看过，本次属于用户批准的事后机制研究。未将新假设称为期货事前假设；后续 ETF 前仍须重新冻结。','',
        '看板默认使用“激进研究假设”，可切回“原始登记”；原始视图只显示原 PDF 输出。新增表达有独立英文名和“新增”标记，旧定义、旧值、旧 IC 均保留。逻辑预期不会改变原始 IC 的颜色。','',
        '## 如何指定方向','',
        '- 单个 LastPrice 最新跳动优先提出回摆负向假设；5／10 秒净位移、净方向效率和趋势斜率采用延续正向假设。单跳与持续路径采用不同机制，需在各期限检验。',
        '- 两侧报价上移、买侧承接、上行分量采用正向；报价下移、卖侧供给、下行分量采用负向。单侧量的假设需要另一侧活动相近，强度型原角色保留，避免把条件关系当恒等式。',
        '- 均线偏离、成交加权均价偏离和趋势残差采用回归负向假设；均线上下持续状态、完整穿越频率采用延续假设。两者是竞争机制，不能据此保证同一个市场状态同时满足两个预测。',
        '- 总波动、总成交量、总深度、无方向年龄等仍有对称性：同样的数值可以出现在上涨或下跌市场。选其中可解释的表达，加入已知 QI、OFI 或价格状态方向，再作为新因子测。未把原值强行改成有方向。','',
        '## 新表达的统一规则','',
        '- 原历史窗口、观察期、标签、任务数、每日最少 20 对、缺口规则均不变。新增因子只读取当前原子行的父因子，绝不读取未来标签；有效性取父因子交集。',
        '- 对数统一使用 log1p，年龄尺度 1 秒、活动量尺度 1（按登记单位），均在重测前固定；未用本批 IC 选阈值。',
        '- 同向／反向偏离用 MADeviation × OFIDirection 的正负划分，乘积为零时均记零；只对父因子全有效的行记零。无行情、年龄左删失或父因子缺失仍为缺失。',
        '- 门控表达的主 IC 包含有效但未激活的零值，是全任务评分 IC，不是只在触发子样本计算的条件 IC。',
        '- 新交互均与父因子共享信息。更高 IC 不等于独立增量信息；本轮不宣称已完成控制父因子的增量检验或筛选。所有正负结果完整保留。','',
        '## 修改清单（原值不变）','',
        '逐项条件、相反机制与原始预期见 [逻辑预期逐项审阅.csv](逻辑预期逐项审阅.csv)。','',
        '| 因子 | 原预期 → 本次假设 | 主要理由 |','|---|---|---|']
    for r in changed.itertuples():lines.append(f'| {r.factor} | 不确定 → {names[r.expected_sign_signed]} | {r.logical_reason} |')
    lines+=['','## 新增因子的全部默认口径结果','',
        '以下固定展示观察 1 秒、观察结束起算、有方向累计响应、逐期限样本的 Spearman 日 IC 等权均值。只选固定展示秒数，不据结果选最佳期限；网页和数据包含两种方法、全部观察期与 1–30 秒完整曲线。','',
        '| 新因子 | 假设 | 有效因子行 | 1s | 5s | 10s | 20s | 30s |','|---|---|---:|---:|---:|---:|---:|---:|']
    rows=[]
    for r in new.itertuples():
        c=base.loc[base.factor.eq(r.factor)].set_index('horizon_seconds')
        n=int(coverage.set_index('factor').loc[r.factor,'valid_rows'])
        vals=[c.loc[u,'mean_rank_ic'] for u in (1,5,10,20,30)]
        cells=['—' if not np.isfinite(v) else f'{v:+.4f}' for v in vals]
        lines.append(f'| {r.factor} | {names[r.expected_sign_signed]} | {n} | '+ ' | '.join(cells)+' |')
        rows.append(dict(factor=r.factor,expected_sign_signed=r.expected_sign_signed,
            valid_rows=n,green_horizons=int(c.mean_rank_ic.gt(0).sum()),red_horizons=int(c.mean_rank_ic.lt(0).sum()),
            valid_horizons=int(c.mean_rank_ic.notna().sum()),max_valid_days=int(c.valid_days.max()),
            min_valid_days=int(c.valid_days.min())))
    result=pd.DataFrame(rows)
    result.to_csv(WORK/'新增方向因子默认结果概览.csv',index=False,encoding='utf-8-sig')
    opposite_positive=int((result.expected_sign_signed.eq('positive')&result.red_horizons.eq(30)).sum())
    opposite_negative=int((result.expected_sign_signed.eq('negative')&result.green_horizons.gt(15)).sum())
    lines.insert(8,f'本次重测的重要结果：在下述固定默认口径下，{opposite_positive} 个预期正向的新增表达在全部 30 个期限均为负 IC；{opposite_negative} 个预期负向的新增表达在多数期限为正 IC。方向化并未因此证明预测改善。这些反例完整保留，未按结果反改假设或翻号。')
    lines+=['','## 支持不足与保留的不确定','',
        f'原有 {len(remain)} 个计算输出继续保留不确定，均有逐项原因及下一步改造方向。B09 的假设可以明确，但原来的成熟门槛与有效日数量不因此增加；不为得到数字放宽门槛。',
        '年龄派生取精确年龄，可能减少覆盖；由父因子联合有效性导致的减少见网页详情。部分行情缺口仍是 2026-07-21 的 10.5 秒，未新增删日。','',
        '## 机制参考及适用边界','',
        '- [Cont、Kukanov、Stoikov：The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402)：支持最佳档供需失衡与价格冲击、深度调节作用的机制。其同步股票价格冲击结果不等于本批期货未来 LastPrice IC 的符号证明。',
        '- [Gould、Bonart：Queue Imbalance as a One-Tick-Ahead Price Predictor](https://arxiv.org/abs/1512.03492)：研究队列失衡与股票下一次中间价变动。我们仅参考方向机制，评价标签依然全部使用 LastPrice。',
        '- [NBER：Reversals and the Returns to Liquidity Provision](https://www.nber.org/papers/w30917)：作为暂时冲击与流动性回补的回归机制参考，不能直接外推到本实验的开盘 1–30 秒。',
        '均线、门控与活动方向化的具体表达是本项目研究假设，并非上述文献逐条验证过的现成结论。','',
        '## 可追溯记录','',
        '- factor_hypotheses.csv：原始冻结，原样保留。directional_hypotheses.csv：本次完整冻结，包含原／新预期、成立条件、替代机制与新表达父因子。',
        '- directional_review_manifest.json：冻结时间、文件指纹、旧表与旧 IC 不变检查、新计算规模及验收记录。',
        '- 新增方向因子完整IC.parquet：新增全部表达、全部评价口径；完整 daily_ic.parquet 同时保留原始与新增逐日明细。','']
    (WORK/'有方向逻辑预期审阅.md').write_text('\n'.join(lines),encoding='utf-8')


def build():
    snapshot=archive_base();WORK.mkdir(parents=True,exist_ok=True)
    cfg=ICConfig();base_registry=feature_registry(cfg);registry=reviewed_registry(base_registry)
    frozen=WORK/'directional_hypotheses.csv'
    registry.to_csv(frozen,index=False,encoding='utf-8-sig')
    review=dict(version=REVIEW_VERSION,status=REVIEW_STATUS,registered_at=datetime.now(timezone.utc).isoformat(),
        hypotheses_sha256=sha(frozen),implementation_sha256=sha(ROOT/'utils/directional_review.py'),
        base_snapshot=snapshot,original_computed=275,new_variants=len(variants()),
        changed_original=int(((registry.review_change=='new_hypothesis') & registry.evaluate).sum()))
    (WORK/'directional_review_manifest.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Hypotheses frozen before reading labels / evaluating new outputs',review['registered_at'],flush=True)
    original=pd.read_parquet(BASE/'atomic.parquet');atomic=enrich_atomic(original)
    pd.testing.assert_frame_equal(atomic[original.columns],original)
    assert len(atomic)==9500 and atomic.trade_date.nunique()==38
    write_atomic(atomic,registry,review)
    pd.testing.assert_frame_equal(atomic,pd.read_parquet(WORK/'atomic.parquet'))
    review['original_atomic_columns_unchanged']=len(original.columns)
    review['original_atomic_values_and_statuses_unchanged']=True
    metadata=json.loads((BASE/'experiment_manifest.json').read_text(encoding='utf-8'))
    metadata['verification']={k:v for k,v in metadata['verification'].items() if not k.endswith('_verification')}
    metadata.update(prior_version=REVIEW_VERSION,registry_outputs=len(registry),computed_outputs=int(registry.evaluate.sum()),
        feature_columns=registry.loc[registry.evaluate,'factor'].tolist(),published=False,
        atomic_sha256=sha(WORK/'atomic.parquet'),directional_review=review,
        hypotheses_note='Original freeze retained; explicit exploratory directional review registered separately before new variant evaluation.')
    metadata['verification'].update(verify_previous_features(atomic))
    for name in ['factor_hypotheses.csv']:
        shutil.copy2(BASE/name,WORK/name)
    for name in ['excluded_openings.csv','dashboard_qa_probes.json']:
        shutil.copy2(OUT/name,WORK/name)
    registry.to_csv(WORK/'feature_registry.csv',index=False,encoding='utf-8-sig')
    coverage=feature_audit(atomic,registry,WORK)
    new_dir=WORK/'new_evaluation';new_dir.mkdir(exist_ok=True)
    new_registry=registry.loc[registry.review_origin.eq('derived')]
    review['evaluation_started_at']=datetime.now(timezone.utc).isoformat()
    fresh,n_daily=evaluate_all(atomic,new_registry,cfg,new_dir)
    assert sha(frozen)==review['hypotheses_sha256']
    old_summary=pd.read_parquet(BASE/'summary_ic.parquet')
    summary=pd.concat([old_summary,fresh],ignore_index=True)
    summary.to_parquet(WORK/'summary_ic.parquet',compression='zstd',index=False)
    summary.to_csv(WORK/'summary_ic.csv',index=False,encoding='utf-8-sig')
    pd.testing.assert_frame_equal(pd.read_parquet(WORK/'summary_ic.parquet').iloc[:len(old_summary)],old_summary)
    # Copy Arrow batches; compare the entire original prefix against its source.
    writer=None;copied=0
    try:
        for source in [BASE/'daily_ic.parquet',new_dir/'daily_ic.parquet']:
            for batch in pq.ParquetFile(source).iter_batches(batch_size=100000):
                if writer is None:writer=pq.ParquetWriter(WORK/'daily_ic.parquet',batch.schema,compression='zstd')
                writer.write_batch(batch);copied+=len(batch)
    finally:
        if writer is not None:writer.close()
    original_batches=pq.ParquetFile(BASE/'daily_ic.parquet').iter_batches(batch_size=100000)
    final_batches=pq.ParquetFile(WORK/'daily_ic.parquet').iter_batches(batch_size=100000)
    compared=0
    for batch in original_batches:
        other=next(final_batches)
        # Last original batch may include initial new rows when rebatched.
        assert batch.equals(other.slice(0,len(batch)))
        compared+=len(batch)
    review.update(original_summary_rows_unchanged=len(old_summary),original_daily_rows_unchanged=compared,
        new_daily_rows=n_daily,new_summary_rows=len(fresh),all_daily_rows=copied,
        original_summary_sha256=snapshot['sha256']['summary_ic.parquet'],
        completed_at=datetime.now(timezone.utc).isoformat())
    metadata.update(ic_daily_rows=copied,ic_summary_rows=len(summary))
    metadata['verification'].update(verify_previous_ic(WORK))
    metadata['verification']['directional_review_original_unchanged']=review
    evaluation=dict(fingerprint=evaluation_fingerprint(atomic,registry,cfg),daily_rows=copied,
        daily_sha256=sha(WORK/'daily_ic.parquet'),summary_sha256=sha(WORK/'summary_ic.parquet'))
    (WORK/'evaluation_manifest.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2),encoding='utf-8')
    (WORK/'experiment_manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    render()


def render():
    metadata=json.loads((WORK/'experiment_manifest.json').read_text(encoding='utf-8'))
    registry=pd.read_csv(WORK/'feature_registry.csv');coverage=pd.read_csv(WORK/'factor_coverage.csv')
    summary=pd.read_parquet(WORK/'summary_ic.parquet')
    render_report(summary,registry,coverage,metadata,WORK)
    write_review_report(summary,registry,coverage,metadata)
    (WORK/'directional_review_manifest.json').write_text(json.dumps(metadata['directional_review'],ensure_ascii=False,indent=2),encoding='utf-8')
    (WORK/'experiment_manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')


def publish():
    metadata=json.loads((WORK/'experiment_manifest.json').read_text(encoding='utf-8'))
    assert sha(WORK/'atomic.parquet')==metadata['atomic_sha256']
    assert sha(WORK/'directional_hypotheses.csv')==metadata['directional_review']['hypotheses_sha256']
    assert sha(ROOT/'utils/directional_review.py')==metadata['directional_review']['implementation_sha256']
    for name in ('acceptance_verification.json','dashboard_verification.json','directional_verification.json','unit_test_verification.json'):
        check=json.loads((WORK/name).read_text(encoding='utf-8'))
        if name in ('acceptance_verification.json','directional_verification.json'):
            assert check['atomic_sha256']==sha(WORK/'atomic.parquet')
            assert check['summary_sha256']==sha(WORK/'summary_ic.parquet')
        if name=='dashboard_verification.json':assert check['html_sha256']==sha(WORK/'全部因子IC与衰减.html')
        if name=='unit_test_verification.json':
            assert check['passed'] and check['errors']==check['failures']==0
            for test,digest in check['test_sha256'].items():assert sha(ROOT/'tests'/test)==digest
        metadata['verification'][name.replace('.json','')]=check
    for name in ['utils/directional_review.py','scripts/review_directional_ic.py','scripts/render_opening_ic.py','scripts/filter_positive_ic.py',
                 'utils/ic_dashboard.html','scripts/verify_opening_artifacts.py','scripts/verify_directional_review.py',
                 'tests/test_opening_directional_review.py','tests/opening_dashboard.cjs','scripts/test_opening.py']:
        metadata['source_sha256'][name]=sha(ROOT/name)
    metadata['directional_review_git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    metadata['published']=True;metadata['published_at']=datetime.now(timezone.utc).isoformat()
    for path in WORK.iterdir():
        if path.is_file() and path.name not in ('atomic.parquet','experiment_manifest.json'):
            os.replace(path,OUT/path.name)
    os.replace(WORK/'atomic.parquet',ATOMIC)
    metadata['artifact_sha256']={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='experiment_manifest.json'}
    (OUT/'experiment_manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Published reviewed research: 38 dates / 9500 tasks /',metadata['computed_outputs'],'computed outputs',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','render','publish'])
    globals()[p.parse_args().action]()
