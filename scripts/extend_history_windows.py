"""Extend the reviewed experiment to integer history windows 1..10 seconds.

Existing values/labels/IC are immutable. Only missing window versions are computed.
Feature days and IC families are checkpointed for resumable native execution.
"""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import os,sys,json,shutil,subprocess
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from utils.opening_ic import ICConfig,SessionData,feature_registry
from utils.directional_review import reviewed_registry,enrich_atomic
from utils.ic_statistics import evaluate_all
from scripts.run_opening_ic import sha,feature_audit,evaluation_fingerprint
from scripts.audit_opening_data import RAW,EXPECTED_SHA256

OUT=ROOT/'result/opening_execution';WORK=OUT/'.window_pending';BASE=ROOT/'.cache/history_window_base'
WINDOWS=tuple(range(1,11));VERSION='history-input-1-10s-20260922-v1'
CONFIG=ICConfig(history_seconds=WINDOWS)


def registry():
    old=pd.read_csv(BASE/'feature_registry.csv')
    complete=reviewed_registry(feature_registry(CONFIG))
    new=complete.loc[~complete.factor.isin(old.factor)].copy()
    new['window_extension']=True
    new['prior_status']='新增回看参数，沿用同表达的机制假设；在本窗口 IC 计算前登记；本批期货数据已见'
    old['window_extension']=False
    return pd.concat([old,new],ignore_index=True)


def archive():
    BASE.mkdir(parents=True,exist_ok=True)
    fingerprint=BASE/'snapshot.json'
    if fingerprint.exists():
        info=json.loads(fingerprint.read_text(encoding='utf-8'))
        for n,h in info.items():assert sha(BASE/n)==h
        return
    for name in ['feature_registry.csv','factor_hypotheses.csv','directional_hypotheses.csv','experiment_manifest.json',
                 'summary_ic.parquet','daily_ic.parquet','atomic.parquet']:
        source=ROOT/'sample_snapshot_原子执行总表.parquet' if name=='atomic.parquet' else OUT/name
        shutil.copy2(source,BASE/name)
    info={p.name:sha(p) for p in BASE.iterdir() if p.is_file()}
    fingerprint.write_text(json.dumps(info,indent=2),encoding='utf-8')


def feature_day(date,frame):
    start=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai')
    data=SessionData(frame,start,CONFIG)
    rows=[]
    for t in range(11,65):
        row=data.features(t);row['decision_elapsed_seconds']=t;rows.append(row)
    values=enrich_atomic(pd.DataFrame(rows),WINDOWS)
    names=registry().factor.tolist()
    values=values[['decision_elapsed_seconds']+[c for n in names for c in (n,n+'__status')]]
    path=WORK/'feature_days'/f'{date}.parquet';values.to_parquet(path,index=False,compression='zstd')
    return date


def features():
    archive();WORK.mkdir(parents=True,exist_ok=True);(WORK/'feature_days').mkdir(exist_ok=True)
    reg=registry();freeze=WORK/'window_hypotheses.csv';reg.to_csv(freeze,index=False,encoding='utf-8-sig')
    meta=json.loads((BASE/'experiment_manifest.json').read_text(encoding='utf-8'))
    source_hashes={n:sha(ROOT/n) for n in ['utils/opening_features.py','utils/opening_ic.py','utils/factor_catalog.py','utils/directional_review.py']}
    manifest_path=WORK/'window_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        assert manifest['feature_sources']==source_hashes and manifest['hypotheses_sha256']==sha(freeze)
    else:
        manifest=dict(version=VERSION,windows=list(WINDOWS),registered_at=datetime.now(timezone.utc).isoformat(),
            hypotheses_sha256=sha(freeze),feature_sources=source_hashes,raw_sha256=EXPECTED_SHA256)
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    assert sha(RAW)==EXPECTED_SHA256
    raw=pd.read_parquet(RAW);raw['Datetime']=pd.to_datetime(raw.Datetime).dt.tz_convert('Asia/Shanghai')
    raw['source_row']=np.arange(len(raw));dates=raw.Datetime.dt.strftime('%Y-%m-%d')
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs=[]
        for date in meta['included_dates']:
            if (WORK/'feature_days'/f'{date}.parquet').exists():continue
            start=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai')
            frame=raw.loc[dates.eq(date)&raw.Datetime.ge(start)&raw.Datetime.le(start+pd.Timedelta(seconds=65))].copy()
            jobs.append(pool.submit(feature_day,date,frame))
        for job in as_completed(jobs):print('Window features completed',job.result(),flush=True)
    del raw
    old=pd.read_parquet(BASE/'atomic.parquet');oldreg=pd.read_csv(BASE/'feature_registry.csv')
    additions=[];checked=0
    newnames=reg.loc[~reg.factor.isin(oldreg.factor),'factor'].tolist()
    for date,day in old.groupby('trade_date',sort=False):
        computed=pd.read_parquet(WORK/'feature_days'/f'{date}.parquet').set_index('decision_elapsed_seconds')
        aligned=computed.loc[day.decision_elapsed_seconds].reset_index(drop=True)
        for r in oldreg.itertuples():
            np.testing.assert_allclose(aligned[r.factor],day[r.factor],atol=1e-10,rtol=1e-12,equal_nan=True)
            assert aligned[r.factor+'__status'].tolist()==day[r.factor+'__status'].tolist(),r.factor
            checked+=len(day)
        part=aligned[[c for n in newnames for c in (n,n+'__status')]];part.index=day.index;additions.append(part)
    atomic=pd.concat([old,pd.concat(additions).sort_index()],axis=1)
    pd.testing.assert_frame_equal(atomic[old.columns],old)
    meta.update(config=CONFIG.to_dict(),registry_outputs=len(reg),computed_outputs=int(reg.evaluate.sum()),
        quality_outputs=int(reg.kind.eq('quality').sum()),alias_outputs=int(reg.kind.eq('alias').sum()),
        feature_columns=reg.loc[reg.evaluate,'factor'].tolist(),window_extension=manifest,published=False)
    meta['fixed_feature_parameters']['history_seconds']=list(WINDOWS)
    meta['verification']={k:v for k,v in meta['verification'].items() if not k.endswith('_verification')}
    meta['verification']['existing_feature_values_and_statuses_unchanged']=checked
    table=pa.Table.from_pandas(atomic,preserve_index=False)
    embedded={k:v for k,v in meta.items() if k not in ('atomic_sha256','artifact_sha256')}
    table=table.replace_schema_metadata({**table.schema.metadata,b'atomic_ic':json.dumps(embedded,ensure_ascii=False).encode()})
    pq.write_table(table,WORK/'atomic.parquet',compression='zstd')
    meta['atomic_sha256']=sha(WORK/'atomic.parquet')
    reg.to_csv(WORK/'feature_registry.csv',index=False,encoding='utf-8-sig');feature_audit(atomic,reg,WORK)
    for name in ['factor_hypotheses.csv','directional_hypotheses.csv']:
        shutil.copy2(BASE/name,WORK/name)
    for name in ['excluded_openings.csv','dashboard_qa_probes.json','有方向逻辑预期审阅.md','逻辑预期逐项审阅.csv']:
        shutil.copy2(OUT/name,WORK/name)
    (WORK/'experiment_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FEATURES READY',len(atomic),len(reg),int(reg.evaluate.sum()),checked,flush=True)


def evaluate_family(family,names):
    out=WORK/'ic_families'/family;out.mkdir(parents=True,exist_ok=True)
    done=out/'complete.json'
    meta=json.loads((WORK/'experiment_manifest.json').read_text(encoding='utf-8'))
    fingerprint=dict(atomic_sha256=meta['atomic_sha256'],statistics_sha256=sha(ROOT/'utils/ic_statistics.py'))
    if done.exists():
        saved=json.loads(done.read_text(encoding='utf-8'))
        assert saved['fingerprint']==fingerprint
        assert sha(out/'daily_ic.parquet')==saved['daily_sha256'] and sha(out/'summary_ic.parquet')==saved['summary_sha256']
        return saved
    reg=pd.read_csv(WORK/'feature_registry.csv');reg=reg.loc[reg.factor.isin(names)]
    labels=[f'y_{anchor}_{prefix}{u}s' for anchor in ('decision','arrival') for prefix in ('','incremental_') for u in range(1,31)]
    atomic=pd.read_parquet(WORK/'atomic.parquet',columns=['trade_date','session','task_time','observation_seconds']+names+labels)
    summary,n=evaluate_all(atomic,reg,CONFIG,out)
    record=dict(family=family,rows=n,summary_rows=len(summary),fingerprint=fingerprint,daily_sha256=sha(out/'daily_ic.parquet'),summary_sha256=sha(out/'summary_ic.parquet'))
    done.write_text(json.dumps(record,indent=2),encoding='utf-8');return record


def evaluate():
    manifest=json.loads((WORK/'window_manifest.json').read_text(encoding='utf-8'))
    for n,h in manifest['feature_sources'].items():assert sha(ROOT/n)==h
    assert sha(WORK/'window_hypotheses.csv')==manifest['hypotheses_sha256']
    reg=pd.read_csv(WORK/'feature_registry.csv');new=reg.loc[reg.evaluate & reg.window_extension]
    groups=list(new.groupby('family_id',sort=False));results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(evaluate_family,f,g.factor.tolist()) for f,g in groups]
        for job in as_completed(jobs):
            result=job.result();results.append(result);print('WINDOW IC READY',result['family'],result['rows'],flush=True)
    old=pd.read_parquet(BASE/'summary_ic.parquet')
    summary=pd.concat([old]+[pd.read_parquet(WORK/'ic_families'/f/'summary_ic.parquet') for f,_ in groups],ignore_index=True)
    summary.to_parquet(WORK/'summary_ic.parquet',compression='zstd',index=False)
    summary.to_csv(WORK/'summary_ic.csv',index=False,encoding='utf-8-sig')
    pd.testing.assert_frame_equal(summary.iloc[:len(old)],old)
    writer=None;count=0
    try:
        for file in [BASE/'daily_ic.parquet']+[WORK/'ic_families'/f/'daily_ic.parquet' for f,_ in groups]:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=100000):
                if writer is None:writer=pq.ParquetWriter(WORK/'daily_ic.parquet',batch.schema,compression='zstd')
                writer.write_batch(batch);count+=len(batch)
    finally:
        if writer is not None:writer.close()
    merged=pq.ParquetFile(WORK/'daily_ic.parquet').iter_batches(batch_size=100000);checked=0
    for batch in pq.ParquetFile(BASE/'daily_ic.parquet').iter_batches(batch_size=100000):
        assert batch.equals(next(merged).slice(0,len(batch)));checked+=len(batch)
    meta=json.loads((WORK/'experiment_manifest.json').read_text(encoding='utf-8'))
    meta.update(ic_daily_rows=count,ic_summary_rows=len(summary))
    meta['verification'].update(existing_summary_rows_unchanged=len(old),existing_daily_rows_unchanged=checked)
    meta['window_extension'].update(new_computed_outputs=len(new),evaluation_completed_at=datetime.now(timezone.utc).isoformat())
    (WORK/'experiment_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EVALUATION READY',count,len(summary),flush=True)


def render():
    from scripts.render_opening_ic import render_report
    reg=pd.read_csv(WORK/'feature_registry.csv');coverage=pd.read_csv(WORK/'factor_coverage.csv')
    summary=pd.read_parquet(WORK/'summary_ic.parquet');meta=json.loads((WORK/'experiment_manifest.json').read_text(encoding='utf-8'))
    render_report(summary,reg,coverage,meta,WORK)
    default=summary.loc[summary.observation_seconds.eq(1)&summary.anchor.eq('decision')&summary.label_type.eq('cumulative')&summary.target.eq('signed')&summary.pair_set.eq('own')]
    supported=default.groupby('factor').valid_days.max()
    lines=['# 可输入回看窗口说明','',
        '同一计算表达合并为一行。上方输入默认回看秒数，也可在该因子行内单独输入；按 Enter 或离开输入框应用。当前已完整计算 1–10 秒整数，不支持的小数或范围外数值会提示且保持原值，不做 IC 插值。','',
        '**182 种计算表达，对应 1,271 个参数版本**；另外保留 6 个别名和 1,100 个质量版本。合并只改变展示，没有将不同窗口的 IC 平均到一起，也没有自动选最优窗口。','',
        '## 参数边界','',
        '- 普通滚动窗口、均线内层窗口、其方向交互可输入 1–10 秒。每行名称、公式、IC、覆盖和详情同步指向实际窗口版本。',
        '- 前后两段仍各 2／5 秒；M04/M05 滞后仍 2 秒；M06 短长窗口仍 5／10 秒；M07–M09 外层仍 5 秒，只调整内层窗口。',
        '- B09 恢复历史仍 10 秒、事件观察 2 秒；D09 滞后相关仍 10 秒历史、2 秒滞后。此类固定参数行单独标注，不随普通回看输入变动。',
        '- 快照、开盘累计、单次事件年龄没有普通滚动窗口，不假装可以调整其时间含义。',
        '- 原始登记只含旧 5／10 秒版本；如从 3 秒切回原始登记，界面会提示并恢复 5 秒。',
        '- 共同样本仍针对当前窗口版本，要求跨全部观察期和 30 个未来期限有效；不跨回看秒数求交集。切换回看参数可能改变有效样本，应一起看覆盖率。','',
        '## 支持不足不会放宽门槛','',
        '历史斜率仍需至少 3 个正权重的不同时间点，历史相关仍需至少 8 对，逐日 IC 仍需至少 20 对。短窗口因此可能无因子值或无有效 IC；完整列出，缺失不填零。','',
        '| 回看秒数 | 计算版本数（含固定参数） | 至少一行因子有效 | 默认口径至少一期有有效日 IC |','|---|---:|---:|---:|']
    for h in range(1,11):
        group=coverage.loc[coverage.evaluate & coverage.history_seconds.eq(h)]
        lines.append(f'| {h} | {len(group)} | {int(group.valid_rows.gt(0).sum())} | {int(supported.reindex(group.factor).gt(0).sum())} |')
    lines+=['','## 保留与复核','',
        '原有 303 个计算版本及全部旧列、标签、状态保留；新增窗口仅追加列。原有 727,200 行 IC 汇总和 27,633,600 行逐日 IC 保持一致。仍为 38 日、9,500 行，无开发／验证划分。',
        '新窗口假设在本次计算前记录于 window_hypotheses.csv，仍属于已见期货数据上的探索性研究。参数扩展不会增加独立因子数量，不以多窗口结果自动反改原假设。','',
        '完整原始版本保留在 feature_registry.csv、factor_coverage.csv、summary_ic.parquet / CSV 与 daily_ic.parquet。看板可切换“展开全部窗口版本”复核。','']
    (WORK/'可输入回看窗口说明.md').write_text('\n'.join(lines),encoding='utf-8')
    (WORK/'experiment_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')


def publish():
    meta=json.loads((WORK/'experiment_manifest.json').read_text(encoding='utf-8'))
    assert sha(WORK/'atomic.parquet')==meta['atomic_sha256']
    assert sha(WORK/'window_hypotheses.csv')==meta['window_extension']['hypotheses_sha256']
    for name in ('acceptance_verification.json','window_verification.json','dashboard_verification.json','unit_test_verification.json'):
        check=json.loads((WORK/name).read_text(encoding='utf-8'))
        if name in ('acceptance_verification.json','window_verification.json'):
            assert check['atomic_sha256']==meta['atomic_sha256']
            assert check['summary_sha256']==sha(WORK/'summary_ic.parquet')
        if name=='dashboard_verification.json':assert check['html_sha256']==sha(WORK/'全部因子IC与衰减.html')
        if name=='unit_test_verification.json':
            assert check['passed'] and check['errors']==check['failures']==0
            for test,h in check['test_sha256'].items():assert sha(ROOT/'tests'/test)==h
        meta['verification'][name.removesuffix('.json')]=check
    # The fingerprint also makes this larger parameter grid independently reproducible.
    atomic=pd.read_parquet(WORK/'atomic.parquet');reg=pd.read_csv(WORK/'feature_registry.csv')
    evaluation=dict(fingerprint=evaluation_fingerprint(atomic,reg,CONFIG),daily_rows=meta['ic_daily_rows'],
        daily_sha256=sha(WORK/'daily_ic.parquet'),summary_sha256=sha(WORK/'summary_ic.parquet'))
    (WORK/'evaluation_manifest.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in list(meta['source_sha256'])+['scripts/extend_history_windows.py','scripts/verify_history_windows.py','tests/test_opening_history_windows.py']:
        meta['source_sha256'][name]=sha(ROOT/name)
    meta['window_extension_git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    meta['published']=True;meta['published_at']=datetime.now(timezone.utc).isoformat()
    for p in WORK.iterdir():
        if p.is_file() and p.name not in ('atomic.parquet','experiment_manifest.json'):os.replace(p,OUT/p.name)
    os.replace(WORK/'atomic.parquet',ROOT/'sample_snapshot_原子执行总表.parquet')
    meta['artifact_sha256']={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='experiment_manifest.json'}
    (OUT/'experiment_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PUBLISHED: integer input 1..10; 182 expressions / 1271 versions; 9500 tasks',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['features','evaluate','render','publish'])
    globals()[p.parse_args().action]()
