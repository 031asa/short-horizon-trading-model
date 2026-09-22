"""Build, verify and publish the complete all-sample opening IC experiment."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from utils.opening_ic import ICConfig,RULES_VERSION,build_session_atomic,feature_registry,base_feature_registry
from utils.opening_schedule import ScheduleConfig
from utils.factor_catalog import FAMILIES,PARAMETERS,PRIOR_VERSION
from utils.ic_statistics import evaluate_all,summarize_daily
from utils.ic_validation import verify_actual_rows
from scripts.audit_opening_data import RAW,EXPECTED_SHA256,OUT
CURRENT_ATOMIC=ROOT/'sample_snapshot_原子执行总表.parquet'
FEATURE_SOURCES=['utils/opening_ic.py','utils/opening_features.py','utils/factor_catalog.py','utils/opening_schedule.py']

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def has_opening_data(frame,open_time):
    return bool(((frame.Datetime>=open_time)&(frame.Datetime<open_time+pd.Timedelta(seconds=60))).any())

def verify_previous_features(atomic):
    path=ROOT/'.cache/ic_compatibility/atomic.parquet'
    if not path.exists():return {'legacy_comparison':'reference unavailable'}
    keys=['trade_date','session','task_time','observation_seconds']
    old=pd.read_parquet(path).sort_values(keys).reset_index(drop=True)
    new=atomic.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(new[keys],old[keys])
    cols=base_feature_registry(ICConfig()).factor.tolist()+[f'y_{a}_{u}s' for a in ('decision','arrival') for u in range(1,6)]
    np.testing.assert_allclose(new[cols],old[cols],rtol=1e-12,atol=1e-10,equal_nan=True)
    return dict(legacy_feature_columns=30,legacy_feature_and_label_values_checked=len(new)*len(cols))

def verify_previous_ic(work):
    path=ROOT/'.cache/ic_compatibility/daily_ic.parquet'
    if not path.exists():return {'legacy_ic_comparison':'reference unavailable'}
    old=pd.read_parquet(path);old=old.loc[old.pair_set.eq('own')]
    keys=['trade_date','session','factor','observation_seconds','horizon_seconds','anchor','target','pair_set']
    values=['n_pairs','ic','rank_ic','status']
    new=pd.read_parquet(work/'daily_ic.parquet',columns=keys+values,
        filters=[('factor','in',base_feature_registry(ICConfig()).factor.tolist()),('horizon_seconds','<=',5),('pair_set','==','own'),('label_type','==','cumulative')])
    joined=old[keys+values].merge(new,on=keys,how='left',validate='one_to_one',suffixes=('_old','_new'),indicator=True)
    assert joined._merge.eq('both').all()
    for v in values[:-1]:np.testing.assert_allclose(joined[v+'_old'],joined[v+'_new'],rtol=1e-10,atol=1e-10,equal_nan=True)
    assert joined.status_old.eq(joined.status_new).all()
    return dict(legacy_own_ic_rows_checked=len(joined),legacy_ic_tolerance=1e-10)

def feature_audit(atomic,registry,output):
    rows=[]
    for r in registry.to_dict('records'):
        values=atomic[r['factor']];statuses=atomic[r['factor']+'__status']
        assert (values.notna()==statuses.eq('ok')).all(),r['factor']
        rows.append({**r,'valid_rows':int(values.notna().sum()),'coverage':float(values.notna().mean()),
            'valid_dates':int(atomic.loc[values.notna(),'trade_date'].nunique()),'distinct_values':int(values.nunique()),
            'status_counts':json.dumps(statuses.value_counts().to_dict(),ensure_ascii=False)})
    coverage=pd.DataFrame(rows)
    coverage.to_csv(output/'factor_coverage.csv',index=False,encoding='utf-8-sig')
    families=[]
    for fid,title in FAMILIES.items():
        group=coverage.loc[coverage.family_id.eq(fid)]
        families.append(dict(family_id=fid,family=title,outputs=len(group),computed=int(group.evaluate.sum()),
            aliases=int(group.kind.eq('alias').sum()),quality=int(group.kind.eq('quality').sum()),
            available_outputs=int(group.valid_rows.gt(0).sum()),status='quality_only' if fid=='R06' else 'implemented'))
    pd.DataFrame(families).to_csv(output/'family_coverage.csv',index=False,encoding='utf-8-sig')
    return coverage

def evaluation_fingerprint(atomic,registry,config):
    columns=['trade_date','session','task_time','observation_seconds']+registry.loc[registry.evaluate,'factor'].tolist()
    columns += [f'y_{a}_{prefix}{u}s' for a in ('decision','arrival') for prefix in ('','incremental_') for u in config.horizons_seconds]
    digest=hashlib.sha256(pd.util.hash_pandas_object(atomic[columns],index=False).to_numpy().tobytes()).hexdigest()
    return dict(input_hash=digest,columns=columns,minimum_pairs=config.minimum_pairs,statistics_source=sha(ROOT/'utils/ic_statistics.py'))

def run(cold_start=10.,observations=(1.,2.,3.,4.,5.),publish=False,features_only=False,reuse_features=False,reuse_evaluation=False):
    config=ICConfig(schedule=ScheduleConfig(cold_start_seconds=cold_start,observation_seconds=tuple(observations)))
    registry=feature_registry(config);work=OUT/'.pending';work.mkdir(parents=True,exist_ok=True)
    if sha(RAW)!=EXPECTED_SHA256:raise ValueError('Raw data fingerprint mismatch')
    if reuse_features:
        metadata=json.loads((work/'feature_manifest.json').read_text(encoding='utf-8'))
        assert metadata['config']==json.loads(json.dumps(config.to_dict()))
        assert metadata['feature_source_sha256']=={n:sha(ROOT/n) for n in FEATURE_SOURCES}
        assert metadata['atomic_sha256']==sha(work/'atomic.parquet')
        assert metadata['hypotheses_sha256']==sha(work/'factor_hypotheses.csv')
        atomic=pd.read_parquet(work/'atomic.parquet')
    else:
        prior_file=work/'factor_hypotheses.csv'
        prior_cols=['factor','expected_sign_signed','expected_sign_absolute','prior_version']
        # Preserve the original pre-IC directional hypothesis record when adding only diagnostics.
        keep_prior=False
        if prior_file.exists() and (work/'feature_manifest.json').exists():
            old=pd.read_csv(prior_file)
            keep_prior=old.loc[old.evaluate,prior_cols].reset_index(drop=True).equals(registry.loc[registry.evaluate,prior_cols].reset_index(drop=True))
        if keep_prior:
            registered_at=json.loads((work/'feature_manifest.json').read_text(encoding='utf-8'))['hypotheses_registered_at']
        else:
            registry.to_csv(prior_file,index=False,encoding='utf-8-sig')
            registered_at=datetime.now(timezone.utc).isoformat()
        raw=pd.read_parquet(RAW);raw['Datetime']=pd.to_datetime(raw.Datetime).dt.tz_convert('Asia/Shanghai')
        raw['source_row']=np.arange(len(raw));raw['trade_date']=raw.Datetime.dt.strftime('%Y-%m-%d')
        dates=sorted(raw.trade_date.unique());contract=raw.Contract.dropna().unique();assert len(contract)==1
        parts=[];included=[];excluded=[];gaps=[]
        for date in dates:
            open_time=pd.Timestamp(f'{date} 09:30',tz='Asia/Shanghai');frame=raw.loc[raw.trade_date.eq(date)]
            if not has_opening_data(frame,open_time):
                excluded.append(dict(trade_date=date,session='AM',reason='NO_FIRST_MINUTE_DATA'));continue
            included.append(date)
            stamps=frame.loc[(frame.Datetime>=open_time)&(frame.Datetime<=open_time+pd.Timedelta(seconds=96)),'Datetime'].reset_index(drop=True)
            intervals=stamps.diff().dt.total_seconds()
            for i in np.flatnonzero(intervals.gt(config.maximum_gap_seconds).to_numpy()):
                gaps.append(dict(trade_date=date,start=str(stamps.iloc[i-1]),end=str(stamps.iloc[i]),seconds=float(intervals.iloc[i])))
            parts.append(build_session_atomic(frame,open_time,'AM',contract[0],config))
            print(f'Features {date}: {len(parts)}/38 dates',flush=True)
        if not parts:raise ValueError('No valid openings')
        atomic=pd.concat(parts,ignore_index=True)
        assert not atomic.duplicated(['trade_date','session','task_time','observation_seconds']).any()
        assert atomic.task_elapsed_seconds.ge(cold_start).all() and atomic.task_elapsed_seconds.lt(60).all()
        assert registry.family_id.nunique()==59
        if cold_start==10 and tuple(observations)==(1,2,3,4,5):assert len(atomic)==9500 and len(included)==38
        assert atomic.groupby(['trade_date','session','decision_time'])[registry.factor.tolist()].nunique(dropna=False).le(1).all().all()
        checks=verify_previous_features(atomic)
        checks.update(verify_actual_rows(atomic,raw,config,included))
        print('Verified original outputs, causal prefixes and labels',checks,flush=True)
        metadata=dict(rules_version=RULES_VERSION,table_role='task_feature_label',phase='all_sample',config=config.to_dict(),
            fixed_feature_parameters=PARAMETERS,raw_sha256=EXPECTED_SHA256,raw_rows=len(raw),raw_dates=dates,
            included_dates=included,excluded_openings=excluded,opening_gaps=gaps,atomic_rows=len(atomic),family_count=59,
            registry_outputs=len(registry),computed_outputs=int(registry.evaluate.sum()),alias_outputs=int(registry.kind.eq('alias').sum()),
            quality_outputs=int(registry.kind.eq('quality').sum()),feature_columns=registry.loc[registry.evaluate,'factor'].tolist(),
            feature_source_sha256={n:sha(ROOT/n) for n in FEATURE_SOURCES},prior_version=PRIOR_VERSION,
            hypotheses_registered_at=registered_at,hypotheses_sha256=sha(work/'factor_hypotheses.csv'),
            feature_status_counts=atomic.feature_status.value_counts().to_dict(),verification=checks,
            time_range_policy='first_minute_tasks_followup_allowed',missing_opening_policy='exclude_entirely_missing_opening_keep_partial_gap_day',
            scientific_status='all-sample exploratory research; overlapping tasks; no fresh holdout',
            old_current_sha256=sha(CURRENT_ATOMIC) if CURRENT_ATOMIC.exists() else None)
        table=pa.Table.from_pandas(atomic,preserve_index=False)
        table=table.replace_schema_metadata({**(table.schema.metadata or {}),b'atomic_ic':json.dumps(metadata,ensure_ascii=False).encode()})
        pq.write_table(table,work/'atomic.parquet',compression='zstd')
        pd.testing.assert_frame_equal(atomic,pd.read_parquet(work/'atomic.parquet'))
        metadata['atomic_sha256']=sha(work/'atomic.parquet');metadata['verification']['parquet_roundtrip']=True
        (work/'feature_manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    registry.to_csv(work/'feature_registry.csv',index=False,encoding='utf-8-sig')
    coverage=feature_audit(atomic,registry,work)
    pd.DataFrame(metadata['excluded_openings']).to_csv(work/'excluded_openings.csv',index=False,encoding='utf-8-sig')
    if features_only:
        print(json.dumps({k:metadata[k] for k in ('atomic_rows','family_count','registry_outputs','computed_outputs','alias_outputs','quality_outputs')},ensure_ascii=False),flush=True)
        return metadata
    fingerprint=evaluation_fingerprint(atomic,registry,config)
    if reuse_evaluation:
        cached=json.loads((work/'evaluation_manifest.json').read_text(encoding='utf-8'))
        assert cached['fingerprint']==fingerprint,'IC inputs or method changed'
        assert cached['daily_sha256']==sha(work/'daily_ic.parquet')
        assert cached['summary_sha256']==sha(work/'summary_ic.parquet')
        summary=pd.read_parquet(work/'summary_ic.parquet');n_daily=cached['daily_rows']
        metadata['verification']['reused_ic_exact_input_and_method_match']=True
    else:
        summary,n_daily=evaluate_all(atomic,registry,config,work)
        cached=dict(fingerprint=fingerprint,daily_rows=n_daily,daily_sha256=sha(work/'daily_ic.parquet'),summary_sha256=sha(work/'summary_ic.parquet'))
        (work/'evaluation_manifest.json').write_text(json.dumps(cached,ensure_ascii=False,indent=2),encoding='utf-8')
    metadata['ic_daily_rows']=n_daily;metadata['ic_summary_rows']=len(summary)
    metadata['verification'].update(verify_previous_ic(work))
    metadata['source_pdf']={'path':'研究资料/开盘首分钟_L1特征汇总_双类排版版.pdf','sha256':sha(ROOT/'研究资料/开盘首分钟_L1特征汇总_双类排版版.pdf')}
    metadata['hypotheses_note']='Computed-output priors retain original pre-IC registration; subsequently expanded quality-only diagnostics are recorded as not_applicable in feature_registry.csv.'
    source_files=FEATURE_SOURCES+['utils/ic_statistics.py','utils/ic_validation.py','scripts/run_opening_ic.py','scripts/render_opening_ic.py','scripts/filter_positive_ic.py','utils/ic_dashboard.html','scripts/verify_opening_artifacts.py','tests/opening_dashboard.cjs','tests/test_opening_full_features.py']
    metadata['source_sha256']={n:sha(ROOT/n) for n in source_files}
    from scripts.render_opening_ic import render_report
    render_report(summary,registry,coverage,metadata,work)
    metadata['published']=False
    for name in ('acceptance_verification.json','dashboard_verification.json','unit_test_verification.json'):
        check_path=work/name
        if check_path.exists():
            checked=json.loads(check_path.read_text(encoding='utf-8'))
            if name.startswith('acceptance'):
                assert checked['atomic_sha256']==metadata['atomic_sha256']
                assert checked['summary_sha256']==sha(work/'summary_ic.parquet')
            elif name.startswith('dashboard'):
                assert checked['html_sha256']==sha(work/'全部因子IC与衰减.html'),'Re-run browser verification after changing HTML'
            metadata['verification'][name.replace('.json','')]=checked

    if publish:
        for path in work.iterdir():
            if path.name not in ('atomic.parquet','feature_manifest.json','experiment_manifest.json') and path.is_file():os.replace(path,OUT/path.name)
        os.replace(work/'atomic.parquet',CURRENT_ATOMIC)
        assert sha(CURRENT_ATOMIC)==metadata['atomic_sha256'];metadata['published']=True
        for obsolete in ('daily_ic.csv','ic_heatmap.png'):(OUT/obsolete).unlink(missing_ok=True)
    destination=OUT if publish else work
    metadata['artifact_sha256']={p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name not in ('experiment_manifest.json','feature_manifest.json','atomic.parquet')}
    (destination/'experiment_manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(rows=len(atomic),families=59,computed=metadata['computed_outputs'],daily_rows=n_daily,published=metadata['published'],checks=metadata['verification']),ensure_ascii=False,indent=2),flush=True)
    return metadata

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cold-start',type=float,default=10.)
    parser.add_argument('--observations',type=float,nargs='+',default=[1.,2.,3.,4.,5.])
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--features-only',action='store_true')
    parser.add_argument('--reuse-features',action='store_true')
    parser.add_argument('--reuse-evaluation',action='store_true')
    args=parser.parse_args();run(args.cold_start,args.observations,args.publish,args.features_only,args.reuse_features,args.reuse_evaluation)
