"""Implement the fixed-core enhancement experiment, preserving previous outputs."""
from pathlib import Path
import os,sys,json,hashlib,argparse
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ.setdefault(name,'1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from utils.opening_ic import SessionData,ICConfig
from utils.opening_schedule import ScheduleConfig
from utils.opening_task_window import ALL,BASE,NAMES,bounded_features
from utils.opening_prediction import folds,fit_model
from utils.opening_anchored import RULES,prepare,search,pick,pack,predict
from scripts.audit_opening_data import RAW,EXPECTED_SHA256
from scripts.run_opening_prediction import save_json

OUT=ROOT/'result/opening_prediction/three_factor_enhancement'
OLD=ROOT/'result/opening_prediction/no_cold_start'
KEY=['trade_date','task_elapsed_seconds']

def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'utils/opening_task_window.py',ROOT/'utils/opening_ic.py',ROOT/'utils/opening_anchored.py']}
    original={str(p.relative_to(ROOT)):sha(p) for p in OLD.rglob('*') if p.is_file()}
    original['sample_snapshot_原子执行总表.parquet']=sha(ROOT/'sample_snapshot_原子执行总表.parquet')
    assert sha(RAW)==EXPECTED_SHA256
    config=dict(rules=RULES,source_sha256=sources,raw_sha256=EXPECTED_SHA256,old_files=original)
    freeze=OUT/'实验口径.json'
    if freeze.exists():assert json.loads(freeze.read_text(encoding='utf-8'))==config
    else:save_json(freeze,config)
    target=OUT/'固定三因子原子表.parquet'
    if target.exists():
        assert json.loads((OUT/'数据复核.json').read_text(encoding='utf-8'))['atomic_sha256']==sha(target)
        return pd.read_parquet(target)
    raw=pd.read_parquet(RAW);rows=[];checks=0
    cfg=ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,observation_seconds=tuple(float(w) for w in RULES['windows'])),horizons_seconds=tuple(range(1,11)))
    for date,g in raw.groupby(raw.Date.astype(str),sort=True):
        if date in ['2026-07-20','2026-07-27']:continue
        d=SessionData(g,pd.Timestamp(date+' 09:30',tz='Asia/Shanghai'),cfg)
        labels={end:d.labels(end) for end in range(1,70)}
        for t in range(60):
            for w in RULES['windows']:
                e=t+w;lab=labels[e];feature=bounded_features(d,t,w)
                row=dict(trade_date=date,task_elapsed_seconds=t,window=w,decision_seconds=e,**feature)
                for h in range(1,11):
                    for kind,key in [('cum',f'y_decision_{h}s'),('inc',f'y_decision_incremental_{h}s')]:
                        row[f'{kind}_{h}']=lab[key];row[f'{kind}_{h}_status']=lab[key+'_status']
                rows.append(row)
                if t in [0,17,48] and w in [1,2,6,7,9,10]:
                    clipped=d.frame[(d.times>=t)&(d.times<=e)]
                    sub=SessionData(clipped,d.open_time,cfg);recomputed=bounded_features(sub,t,w)
                    np.testing.assert_allclose([feature[f] for f in ALL],[recomputed[f] for f in ALL],rtol=1e-10,atol=1e-10,equal_nan=True);checks+=1
        print('Built '+date+' (600 rows)',flush=True)
    a=pd.DataFrame(rows);assert len(a)==22800 and a.trade_date.nunique()==38
    assert not a.duplicated(KEY+['window']).any()
    assert (a.used_start.dropna()>=a.loc[a.used_start.notna(),'task_elapsed_seconds']).all()
    assert (a.source_seconds.dropna()<=a.loc[a.source_seconds.notna(),'decision_seconds']).all()
    old=pd.read_parquet(OLD/'无冷启动原子表.parquet')
    matched=a.merge(old,on=KEY+['window'],suffixes=('_new','_old'),validate='one_to_one');assert len(matched)==15960
    for f in ALL:np.testing.assert_allclose(matched[f+'_new'],matched[f+'_old'],rtol=0,atol=0,equal_nan=True)
    np.testing.assert_allclose(matched.cum_3,matched.y3,rtol=0,atol=0,equal_nan=True)
    for h in range(2,11):np.testing.assert_allclose(a[f'inc_{h}'],a[f'cum_{h}']-a[f'cum_{h-1}'],equal_nan=True)
    a.to_parquet(target,index=False)
    save_json(OUT/'数据复核.json',dict(rows=len(a),dates=38,old_feature_rows_equal=15960,old_columns_equal=len(ALL),
        strict_sliced_tasks=checks,incremental_identity=True,atomic_sha256=sha(target),names=NAMES))
    return a

def worker(job):
    fold,window,train_dates,test_dates,validation_keys=job
    checkpoint=OUT/f'fold{fold}_w{window}.json'
    digest=sha(ROOT/'utils/opening_anchored.py')
    if checkpoint.exists():
        result=json.loads(checkpoint.read_text(encoding='utf-8'));assert result['code_sha256']==digest
        return fold,window,result['selection']['candidate_count'],True
    a=pd.read_parquet(OUT/'固定三因子原子表.parquet');g=a[a.window==window]
    clean,info=prepare(g[g.trade_date.isin(train_dates)])
    selection,val=search(clean,info,pd.DataFrame(validation_keys),window)
    test=g[g.trade_date.isin(test_dates)].dropna(subset=info['pool']+[f'cum_{h}' for h in range(1,11)])
    models=[];predictions=[];validation=[];fitted={}
    for method,choice in selection['choices'].items():
        if choice is None:continue
        fs=choice['features'];v=val[KEY+['window']].copy();vp=predict(selection['inner_models'][method],val[fs].to_numpy())
        v['method']=method
        for i,s in enumerate(['down','flat','up']):v['p_'+s]=vp[:,i]
        validation.append(v)
        if fold=='final':continue
        model_key=(choice['candidate_id'],choice['C'])
        if model_key not in fitted:fitted[model_key]=pack(fit_model(clean[fs].to_numpy(),np.sign(clean.cum_3).to_numpy(int),clean.trade_date.to_numpy(),choice['C']))
        m=fitted[model_key]
        models.append(dict(fold=fold,window=window,method=method,structure=choice['structure'],features=fs,C=choice['C'],
            train_dates=train_dates,test_dates=test_dates,train_rows=len(clean),test_rows=len(test),**m))
        z=test[KEY+['window','decision_seconds','snapshot_count']].copy();p=predict(m,test[fs].to_numpy())
        z['fold']=fold;z['method']=method
        for i,s in enumerate(['down','flat','up']):z['p_'+s]=p[:,i]
        predictions.append(z)
    if predictions:pd.concat(predictions,ignore_index=True).to_parquet(OUT/f'fold{fold}_w{window}_预测.parquet',index=False)
    pd.concat(validation,ignore_index=True).to_parquet(OUT/f'fold{fold}_w{window}_验证.parquet',index=False)
    save_json(checkpoint,dict(code_sha256=digest,selection=selection,models=models))
    return fold,window,selection['candidate_count'],False

def run(workers):
    a=build();dates=sorted(a.trade_date.unique());jobs=[];foldkeys={}
    splits=[(str(i),tr,te) for i,(tr,te) in enumerate(folds(dates),1)]+[('final',dates,[])]
    for fold,tr,te in splits:
        keys=None
        for w in RULES['windows']:
            clean,info=prepare(a[(a.window==w)&a.trade_date.isin(tr)])
            val=clean[clean.trade_date.isin(info['validation_dates'])].dropna(subset=[f'cum_{h}' for h in range(1,11)])[KEY]
            keys=val if keys is None else keys.merge(val,on=KEY,validate='one_to_one')
        assert keys.groupby('trade_date').size().min()>=15
        foldkeys[fold]=keys.to_dict('records')
        jobs.extend((fold,w,tr,te,foldkeys[fold]) for w in RULES['windows'])
    save_json(OUT/'共同验证任务.json',foldkeys)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,job) for job in jobs]
        for f in as_completed(futures):
            fold,w,n,cached=f.result();print(f'fold={fold}, W={w}, candidates={n}, cached={cached}',flush=True)
    decisions=[];model_list=[];predictions=[];selections=[]
    for fold,tr,te in splits:
        records=[json.loads((OUT/f'fold{fold}_w{w}.json').read_text(encoding='utf-8')) for w in RULES['windows']]
        options=[r['selection']['choices']['selected'] for r in records];chosen=pick(options)
        decision=dict(fold=fold,train_dates=tr,test_dates=te,common_validation_tasks=len(foldkeys[fold]),choice=chosen,options=options)
        decisions.append(decision)
        if fold=='final':
            w=chosen['window'];record=records[w-1]['selection'];clean,_=prepare(a[(a.window==w)&a.trade_date.isin(tr)])
            final_models={}
            for method in ['selected','baseline']:
                c=record['choices'][method];fs=c['features']
                final_models[method]=dict(features=fs,C=c['C'],**pack(fit_model(clean[fs].to_numpy(),np.sign(clean.cum_3).to_numpy(int),clean.trade_date.to_numpy(),c['C'])))
            save_json(OUT/'下一批行情配置.json',dict(window=w,train_dates=tr,training_horizon=3,choice=chosen,models=final_models,
                status='full-history fit for new data only; no independent validation of this final fixed configuration'))
            continue
        for r in records:model_list.extend(r['models']);selections.append(dict(fold=fold,**r['selection']))
        regular=pd.concat([pd.read_parquet(OUT/f'fold{fold}_w{w}_预测.parquet') for w in RULES['windows']],ignore_index=True)
        adaptive=regular[(regular.window==chosen['window'])&regular.method.isin(['selected','baseline'])].copy()
        adaptive['method']='adaptive_'+adaptive.method
        predictions.extend([regular,adaptive])
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'逐任务预测.parquet',index=False)
    save_json(OUT/'模型参数.json',model_list);save_json(OUT/'选择过程.json',selections);save_json(OUT/'窗口选择.json',decisions)
    original=json.loads((OUT/'实验口径.json').read_text(encoding='utf-8'))['old_files']
    assert all(sha(ROOT/p)==v for p,v in original.items())
    print('Completed searches and predictions; previous files unchanged.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2)
    run(parser.parse_args().workers)
