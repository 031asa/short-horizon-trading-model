"""Frozen small-model pilot; run with native Python and project package caches."""
from pathlib import Path
import sys, json, hashlib, time, warnings
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'.cache/model-packages')); sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_prediction import *

OUT = ROOT/'result/opening_prediction'
ATOMIC = ROOT/'sample_snapshot_原子执行总表.parquet'

def save_json(path, data):
    # Replace nonfinite floats with JSON null, including nested objects.
    def clean(x):
        if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [clean(v) for v in x]
        if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer): return int(x)
        return x
    path.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2),encoding='utf-8')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    features = sorted({f for s in specs() for f in s['features']})
    columns = ['trade_date','task_time','observation_seconds','decision_time','feature_time','feature_source_age_seconds']+features+[f'y_decision_{h}s' for h in [1,2,3]]
    frame = pd.read_parquet(ATOMIC,columns=columns)
    frame.trade_date = frame.trade_date.astype(str)
    dates = sorted(frame.trade_date.unique())
    assert len(frame)==9500 and len(dates)==38
    assert not frame.duplicated(['trade_date','task_time','observation_seconds']).any()
    assert (frame.feature_time <= frame.decision_time).all()
    frozen = dict(rules=RULES, models=specs(), input_sha256=sha(ATOMIC), dates=dates,
                  folds=[dict(train=a,test=b) for a,b in folds(dates)])
    # Freeze BEFORE any model fit. Subsequent runs must use identical research settings/input.
    frozen_path = OUT/'实验口径.json'
    if frozen_path.exists(): assert json.loads(frozen_path.read_text(encoding='utf-8'))==frozen
    else: save_json(frozen_path,frozen)
    predictions, fits, timings, exclusions = [], [], [], []
    configs = specs()
    for h in RULES['horizons']:
        label = f'y_decision_{h}s'
        ready = {}
        for w in RULES['windows']:
            g = frame[frame.observation_seconds==w].copy()
            ok_x = np.isfinite(g[factors(w)].to_numpy()).all(axis=1)
            ok_y = np.isfinite(g[label].to_numpy())
            exclusions.append(dict(w=w,horizon=h,scheduled=len(g),feature_invalid=int((~ok_x).sum()),
                                   label_invalid=int((~ok_y).sum()),valid=int((ok_x&ok_y).sum())))
            g = g[ok_x&ok_y].copy(); g['y']=np.sign(g[label]).astype(int)
            ready[w]=g
        for fold, (train_dates,test_dates) in enumerate(folds(dates),1):
            fold_predictions, fold_fits = {}, {}
            for spec in configs:
                w = spec['w']; names = spec['features']; model_id=spec['model']
                g=ready[w]; train=g[g.trade_date.isin(train_dates)]; test=g[g.trade_date.isin(test_dates)]
                assert train.trade_date.max() < test.trade_date.min()
                winner,trials=tune(train,names,'y')
                model=fit_model(train[names].to_numpy(),train.y.to_numpy(),train.trade_date.to_numpy(),winner['C'])
                p=model.predict_proba(test[names].to_numpy())
                prior=class_prior(train.y.to_numpy(),train.trade_date.to_numpy())
                result=test[['trade_date','task_time','decision_time','observation_seconds','y',label]].copy().rename(columns={label:'change_ticks'})
                result['model']=model_id; result['fold']=fold; result['horizon']=h
                for j,s in enumerate(['down','flat','up']): result[f'p_{s}']=p[:,j]; result[f'baseline_{s}']=prior[j]
                predictions.append(result); fold_predictions[model_id]=result
                detail=dict(model=model_id,fold=fold,horizon=h,w=w,features=names,C=winner['C'],
                    validation_logloss=winner['validation_logloss'],trials=trials,train_rows=len(train),test_rows=len(test),
                    train_dates=train_dates,test_dates=test_dates,classes=model.classes_.tolist(),
                    mean=model['scale'].mean_.tolist(),scale=model['scale'].scale_.tolist(),
                    coefficients=model['model'].coef_.tolist(),intercept=model['model'].intercept_.tolist(),
                    baseline=prior.tolist(),iterations=model['model'].n_iter_.tolist(),gradient_max=model['model'].gradient_max)
                fits.append(detail); fold_fits[model_id]=detail
                if fold==4:
                    x=test[names].to_numpy()[:1]
                    for _ in range(30): model.predict_proba(x)
                    times=[]
                    for _ in range(300):
                        start=time.perf_counter_ns(); model.predict_proba(x); times.append((time.perf_counter_ns()-start)/1e6)
                    timings.append(dict(model=model_id,horizon=h,features=len(names),p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),
                                        scope='one-row StandardScaler + predict_proba; excludes factor calculation and feed/order latency'))
            # A single-factor benchmark selected ONLY within outer training dates.
            best=min([f'single_{i}' for i in range(6)],key=lambda m:fold_fits[m]['validation_logloss'])
            chosen=fold_predictions[best].copy(); chosen['model']='selected_single'; predictions.append(chosen)
            fits.append(dict(model='selected_single',fold=fold,horizon=h,selected_model=best,train_dates=train_dates,test_dates=test_dates))
            print(f'horizon={h}s fold={fold}/4 completed; training-only best single={best}',flush=True)
    pred=pd.concat(predictions,ignore_index=True)
    assert np.isfinite(pred[['p_down','p_flat','p_up']]).all().all()
    assert np.allclose(pred[['p_down','p_flat','p_up']].sum(axis=1),1)
    assert pred.trade_date.nunique()==18
    # Intersection across all five FULL window configurations, separately for each horizon.
    keys=['trade_date','task_time','horizon']
    full=pred[pred.model.isin([f'full_{w}' for w in range(1,6)])]
    counts=full.groupby(keys).model.nunique()
    common=counts[counts==5].reset_index()[keys]
    own=pred.assign(sample='own_window')
    cross=pred.merge(common,on=keys,validate='many_to_one').assign(sample='common_windows')
    for h in [1,2,3]:
        sets=[set(zip(g.trade_date,g.task_time)) for _,g in cross[cross.horizon==h].groupby('model')]
        assert all(s==sets[0] for s in sets)
    allpred=pd.concat([own,cross],ignore_index=True)
    summary,daily,cms,conf=evaluate(allpred)
    baseline_summary,baseline_daily,baseline_cm,baseline_conf=evaluate(allpred,prefix='baseline')
    baseline_summary['is_baseline']=True; summary['is_baseline']=False
    comparisons=[]
    # Attach a matched class-prior benchmark for each full-model comparison.
    b=baseline_daily[baseline_daily.model=='full_1'].copy(); b['model']='class_prior'
    comparison_daily=pd.concat([daily,b],ignore_index=True)
    for model,ref in [('full_1','snapshot'),('full_1','selected_single'),('full_1','single_2'),('full_1','class_prior'),('snapshot','selected_single')]+[(f'full_{w}','full_1') for w in range(2,6)]:
        for h in [1,2,3]:
            for metric in ['accuracy','logloss','balanced_accuracy']:
                comparisons.append(paired_daily(comparison_daily,model,ref,h,metric))
    pred.to_parquet(OUT/'逐任务预测.parquet',index=False)
    allpred[['trade_date','task_time','model','horizon','sample']].to_parquet(OUT/'评价样本清单.parquet',index=False)
    summary.to_csv(OUT/'模型评价.csv',index=False,encoding='utf-8-sig')
    baseline_summary.to_csv(OUT/'比例基准评价.csv',index=False,encoding='utf-8-sig')
    daily.to_csv(OUT/'逐日评价.csv',index=False,encoding='utf-8-sig')
    conf.to_csv(OUT/'置信度与覆盖.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(comparisons).to_csv(OUT/'配对模型差异.csv',index=False,encoding='utf-8-sig')
    save_json(OUT/'各折模型参数.json',fits)
    data=dict(rules=RULES,models=configs+[dict(model='selected_single',name='训练期选出的单因子',w=1,features=[])],
              summary=summary.to_dict('records'),baseline=baseline_summary.to_dict('records'),confusion=cms,
              confidence=conf.to_dict('records'),paired=comparisons,timing=timings,exclusions=exclusions,
              daily=daily.to_dict('records'),baseline_daily=baseline_daily.to_dict('records'),folds=frozen['folds'])
    save_json(OUT/'看板数据.json',data)
    manifest=dict(input_sha256=sha(ATOMIC),input_unchanged=sha(ATOMIC)==frozen['input_sha256'],
                  atomic_rows=len(frame),dates=len(dates),test_dates=18,predictions=len(pred),
                  chronological_folds=True,common_keys_equal=True,finite_normalized_probabilities=True,
                  source_sha256={p:sha(ROOT/p) for p in ['utils/opening_prediction.py','scripts/run_opening_prediction.py']},
                  versions=dict(solver='NumPy ridge softmax Newton, gradient tolerance 1e-8',numpy=np.__version__,pandas=pd.__version__))
    save_json(OUT/'模型验收.json',manifest)
    print(summary[(summary['sample']=='common_windows') & summary.model.isin(['full_1','snapshot','selected_single','full_5'])][['model','horizon','accuracy','balanced_accuracy','logloss','flat_recall','n']].to_string(index=False),flush=True)

if __name__=='__main__':run()
