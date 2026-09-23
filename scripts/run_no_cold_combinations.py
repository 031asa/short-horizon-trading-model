"""Rebuild zero-cold-start task windows and search small future-three-second models."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_ic import SessionData,ICConfig
from utils.opening_schedule import ScheduleConfig
from utils.opening_task_window import ALL,BASE,CATALOG,INTERACTIONS,NAMES,bounded_features
from utils.opening_small_combinations import RULES,select_small
from utils.opening_prediction import fit_model,folds,class_prior,score,daily_loss
from scripts.audit_opening_data import RAW,EXPECTED_SHA256
from scripts.run_opening_prediction import save_json
OUT=ROOT/'result/opening_prediction/no_cold_start'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    assert sha(RAW)==EXPECTED_SHA256
    old=ROOT/'sample_snapshot_原子执行总表.parquet'
    config=dict(rules=RULES,raw_sha256=EXPECTED_SHA256,previous_atomic_sha256=sha(old),features=[dict(factor=f,name=n,group=g,prior=p,formula=formula) for f,n,g,p,formula in CATALOG],interactions=INTERACTIONS)
    # JSON normalizes tuple interaction parent pairs to lists.
    config=json.loads(json.dumps(config))
    freeze=OUT/'实验口径.json'
    if freeze.exists():assert json.loads(freeze.read_text(encoding='utf-8'))==config
    else:save_json(freeze,config)
    source_hash={n:sha(ROOT/n) for n in ['utils/opening_task_window.py','utils/opening_ic.py']}
    target=OUT/'无冷启动原子表.parquet';cache=OUT/'特征复核.json'
    if target.exists():
        proof=json.loads(cache.read_text(encoding='utf-8'));assert proof['source_sha256']==source_hash and proof['atomic_sha256']==sha(target)
        return pd.read_parquet(target)
    raw=pd.read_parquet(RAW);rows=[];sessions={}
    cfg=ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,observation_seconds=tuple(float(w) for w in RULES['windows'])),history_seconds=(1,),horizons_seconds=(1,2,3))
    excluded=['2026-07-20','2026-07-27']
    for date,frame in raw.groupby(raw.Date.astype(str),sort=True):
        if date in excluded:continue
        opening=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai');d=SessionData(frame,opening,cfg);sessions[date]=d
        labels={t:d.labels(t) for t in range(1,70)}
        for task in RULES['task_seconds']:
            for w in RULES['windows']:
                row=bounded_features(d,task,w);lab=labels[task+w]
                rows.append(dict(trade_date=date,task_elapsed_seconds=task,window=w,decision_seconds=task+w,**row,
                       y3=lab['y_decision_3s'],label_status=lab['y_decision_3s_status'],label_end=lab['y_decision_3s_end_time']))
        print(f'features {date}: 420 task-window rows',flush=True)
    a=pd.DataFrame(rows);a['y']=np.sign(a.y3)
    assert len(a)==15960 and a.trade_date.nunique()==38
    assert (a.used_start.dropna()>=a.loc[a.used_start.notna(),'task_elapsed_seconds']).all()
    assert (a.source_seconds.dropna()<=a.loc[a.source_seconds.notna(),'decision_seconds']).all()
    # Independently recompute chosen ordinary formulas using a truly sliced stream.
    from utils.opening_features import FeatureEngine
    checks=0;prefix_checks=0
    sample=a[a.trade_date.isin(sorted(sessions)[::7])].iloc[::137]
    mapping={f:f for f in BASE+['F01_QI','F02_QIChange','A01_Spread']}
    for row in sample.to_dict('records'):
        d=sessions[row['trade_date']];t=row['task_elapsed_seconds'];w=row['window'];e=t+w
        clipped=d.frame[(d.times>=t)&(d.times<=e)]
        cc=ICConfig(schedule=cfg.schedule,history_seconds=(int(w),),horizons_seconds=(1,2,3))
        sub=SessionData(clipped,d.open_time,cc)
        expected=sub.features(e)
        for f in ALL:
            if f in INTERACTIONS:continue
            key=mapping.get(f,f'_unused_{f}')
            if f=='F03_OFI_window':key=f'F03_OFI_h{w}s'
            elif f not in mapping and f not in ['A02_LogDepth','D01_LogVolumeRate']:key=f'{f}_h{w}s'
            value=expected.get(key,np.nan)
            if f=='A02_LogDepth':value=np.log1p(expected.get('A02_Depth',np.nan))
            if f=='D01_LogVolumeRate':value=np.log1p(expected.get(f'D01_VolumeRate_h{w}s',np.nan))
            np.testing.assert_allclose(row[f],value,atol=1e-9,rtol=1e-9,equal_nan=True);checks+=1
        # Rebuilding from a slice removes BOTH past-before-task and future rows.
        bounded=bounded_features(sub,t,w)
        np.testing.assert_allclose([row[f] for f in ALL],[bounded[f] for f in ALL],atol=1e-9,rtol=1e-9,equal_nan=True);prefix_checks+=1
    a.to_parquet(target,index=False)
    assert sha(old)==config['previous_atomic_sha256']
    save_json(cache,dict(rows=len(a),dates=38,excluded=excluded,source_sha256=source_hash,atomic_sha256=sha(target),
            independent_formula_checks=checks,strict_task_slice_checks=prefix_checks,input_unchanged=True))
    return a

def run():
    a=build();dates=sorted(a.trade_date.unique());predictions=[];selection=[];models=[]
    for fold,(train_dates,test_dates) in enumerate(folds(dates),1):
        for w in RULES['windows']:
            checkpoint=OUT/f'fold{fold}_w{w}.json';pp=OUT/f'fold{fold}_w{w}.parquet'
            code_sha=sha(ROOT/'utils/opening_small_combinations.py')
            if checkpoint.exists():
                saved=json.loads(checkpoint.read_text(encoding='utf-8'));assert saved['code_sha256']==code_sha
                selection.append(saved['selection']);models.extend(saved['models']);predictions.append(pd.read_parquet(pp));continue
            g=a[(a.window==w)&a.y.notna()].copy()
            train=g[g.trade_date.isin(train_dates)]
            chosen,baseline,clean,val,info=select_small(train)
            record=dict(fold=fold,window=w,train_dates=train_dates,test_dates=test_dates,**info);selection.append(record)
            local=[];localmodels=[]
            prior=class_prior(clean.y.to_numpy(int),clean.trade_date.to_numpy())
            for kind,config in [('selected',chosen),('baseline',baseline)]:
                fs=config['features'];model=fit_model(clean[fs].to_numpy(),clean.y.to_numpy(int),clean.trade_date.to_numpy(),config['C'])
                # Thresholds must use predictions made BEFORE refitting validation dates.
                inner=clean[clean.trade_date.isin(info['train_days'])]
                vm=fit_model(inner[fs].to_numpy(),inner.y.to_numpy(int),inner.trade_date.to_numpy(),config['C'])
                vp=vm.predict_proba(val[fs].to_numpy());confidence=np.maximum(vp[:,0],vp[:,2])
                thresholds={str(level):float(np.quantile(confidence,1-level)) for level in RULES['coverage_levels']}
                test=g[g.trade_date.isin(test_dates)].dropna(subset=fs)
                p=model.predict_proba(test[fs].to_numpy())
                out=test[['trade_date','task_elapsed_seconds','window','decision_seconds','source_seconds','used_start','snapshot_count','y','y3','label_end']].copy()
                out['method']=kind;out['fold']=fold
                for j,side in enumerate(['down','flat','up']):out['p_'+side]=p[:,j];out['prior_'+side]=prior[j]
                for level,threshold in thresholds.items():out['threshold_'+level]=threshold
                local.append(out)
                localmodels.append(dict(fold=fold,window=w,method=kind,features=fs,C=config['C'],thresholds=thresholds,
                    mean=model['scale'].mean_.tolist(),scale=model['scale'].scale_.tolist(),coefficients=model['model'].coef_.tolist(),intercept=model['model'].intercept_.tolist(),
                    train_dates=train_dates,test_dates=test_dates,train_rows=len(clean),test_rows=len(test)))
            models.extend(localmodels);p=pd.concat(local,ignore_index=True);predictions.append(p);p.to_parquet(pp,index=False)
            save_json(checkpoint,dict(code_sha256=code_sha,selection=record,models=localmodels))
            print(f'fold={fold} W={w}: {len(chosen["features"])} factors, val loss={chosen["loss"]:.4f}, evaluated={info["candidates_evaluated"]}',flush=True)
    pred=pd.concat(predictions,ignore_index=True)
    # Choose W from training validation only; favour shorter W if within fixed tolerance.
    decisions=[];adaptive=[]
    for fold in range(1,5):
        candidates=[r for r in selection if r['fold']==fold];validation=[]
        for record in candidates:
            w=record['window'];chosen=record['selected'];fs=chosen['features']
            g=a[(a.window==w)&a.y.notna()&a.trade_date.isin(record['train_dates'])].dropna(subset=record['pool'])
            inside=g[g.trade_date.isin(record['train_days'])];val=g[g.trade_date.isin(record['validation_dates'])]
            m=fit_model(inside[fs].to_numpy(),inside.y.to_numpy(int),inside.trade_date.to_numpy(),chosen['C'])
            v=val[['trade_date','task_elapsed_seconds','y']].copy();p=m.predict_proba(val[fs].to_numpy())
            for j,side in enumerate(['down','flat','up']):v['p_'+side]=p[:,j]
            v['window']=w;validation.append(v)
        v=pd.concat(validation,ignore_index=True);n=v.groupby(['trade_date','task_elapsed_seconds']).size()
        common=n[n==len(RULES['windows'])].reset_index()[['trade_date','task_elapsed_seconds']]
        v=v.merge(common,on=['trade_date','task_elapsed_seconds'],validate='many_to_one')
        scores={int(w):daily_loss(g.y.to_numpy(int),g[['p_down','p_flat','p_up']].to_numpy(),g.trade_date.to_numpy()) for w,g in v.groupby('window')}
        v.to_parquet(OUT/f'fold{fold}_共同验证预测.parquet',index=False)
        minimum=min(scores.values())
        best=min([r for r in candidates if scores[r['window']]<=minimum+.002],key=lambda r:r['window'])
        decisions.append(dict(fold=fold,window=best['window'],features=best['selected']['features'],validation_loss=scores[best['window']],
                              common_validation_tasks=len(common),all_window_losses=scores))
        z=pred[(pred.fold==fold)&(pred.window==best['window'])].copy();z['method']='adaptive_'+z.method;adaptive.append(z)
    native=pd.concat([pred,*adaptive],ignore_index=True)
    native.to_parquet(OUT/'逐任务预测.parquet',index=False)
    save_json(OUT/'选择过程.json',selection);save_json(OUT/'模型参数.json',models);save_json(OUT/'训练期选择的窗口.json',decisions)
    evaluate(a,native,selection,decisions)

def evaluate(a,pred,selection,decisions):
    # Keep every method/window on exactly the same task IDs for primary comparison.
    regular=pred[pred.method.isin(['selected','baseline'])]
    count=regular.groupby(['trade_date','task_elapsed_seconds']).size()
    keys=count[count==14].reset_index()[['trade_date','task_elapsed_seconds']]
    common=pred.merge(keys,on=['trade_date','task_elapsed_seconds'],validate='many_to_one')
    common.to_parquet(OUT/'共同任务预测.parquet',index=False)
    daily=[];confidence=[]
    for scope,data in [('own',pred),('common',common)]:
        data=data.copy();data['model_id']=np.where(data.method.str.startswith('adaptive'),data.method,data.method+'_w'+data.window.astype(str))
        for (mid,date),g in data.groupby(['model_id','trade_date']):
            for segment,sub in [('all',g),('opening_0_9',g[g.task_elapsed_seconds<10]),('later_10_59',g[g.task_elapsed_seconds>=10])]:
                if len(sub)==0:continue
                p=sub[['p_down','p_flat','p_up']].to_numpy();y=sub.y.to_numpy(int);score_row,cm=score(y,p)
                prior,_=score(y,sub[['prior_down','prior_flat','prior_up']].to_numpy())
                daily.append(dict(sample=scope,model=mid,trade_date=date,segment=segment,**score_row,prior_accuracy=prior['accuracy'],prior_logloss=prior['logloss']))
                if segment!='all':continue
                conf=np.maximum(p[:,0],p[:,2]);direction=np.where(p[:,2]>=p[:,0],1,-1);hit=direction==y
                for level in RULES['coverage_levels']:
                    # Equal-rank coverage diagnostic uses scores only, never outcomes.
                    n=max(1,int(np.ceil(level*len(sub))));chosen=np.argsort(-conf,kind='stable')[:n]
                    live=conf>=sub['threshold_'+str(level)].to_numpy()
                    for mode,mask in [('fixed_rank',np.isin(np.arange(len(sub)),chosen)),('frozen_threshold',live)]:
                        confidence.append(dict(sample=scope,model=mid,trade_date=date,mode=mode,level=level,signals=int(mask.sum()),available=len(sub),hits=int(hit[mask].sum()),
                                     accuracy=float(hit[mask].mean()) if mask.any() else np.nan))
    d=pd.DataFrame(daily);c=pd.DataFrame(confidence);s=d.groupby(['sample','model','segment']).agg(
        days=('trade_date','nunique'),n=('n','sum'),accuracy=('accuracy','mean'),balanced_accuracy=('balanced_accuracy','mean'),
        logloss=('logloss','mean'),brier=('brier','mean'),prior_accuracy=('prior_accuracy','mean'),prior_logloss=('prior_logloss','mean'),flat_recall=('flat_recall','mean')).reset_index()
    cf=c.groupby(['sample','model','mode','level']).agg(days=('accuracy','count'),signals=('signals','sum'),available=('available','sum'),accuracy=('accuracy','mean'),hits=('hits','sum')).reset_index()
    cf['coverage']=cf.signals/cf.available;cf['pooled_accuracy']=cf.hits/cf.signals
    d.to_csv(OUT/'逐日评价.csv',index=False,encoding='utf-8-sig');s.to_csv(OUT/'模型汇总.csv',index=False,encoding='utf-8-sig');cf.to_csv(OUT/'命中率与覆盖.csv',index=False,encoding='utf-8-sig')
    comparison=[]
    for window in RULES['windows']+['adaptive']:
        sel='adaptive_selected' if window=='adaptive' else f'selected_w{window}'
        base='adaptive_baseline' if window=='adaptive' else f'baseline_w{window}'
        x=d[(d['sample']=='common')&(d.segment=='all')].pivot(index='trade_date',columns='model',values='logloss')
        diff=(x[base]-x[sel]).dropna().to_numpy();rng=np.random.default_rng(20260923)
        # Five-day circular blocks retain short serial dependence within resamples.
        starts=rng.integers(0,len(diff),size=(5000,4));ix=((starts[:,:,None]+np.arange(5))%len(diff)).reshape(5000,-1)[:,:len(diff)]
        ci=np.quantile(diff[ix].mean(1),[.025,.975])
        comparison.append(dict(window=window,logloss_improvement=float(diff.mean()),ci95=ci.tolist(),better_days=int((diff>0).sum()),days=len(diff)))
    save_json(OUT/'看板数据.json',dict(rules=RULES,names=NAMES,summary=s.to_dict('records'),confidence=cf.to_dict('records'),daily=d.to_dict('records'),
        selected=[dict(fold=r['fold'],window=r['window'],features=r['selected']['features'],validation_loss=r['selected']['loss'],vif=r['selected']['refit_stats']['vif'],candidate_pool=len(r['pool']),evaluated=r['candidates_evaluated']) for r in selection],
        adaptive=decisions,comparison=comparison,scheduled_test_tasks=1080,common_test_tasks=len(keys),
        feature_coverage=[dict(window=w,**{f:float(np.isfinite(g[f]).mean()) for f in ALL}) for w,g in a.groupby('window')]))
    print(s[(s['sample']=='common')&(s.segment=='all')][['model','n','accuracy','logloss','flat_recall']].to_string(index=False),flush=True)

if __name__=='__main__':run()
