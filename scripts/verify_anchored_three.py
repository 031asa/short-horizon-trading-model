"""Independent replay of selected models, common cohorts and selection rules."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_task_window import BASE,INTERACTIONS,ALL
from scripts.run_anchored_three import OUT,OLD,KEY,sha
from scripts.check_factor_collinearity import diagnose
from scripts.run_opening_prediction import save_json

def choose(records):
    if not records:return None
    amax=max(r['accuracy3'] for r in records)
    stage1=[r for r in records if amax-r['accuracy3']<=.005+1e-12]
    pmax=max(r['accuracy5to10'] for r in stage1)
    stage2=[r for r in stage1 if pmax-r['accuracy5to10']<=.005+1e-12]
    return sorted(stage2,key=lambda r:(r['window'],len(r['features']),r['logloss3'],r['C'],r['candidate_id']))[0]

def replay(m,frame):
    z=(frame[m['features']].to_numpy()-np.asarray(m['mean']))/np.asarray(m['scale'])
    q=z@np.asarray(m['coefficients']).T+np.asarray(m['intercept']);q-=q.max(1)[:,None]
    pr=np.exp(q);return pr/pr.sum(1)[:,None]

def weights_for(frame):
    return 1/frame.trade_date.map(frame.trade_date.value_counts()).to_numpy()

def run():
    a=pd.read_parquet(OUT/'固定三因子原子表.parquet');pred=pd.read_parquet(OUT/'逐任务预测.parquet')
    config=json.loads((OUT/'实验口径.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==value for p,value in config['old_files'].items())
    assert all(sha(ROOT/p)==value for p,value in config['source_sha256'].items())
    assert len(a)==22800 and a.trade_date.nunique()==38 and a.task_elapsed_seconds.min()==0
    assert not a.duplicated(KEY+['window']).any()
    assert set(a.window)==set(range(1,11))
    assert (a.used_start.dropna()>=a.loc[a.used_start.notna(),'task_elapsed_seconds']).all()
    assert (a.source_seconds.dropna()<=a.loc[a.source_seconds.notna(),'decision_seconds']).all()
    old=pd.read_parquet(OLD/'未来1至30秒标签.parquet')
    labels=[f'{kind}_{h}' for kind in ['cum','inc'] for h in range(1,11)]
    join=a.merge(old[['trade_date','decision_seconds']+labels].drop_duplicates(['trade_date','decision_seconds']),
        on=['trade_date','decision_seconds'],suffixes=('_new','_old'),validate='many_to_one')
    assert len(join)==len(a)
    for f in labels:np.testing.assert_allclose(join[f+'_new'],join[f+'_old'],rtol=0,atol=0,equal_nan=True)
    keys=json.loads((OUT/'共同验证任务.json').read_text(encoding='utf-8'))
    decisions=json.loads((OUT/'窗口选择.json').read_text(encoding='utf-8'))
    all_models=json.loads((OUT/'模型参数.json').read_text(encoding='utf-8'))
    selected_checks=0;validation_rows=0;prediction_rows=0;maxvif=0;all_choices={};counts=dict(candidates=0,eligible=0,successful_fits=0,failed_fits=0)
    for fold in ['1','2','3','4','final']:
        window_choices=[]
        for window in range(1,11):
            checkpoint=json.loads((OUT/f'fold{fold}_w{window}.json').read_text(encoding='utf-8'));r=checkpoint['selection']
            assert max(r['inner_dates'])<min(r['validation_dates'])
            counts['candidates']+=r['candidate_count'];counts['eligible']+=r['eligible_candidates']
            flat=[]
            for c in r['candidates']:
                assert set(BASE).issubset(c['features']) and len(c['features'])<=6
                for feature in c['features']:
                    if feature in INTERACTIONS:assert set(INTERACTIONS[feature]).issubset(c['features'])
                for t in c['trials']:
                    if t['status']=='OK':
                        assert c['inner_stats']['ok'] and c['refit_stats']['ok']
                        flat.append(dict(candidate_id=c['candidate_id'],features=c['features'],structure=c['structure'],window=window,**t));counts['successful_fits']+=1
                    else:counts['failed_fits']+=1
            winners={kind:choose([x for x in flat if x['structure']==kind]) for kind in ['baseline','add1','add2','composite']}
            winners['selected']=choose([x for x in winners.values() if x])
            for method,c in winners.items():
                actual=r['choices'][method]
                if c is None:assert actual is None;continue
                assert c['candidate_id']==actual['candidate_id'] and c['C']==actual['C'];selected_checks+=1
            window_choices.append(winners['selected'])
            clean=a[(a.window==window)&a.trade_date.isin(r['train_dates'])].dropna(subset=r['pool']+['cum_3'])
            inner=clean[clean.trade_date.isin(r['inner_dates'])]
            val=clean.merge(pd.DataFrame(keys[fold]),on=KEY,validate='one_to_one')
            saved=pd.read_parquet(OUT/f'fold{fold}_w{window}_验证.parquet')
            for method,choice in r['choices'].items():
                if choice is None:continue
                m=dict(features=choice['features'],**r['inner_models'][method]);pr=replay(m,val)
                expected=saved[saved.method==method].merge(val[KEY],on=KEY,validate='one_to_one')
                assert list(zip(expected.trade_date,expected.task_elapsed_seconds))==list(zip(val.trade_date,val.task_elapsed_seconds))
                np.testing.assert_allclose(pr,expected[['p_down','p_flat','p_up']],atol=1e-14)
                np.testing.assert_allclose(m['mean'],np.average(inner[m['features']],axis=0,weights=weights_for(inner)),atol=1e-10)
                signal=pr.argmax(1)-1;curve=[]
                for h in range(1,11):
                    hit=signal==np.sign(val[f'cum_{h}'].to_numpy())
                    curve.append(float(pd.Series(hit,index=val.trade_date).groupby(level=0).mean().mean()))
                np.testing.assert_allclose(curve,choice['accuracy_curve'],atol=1e-14)
                np.testing.assert_allclose([curve[2],np.mean(curve[4:])],[choice['accuracy3'],choice['accuracy5to10']],atol=1e-14)
                validation_rows+=len(val)
            if fold=='final':continue
            rowsets=[]
            for m in checkpoint['models']:
                assert max(m['train_dates'])<min(m['test_dates'])
                fs=m['features'];assert set(BASE).issubset(fs)
                expected_rows=a[(a.window==window)&a.trade_date.isin(m['test_dates'])].dropna(subset=r['pool']+[f'cum_{h}' for h in range(1,11)])
                saved_test=pred[(pred.fold.astype(str)==fold)&(pred.window==window)&(pred.method==m['method'])]
                z=saved_test.merge(a[KEY+['window']+fs],on=KEY+['window'],validate='one_to_one')
                assert set(map(tuple,z[KEY].to_numpy()))==set(map(tuple,expected_rows[KEY].to_numpy()))
                rowsets.append(set(map(tuple,z[KEY].to_numpy())))
                np.testing.assert_allclose(replay(m,z),z[['p_down','p_flat','p_up']],atol=1e-14)
                w=weights_for(clean);mean=np.average(clean[fs].to_numpy(),axis=0,weights=w)
                std=np.sqrt(np.average((clean[fs].to_numpy()-mean)**2,axis=0,weights=w))
                np.testing.assert_allclose(mean,m['mean'],atol=1e-10);np.testing.assert_allclose(std,m['scale'],atol=1e-10)
                vif,_,_,rank,_=diagnose(clean[fs].to_numpy(),w)
                assert rank==len(fs) and max(vif)<=5+1e-8;maxvif=max(maxvif,float(max(vif)))
                prediction_rows+=len(z)
            assert all(s==rowsets[0] for s in rowsets)
        chosen=choose(window_choices);actual=next(d['choice'] for d in decisions if d['fold']==fold)
        assert (chosen['window'],chosen['candidate_id'],chosen['C'])==(actual['window'],actual['candidate_id'],actual['C'])
        all_choices[fold]=chosen['window']
        if fold!='final':
            w=chosen['window']
            for method in ['selected','baseline']:
                q=pred[(pred.fold.astype(str)==fold)&(pred.method=='adaptive_'+method)]
                b=pred[(pred.fold.astype(str)==fold)&(pred.method==method)&(pred.window==w)]
                assert q.window.eq(w).all();np.testing.assert_allclose(q[['p_down','p_flat','p_up']],b[['p_down','p_flat','p_up']],atol=0,rtol=0)
    summary=pd.read_csv(OUT/'全部期限结果.csv');daily=pd.read_csv(OUT/'逐日评价.csv')
    for r in summary.to_dict('records'):
        sub=daily[(daily['sample']==r['sample'])&(daily.model==r['model'])&(daily.kind==r['kind'])&(daily.horizon==r['horizon'])]
        assert int(sub.n.sum())==r['n'] and sub.trade_date.nunique()==r['days']
        np.testing.assert_allclose(sub.accuracy.mean(),r['accuracy'],atol=1e-14)
    common=pd.read_parquet(OUT/'共同任务预测.parquet')
    final=json.loads((OUT/'下一批行情配置.json').read_text(encoding='utf-8'))
    fr=json.loads((OUT/f'foldfinal_w{final["window"]}.json').read_text(encoding='utf-8'))['selection']
    ft=a[(a.window==final['window'])&a.trade_date.isin(final['train_dates'])].dropna(subset=fr['pool']+['cum_3'])
    for method,m in final['models'].items():
        assert set(BASE).issubset(m['features'])
        assert m['features']==fr['choices'][method]['features'] and m['C']==fr['choices'][method]['C']
        np.testing.assert_allclose(m['mean'],np.average(ft[m['features']],axis=0,weights=weights_for(ft)),atol=1e-10)
        vif,_,_,rank,_=diagnose(ft[m['features']].to_numpy(),weights_for(ft))
        assert rank==len(m['features']) and max(vif)<=5+1e-8
        assert np.isfinite(replay(m,ft)).all()
    # Independently verify the full common-sample curves from probabilities/labels.
    curves=0
    for model,g in common.groupby('model'):
        signal=g[['p_down','p_flat','p_up']].to_numpy().argmax(1)-1
        for kind in ['cum','inc']:
            for h in range(1,11):
                target=g[f'{kind}_{h}'].to_numpy();valid=np.isfinite(target)
                val=pd.Series(signal[valid]==np.sign(target[valid]),index=g.trade_date.to_numpy()[valid]).groupby(level=0).mean().mean()
                r=summary[(summary['sample']=='common')&(summary.model==model)&(summary.kind==kind)&(summary.horizon==h)].iloc[0]
                np.testing.assert_allclose(val,r.accuracy,atol=1e-14);curves+=1
    proof=dict(rows=len(a),dates=38,label_values_checked=len(a)*len(labels),choices_recomputed=selected_checks,
        validation_predictions_replayed=validation_rows,test_predictions_replayed=prediction_rows,models=len(all_models),
        maximum_refit_vif=maxvif,window_choices=all_choices,summary_rows=len(summary),common_curves_recomputed=curves,
        old_files_unchanged=True,chronological_selection=True,fixed_core_in_all_candidates=True,same_task_cohorts=True,**counts)
    proof['final_config_checked']=True
    proof['artifact_sha256']={name:sha(OUT/name) for name in ['固定三因子原子表.parquet','逐任务预测.parquet','模型参数.json','全部期限结果.csv','窗口选择.json']}
    proof['verification_source_sha256']=sha(Path(__file__))
    save_json(OUT/'独立复核.json',proof);print(json.dumps(proof,ensure_ascii=False))

if __name__=='__main__':run()
