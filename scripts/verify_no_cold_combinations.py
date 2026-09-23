from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_task_window import INTERACTIONS
from utils.opening_prediction import weights
from scripts.check_factor_collinearity import diagnose
OUT=ROOT/'result/opening_prediction/no_cold_start'

def run():
    a=pd.read_parquet(OUT/'无冷启动原子表.parquet');p=pd.read_parquet(OUT/'逐任务预测.parquet')
    frozen=json.loads((OUT/'实验口径.json').read_text(encoding='utf-8'))
    models=json.loads((OUT/'模型参数.json').read_text(encoding='utf-8'));selection=json.loads((OUT/'选择过程.json').read_text(encoding='utf-8'))
    oldpath=ROOT/'sample_snapshot_原子执行总表.parquet'
    assert hashlib.sha256(oldpath.read_bytes()).hexdigest()==frozen['previous_atomic_sha256']
    assert len(a)==15960 and a.trade_date.nunique()==38 and a.task_elapsed_seconds.min()==0
    assert not a.duplicated(['trade_date','task_elapsed_seconds','window']).any()
    old=pd.read_parquet(oldpath,columns=['trade_date','task_elapsed_seconds','observation_seconds','y_decision_3s'])
    old.trade_date=old.trade_date.astype(str);old=old.rename(columns={'observation_seconds':'window'})
    matched=a.merge(old,on=['trade_date','task_elapsed_seconds','window'],validate='one_to_one')
    assert len(matched)==9500
    np.testing.assert_allclose(matched.y3,matched.y_decision_3s,atol=0,rtol=0,equal_nan=True)
    count=0;maxvif=0
    for m in models:
        fs=m['features'];assert len(fs)<=5
        for f in fs:
            if f in INTERACTIONS:assert set(INTERACTIONS[f]).issubset(fs)
        record=next(r for r in selection if r['fold']==m['fold'] and r['window']==m['window'])
        train=a[(a.window==m['window'])&a.trade_date.isin(m['train_dates'])&a.y.notna()].dropna(subset=record['pool'])
        assert max(m['train_dates'])<min(m['test_dates'])
        assert max(record['train_days'])<min(record['validation_dates'])
        assert len(train)==m['train_rows']
        mean=np.average(train[fs].to_numpy(),axis=0,weights=weights(train.trade_date.to_numpy()))
        np.testing.assert_allclose(mean,m['mean'],atol=1e-10)
        vif,_,_,rank,_=diagnose(train[fs].to_numpy(),weights(train.trade_date.to_numpy()))
        assert rank==len(fs) and vif.max()<=5+1e-8;maxvif=max(maxvif,float(vif.max()))
        pred=p[(p.fold==m['fold'])&(p.window==m['window'])&(p.method==m['method'])]
        z=pred.merge(a[['trade_date','task_elapsed_seconds','window']+fs],on=['trade_date','task_elapsed_seconds','window'],validate='one_to_one')
        xx=(z[fs].to_numpy()-m['mean'])/m['scale'];scores=xx@np.asarray(m['coefficients']).T+m['intercept']
        probability=np.exp(scores-scores.max(1)[:,None]);probability/=probability.sum(1)[:,None]
        np.testing.assert_allclose(probability,z[['p_down','p_flat','p_up']],atol=1e-14)
        assert (z.used_start>=z.task_elapsed_seconds).all() and (z.source_seconds<=z.decision_seconds).all()
        count+=len(z)
    chosen=json.loads((OUT/'训练期选择的窗口.json').read_text(encoding='utf-8'))
    for r in chosen:
        v=pd.read_parquet(OUT/f'fold{r["fold"]}_共同验证预测.parquet')
        computed={}
        for w,g in v.groupby('window'):
            loss=-np.log(g[['p_down','p_flat','p_up']].to_numpy()[np.arange(len(g)),g.y.to_numpy(int)+1])
            computed[int(w)]=float(pd.Series(loss,index=g.trade_date).groupby(level=0).mean().mean())
        best=min(computed.values());expected=min(w for w,value in computed.items() if value<=best+.002)
        assert expected==r['window']
    summary=pd.read_csv(OUT/'模型汇总.csv');common=pd.read_parquet(OUT/'共同任务预测.parquet')
    for scope,d in [('own',p),('common',common)]:
        d=d.copy();d['model_id']=np.where(d.method.str.startswith('adaptive'),d.method,d.method+'_w'+d.window.astype(str))
        for model,g in d.groupby('model_id'):
            r=summary[(summary['sample']==scope)&(summary.model==model)&(summary.segment=='all')].iloc[0]
            pr=g[['p_down','p_flat','p_up']].to_numpy();y=g.y.to_numpy(int)
            g=g.assign(hit=pr.argmax(1)-1==y,loss=-np.log(pr[np.arange(len(g)),y+1]))
            np.testing.assert_allclose([g.groupby('trade_date').hit.mean().mean(),g.groupby('trade_date').loss.mean().mean()],[r.accuracy,r.logloss],atol=1e-14)
            assert len(g)==r.n
    proof=dict(rows=len(a),dates=38,old_label_pairs_equal=9500,models_replayed=len(models),prediction_rows_replayed=count,
               max_selected_and_baseline_vif=maxvif,strict_task_bounds=True,chronological_selection=True,window_choices_recomputed=True,
               old_atomic_unchanged=True,summary_groups_verified=32,
               source_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [ROOT/'utils/opening_task_window.py',ROOT/'utils/opening_small_combinations.py',ROOT/'scripts/run_no_cold_combinations.py']})
    (OUT/'独立复核.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(proof,ensure_ascii=False))

if __name__=='__main__':run()
