"""Independent replay of saved coefficients, samples, losses and timing boundaries."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.cache/python-packages'));sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from utils.opening_prediction import factors, objective, weights
OUT=ROOT/'result/opening_prediction'

def run():
    frozen=json.loads((OUT/'实验口径.json').read_text(encoding='utf-8'))
    fits=json.loads((OUT/'各折模型参数.json').read_text(encoding='utf-8'))
    pred=pd.read_parquet(OUT/'逐任务预测.parquet')
    keys=pd.read_parquet(OUT/'评价样本清单.parquet')
    atomic=pd.read_parquet(ROOT/'sample_snapshot_原子执行总表.parquet')
    atomic.trade_date=atomic.trade_date.astype(str)
    assert hashlib.sha256((ROOT/'sample_snapshot_原子执行总表.parquet').read_bytes()).hexdigest()==frozen['input_sha256']
    fit_count=0;row_count=0;max_gradient=0.
    for fit in fits:
        if fit['model']=='selected_single':
            selected=pred[(pred.model=='selected_single')&(pred.fold==fit['fold'])&(pred.horizon==fit['horizon'])].sort_values(['trade_date','task_time'])
            original=pred[(pred.model==fit['selected_model'])&(pred.fold==fit['fold'])&(pred.horizon==fit['horizon'])].sort_values(['trade_date','task_time'])
            np.testing.assert_array_equal(selected[['p_down','p_flat','p_up']],original[['p_down','p_flat','p_up']])
            continue
        w=fit['w'];h=fit['horizon'];fs=fit['features'];label=f'y_decision_{h}s'
        train=atomic[(atomic.observation_seconds==w)&atomic.trade_date.isin(fit['train_dates'])].dropna(subset=factors(w)+[label])
        assert train.trade_date.max()<min(fit['test_dates'])
        assert len(train)==fit['train_rows']
        wts=weights(train.trade_date.to_numpy())
        mean=np.average(train[fs].to_numpy(),axis=0,weights=wts)
        np.testing.assert_allclose(mean,fit['mean'],atol=1e-12)
        theta=np.column_stack([fit['coefficients'],fit['intercept']])
        design=np.column_stack([(train[fs].to_numpy()-mean)/fit['scale'],np.ones(len(train))])
        _,grad=objective(theta,design,np.sign(train[label]).astype(int).to_numpy()+1,wts/wts.sum(),1/(fit['C']*wts.sum()))
        max_gradient=max(max_gradient,float(np.abs(grad).max()));assert np.abs(grad).max()<1e-8
        assert fit['C']==min(fit['trials'],key=lambda x:x['validation_logloss'])['C']
        out=pred[(pred.model==fit['model'])&(pred.fold==fit['fold'])&(pred.horizon==h)]
        matched=out.merge(atomic[atomic.observation_seconds==w][['trade_date','task_time','feature_time',label]+fs],on=['trade_date','task_time'],validate='one_to_one')
        assert (matched.feature_time<=matched.decision_time).all()
        np.testing.assert_array_equal(matched.y,np.sign(matched[label]))
        z=(matched[fs].to_numpy()-fit['mean'])/fit['scale']
        logits=z@np.asarray(fit['coefficients']).T+fit['intercept']
        p=np.exp(logits-logits.max(1)[:,None]);p/=p.sum(1)[:,None]
        np.testing.assert_allclose(p,matched[['p_down','p_flat','p_up']],atol=1e-14)
        fit_count+=1;row_count+=len(out)
    summary=pd.read_csv(OUT/'模型评价.csv');verified=0
    for r in summary.to_dict('records'):
        subset=keys[(keys['sample']==r['sample'])&(keys.model==r['model'])&(keys.horizon==r['horizon'])]
        g=pred.merge(subset.drop(columns='sample'),on=['trade_date','task_time','model','horizon'],validate='one_to_one')
        p=g[['p_down','p_flat','p_up']].to_numpy();y=g.y.to_numpy(int)
        g=g.assign(hit=p.argmax(1)-1==y,loss=-np.log(p[np.arange(len(g)),y+1]))
        assert len(g)==r['n']
        np.testing.assert_allclose(g.groupby('trade_date').hit.mean().mean(),r['accuracy'],atol=1e-14)
        np.testing.assert_allclose(g.groupby('trade_date').loss.mean().mean(),r['logloss'],atol=1e-14)
        verified+=1
    result=dict(models_replayed=fit_count,prediction_rows_replayed=row_count,summary_rows_verified=verified,
                max_fitted_gradient=max_gradient,input_unchanged=True,
                chronological_splits=True,feature_times_not_future=True,labels_match_atomic=True,
                artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.suffix in ['.csv','.parquet','.html','.md']})
    (OUT/'独立复核.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},ensure_ascii=False))

if __name__=='__main__':run()
