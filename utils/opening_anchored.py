"""Fixed three-factor core, exhaustive bounded additions, chronological selection."""
from itertools import combinations
import numpy as np
import pandas as pd
from .opening_task_window import ALL, BASE, CATALOG, INTERACTIONS
from .opening_prediction import fit_model, weights, daily_loss, softmax
from .opening_small_combinations import design_stats

RULES = dict(version=1, windows=list(range(1, 11)), task_seconds=list(range(60)),
    cold_start=0, training_horizon=3, evaluation_horizons=list(range(1, 11)),
    fixed_core=BASE, C=[.1, 1., 10.], maximum_columns=6,
    minimum_span_coverage=.8, minimum_feature_coverage=.9, minimum_pool_coverage=.8,
    max_vif=5., max_pair_correlation=.9, accuracy_tolerance=.005,
    persistence_horizons=list(range(5, 11)), validation_days=5,
    selection='accuracy3 within .005; accuracy5to10 within .005; shorter W; fewer columns; lower loss3; smaller C; lexical ID',
    signal='one frozen three-class argmax for every horizon, no abstention',
    cohorts='common eligible feature pool within W, common validation tasks across W; all ten cumulative labels valid for selection',
    outer_train_sizes=[20,25,30,35], outer_test_sizes=[5,5,5,3],
    phase='exploratory: all dates previously used for factor research')

def prepare(train):
    """Eligibility uses inner training inputs only; no validation/test outcomes."""
    dates=sorted(train.trade_date.unique()); inner_dates=dates[:-5]; validation_dates=dates[-5:]
    inside=train[train.trade_date.isin(inner_dates)]
    coverage={f:float(np.isfinite(inside[f]).mean()) for f in ALL}
    pool=[f for f in ALL if coverage[f]>=.9 and inside[f].nunique()>1]
    if not set(BASE).issubset(pool):raise ValueError('Fixed core lacks inner-training support')
    rejected=[dict(feature=f,reason='LOW_COVERAGE_OR_CONSTANT',coverage=coverage[f]) for f in ALL if f not in pool]
    def hierarchy(p):
        removed=[f for f in p if f in INTERACTIONS and not set(INTERACTIONS[f]).issubset(p)]
        rejected.extend(dict(feature=f,reason='PARENT_UNAVAILABLE') for f in removed)
        return [f for f in p if f not in removed]
    pool=hierarchy(pool)
    while len(inside.dropna(subset=pool))/len(inside)<.8:
        options=[f for f in pool if f not in BASE]
        if not options:raise ValueError('Fixed core common coverage below 80%')
        f=min(options,key=lambda f:(coverage[f],f));pool.remove(f)
        rejected.append(dict(feature=f,reason='POOL_COMMON_COVERAGE',coverage=coverage[f]));pool=hierarchy(pool)
    clean=train.dropna(subset=pool+['cum_3']).copy()
    info=dict(pool=pool,coverage=coverage,rejected=rejected,train_dates=dates,
        inner_dates=inner_dates,validation_dates=validation_dates,
        fit_rows=int(clean.trade_date.isin(inner_dates).sum()),refit_rows=len(clean))
    return clean,info

def candidates(pool):
    mains=[r[0] for r in CATALOG if r[0] in pool and r[0] not in BASE]
    out=[dict(structure='baseline',features=sorted(BASE))]
    for n in [1,2]:
        out.extend(dict(structure='add'+str(n),features=sorted(BASE+list(fs))) for fs in combinations(mains,n))
    for f,parents in INTERACTIONS.items():
        fs=sorted(set(BASE+[f]+list(parents)))
        if set(fs).issubset(pool):out.append(dict(structure='composite',features=fs))
    for r in out:
        assert set(BASE).issubset(r['features']) and len(r['features'])<=6
        r['candidate_id']='|'.join(r['features'])
    return out

def pick(rows):
    if not rows:return None
    highest=max(r['accuracy3'] for r in rows)
    near=[r for r in rows if r['accuracy3']>=highest-.005-1e-12]
    longest=max(r['accuracy5to10'] for r in near)
    near=[r for r in near if r['accuracy5to10']>=longest-.005-1e-12]
    return min(near,key=lambda r:(r['window'],len(r['features']),r['logloss3'],r['C'],r['candidate_id']))

def pack(model):
    return dict(mean=model.scale.mean_.tolist(),scale=model.scale.scale_.tolist(),
        coefficients=model.model.coef_.tolist(),intercept=model.model.intercept_.tolist(),
        iterations=int(model.model.n_iter_[0]),gradient_max=float(model.model.gradient_max))

def predict(model,x):
    z=(np.asarray(x,float)-np.asarray(model['mean']))/np.asarray(model['scale'])
    return softmax(z@np.asarray(model['coefficients']).T+np.asarray(model['intercept']))

def validation_metrics(y,p,dates):
    signal=p.argmax(1)-1
    # The ten columns share the same validation tasks and dates.
    hit=signal[:,None]==np.sign(y)
    acc=pd.DataFrame(hit).groupby(np.asarray(dates)).mean().mean().to_numpy()
    return dict(accuracy3=float(acc[2]),accuracy5to10=float(acc[4:10].mean()),
        logloss3=daily_loss(np.sign(y[:,2]).astype(int),p,dates),accuracy_curve=acc.tolist())

def explained_by_core(frame,features):
    """Input-only weighted auxiliary regression, never residualize model inputs."""
    w=weights(frame.trade_date.to_numpy());x=frame[BASE].to_numpy()
    design=np.column_stack([np.ones(len(x)),x]);root=np.sqrt(w)
    out={}
    for f in features:
        if f in BASE:continue
        y=frame[f].to_numpy();b=np.linalg.lstsq(design*root[:,None],y*root,rcond=None)[0]
        total=np.dot(w,(y-np.average(y,weights=w))**2)
        out[f]=float(1-np.dot(w,(y-design@b)**2)/total) if total>0 else None
    return out

def search(clean,info,validation_keys,window):
    inner=clean[clean.trade_date.isin(info['inner_dates'])]
    val=clean.merge(validation_keys,on=['trade_date','task_elapsed_seconds'],validate='one_to_one')
    assert min(inner.groupby('trade_date').size())>=15 and min(val.groupby('trade_date').size())>=15
    assert max(inner.trade_date)<min(val.trade_date)
    y=np.sign(inner.cum_3.to_numpy()).astype(int); dates=inner.trade_date.to_numpy()
    outcomes=val[[f'cum_{h}' for h in range(1,11)]].to_numpy();assert np.isfinite(outcomes).all()
    records=[];fits={};models={};eligible=0
    for spec in candidates(info['pool']):
        fs=spec['features'];stats=design_stats(inner[fs].to_numpy(),dates)
        refit=design_stats(clean[fs].to_numpy(),clean.trade_date.to_numpy())
        record=dict(**spec,inner_stats=stats,refit_stats=refit,core_r2=explained_by_core(inner,fs),trials=[])
        if not stats['ok'] or not refit['ok']:
            record['status']='REJECTED_COLLINEAR';records.append(record);continue
        eligible+=1
        for c in RULES['C']:
            try:m=fit_model(inner[fs].to_numpy(),y,dates,c)
            except ValueError as error:
                record['trials'].append(dict(C=c,status='FIT_FAILED',error=str(error)));continue
            p=m.predict_proba(val[fs].to_numpy());metrics=validation_metrics(outcomes,p,val.trade_date.to_numpy())
            trial=dict(**spec,window=window,C=c,**metrics)
            record['trials'].append(dict(C=c,status='OK',**metrics))
            fits[(spec['candidate_id'],c)]=trial;models[(spec['candidate_id'],c)]=pack(m)
        record['status']='OK' if any(t['status']=='OK' for t in record['trials']) else 'FIT_FAILED'
        records.append(record)
    choices={kind:pick([r for r in fits.values() if r['structure']==kind]) for kind in ['baseline','add1','add2','composite']}
    if choices['baseline'] is None:raise ValueError('No valid fixed-core baseline')
    choices['selected']=pick([r for r in choices.values() if r is not None])
    inner_models={k:models[(v['candidate_id'],v['C'])] for k,v in choices.items() if v is not None}
    info=dict(**info,window=window,validation_rows=len(val),candidate_count=len(records),eligible_candidates=eligible,
        candidates=records,choices=choices,inner_models=inner_models)
    return info,val
