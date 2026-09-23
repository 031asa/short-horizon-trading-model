"""Train-only bounded beam search for small, low-collinearity combinations."""
import numpy as np
import pandas as pd
from .opening_prediction import fit_model,daily_loss,weights
from .opening_task_window import ALL,BASE,INTERACTIONS

RULES=dict(cold_start=0,task_seconds=list(range(60)),windows=[1,2,3,4,5,8,10],horizon=3,
    interval='task <= every feature input timestamp <= task+W; W is observation and lookback',
    minimum_span_coverage=.8,minimum_feature_coverage=.9,minimum_pool_coverage=.8,
    C=[.1,1.,10.],beam_width=2,max_features=5,max_vif=5.,max_pair_correlation=.9,
    validation_days=5,simplicity_logloss_tolerance=.002,
    outer_train_sizes=[20,25,30,35],outer_test_sizes=[5,5,5,3],
    candidate_scope='25 named main effects plus 6 fixed mechanistic interactions; no p-value selection',
    missing='never impute features; train-selected common candidate pool for search/refit; native and cross-window common evaluation',
    coverage_levels=[.2,.4,.6],phase='exploratory after previous all-date factor screening')

def design_stats(x,dates):
    x=np.asarray(x,float);w=weights(dates);w/=w.sum()
    z=x-np.average(x,axis=0,weights=w);sd=np.sqrt(np.average(z*z,axis=0,weights=w))
    if np.any(sd<1e-12):return dict(ok=False,vif=np.inf,condition=np.inf,correlation=np.nan,reason='CONSTANT')
    z=z/sd;r=(z*w[:,None]).T@z
    ev=np.linalg.eigvalsh(r)
    if ev[0]<=1e-10:return dict(ok=False,vif=np.inf,condition=np.inf,correlation=1.,reason='DEPENDENT')
    inv=np.linalg.inv(r);v=float(np.max(np.diag(inv)));pair=float(np.max(np.abs(r-np.eye(len(r)))))
    return dict(ok=v<=RULES['max_vif']+1e-9 and pair<RULES['max_pair_correlation'],vif=v,
                condition=float(np.sqrt(ev[-1]/ev[0])),correlation=pair,reason='OK' if v<=5 and pair<.9 else 'COLLINEAR')

def select_small(train):
    ds=sorted(train.trade_date.unique());inside=train[train.trade_date.isin(ds[:-5])]
    coverage={f:float(np.isfinite(inside[f]).mean()) for f in ALL}
    pool=[f for f in ALL if coverage[f]>=RULES['minimum_feature_coverage'] and inside[f].nunique()>1]
    if not all(f in pool for f in BASE):raise ValueError('Baseline lacks training support')
    rejected=[]
    for f in ALL:
        if f not in pool:rejected.append(dict(factor=f,reason='LOW_COVERAGE_OR_CONSTANT',coverage=coverage[f]))
    def valid_hierarchy(pool):
        return [f for f in pool if f not in INTERACTIONS or all(p in pool for p in INTERACTIONS[f])]
    pool=valid_hierarchy(pool)
    while len(inside.dropna(subset=pool))/len(inside)<RULES['minimum_pool_coverage']:
        f=min([f for f in pool if f not in BASE],key=lambda f:coverage[f])
        pool.remove(f);rejected.append(dict(factor=f,reason='POOL_COMMON_COVERAGE',coverage=coverage[f]));pool=valid_hierarchy(pool)
    clean=train.dropna(subset=pool).copy()
    a=clean[clean.trade_date.isin(ds[:-5])];b=clean[clean.trade_date.isin(ds[-5:])]
    if min(a.groupby('trade_date').size())<15 or min(b.groupby('trade_date').size())<15:raise ValueError('Insufficient per-date training/validation support')
    x=a[pool].to_numpy();xval=b[pool].to_numpy();names={f:i for i,f in enumerate(pool)}
    y=a.y.to_numpy(int);dates=a.trade_date.to_numpy();cache={}
    def assess(fs):
        fs=tuple(sorted(set(fs)))
        if fs in cache:return cache[fs]
        if len(fs)>5 or not all(f in pool for f in fs):return None
        if any(f in INTERACTIONS and not set(INTERACTIONS[f]).issubset(fs) for f in fs):return None
        ix=[names[f] for f in fs];stats=design_stats(x[:,ix],dates)
        if not stats['ok']:
            cache[fs]=dict(features=list(fs),loss=np.inf,C=np.nan,**stats);return cache[fs]
        candidates=[]
        for c in RULES['C']:
            m=fit_model(x[:,ix],y,dates,c)
            loss=daily_loss(b.y.to_numpy(int),m.predict_proba(xval[:,ix]),b.trade_date.to_numpy())
            candidates.append((loss,c))
        loss,c=min(candidates)
        cache[fs]=dict(features=list(fs),loss=float(loss),C=c,**stats)
        return cache[fs]
    # Seeds include weak-factor groups and hierarchical interactions, even when
    # their individual parent does not enter the top single-feature beam.
    seeds=[BASE,
      ['A07_OFIDirection','D06_PreQIVolume'],['A03_QIMean','F06_PathEfficiency'],
      ['B05_DepletionDifference','F08_SignedVolume'],['D07_UnchangedPriceOFIRate','F07_QuotePosition']]
    seeds += [[f,*parents] for f,parents in INTERACTIONS.items()]
    for f in pool:
        if f not in INTERACTIONS:assess([f])
    for fs in seeds:assess(fs)
    for size in range(1,5):
        front=sorted([r for fs,r in cache.items() if len(fs)==size and np.isfinite(r['loss'])],key=lambda r:r['loss'])[:RULES['beam_width']]
        # Always test additions to the three-factor baseline, independent of its rank.
        if size==3:
            baseline=assess(BASE)
            if baseline and np.isfinite(baseline['loss']):front.append(baseline)
        for row in front:
            for f in pool:
                if f not in row['features']:assess(row['features']+[f])
    ranked=sorted([r for r in cache.values() if np.isfinite(r['loss'])],key=lambda r:r['loss'])
    # Refit-design check uses outer training only; an unseen test never decides eligibility.
    accepted=[]
    for row in ranked:
        fullstats=design_stats(clean[row['features']].to_numpy(),clean.trade_date.to_numpy())
        if fullstats['ok']:accepted.append({**row,'refit_stats':fullstats})
    best=min(r['loss'] for r in accepted)
    near=[r for r in accepted if r['loss']<=best+RULES['simplicity_logloss_tolerance']]
    chosen=min(near,key=lambda r:(len(r['features']),r['loss']))
    baseline=assess(BASE)
    info=dict(pool=pool,rejected=rejected,candidates_evaluated=len(cache),valid_candidates=len(ranked),
              fit_rows=len(a),validation_rows=len(b),refit_rows=len(clean),
              train_days=ds[:-5],validation_dates=ds[-5:],selected=chosen,baseline=baseline,
              top_candidates=ranked[:10])
    return chosen,baseline,clean,b,info
