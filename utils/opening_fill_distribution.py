"""Weighted distribution diagnostics; censored waits are never observed fills."""
from statistics import NormalDist
import numpy as np
import pandas as pd

NORMAL=NormalDist()


def weighted_quantile(x,w,p):
    x=np.asarray(x,float);w=np.asarray(w,float);p=np.asarray(p,float)
    use=np.isfinite(x)&np.isfinite(w)&(w>0)
    if not use.any():return np.full(p.shape,np.nan)
    x=x[use];w=w[use];idx=np.argsort(x,kind='stable');x=x[idx];w=w[idx]
    c=np.cumsum(w)/w.sum()
    return x[np.minimum(np.searchsorted(c,p,'left'),len(x)-1)]


def normal_diagnostics(x,w):
    """Descriptive normal fit, not an IID normality-test p value."""
    x=np.asarray(x,float);w=np.asarray(w,float);ok=np.isfinite(x)&np.isfinite(w)&(w>0)
    x=x[ok];w=w[ok]/w[ok].sum()
    mu=float(w@x);center=x-mu;var=float(w@(center**2));sd=var**.5
    if sd<=0:raise ValueError('Need nonconstant observations')
    skew=float(w@(center**3)/sd**3);kurt=float(w@(center**4)/sd**4-3)
    frame=pd.DataFrame(dict(x=x,w=w)).groupby('x',sort=True).w.sum()
    fitted=np.array([NORMAL.cdf((v-mu)/sd) for v in frame.index])
    empirical=frame.cumsum().to_numpy();before=empirical-frame.to_numpy()
    distance=float(max(np.max(abs(empirical-fitted)),np.max(abs(before-fitted))))
    qs=weighted_quantile(x,w,[.1,.25,.5,.75,.9,.95,.99])
    return dict(n=len(x),mean=mu,sd=sd,skew=skew,excess_kurtosis=kurt,normal_cdf_gap=distance,
                gaussian_negative_mass=NORMAL.cdf(-mu/sd),minimum=float(x.min()),maximum=float(x.max()),
                **dict(zip(['q10','q25','q50','q75','q90','q95','q99'],map(float,qs))))


def block_moment_intervals(frame,value,weight='weight',repeats=5000,dates=None):
    """Date blocks preserve the within-day dependence and cohort weighting."""
    raw={}
    for date,g in frame.groupby('trade_date',sort=True):
        x=g[value].to_numpy(float);w=g[weight].to_numpy(float)
        raw[date]=[w.sum(),*[float(w@(x**j)) for j in range(1,5)]]
    dates=sorted(raw) if dates is None else sorted(dates)
    a=np.asarray([raw.get(d,[0.]*5) for d in dates]);n=len(a);rng=np.random.default_rng(20260928)
    starts=rng.integers(0,n,(repeats,int(np.ceil(n/5))))
    ids=((starts[...,None]+np.arange(5))%n).reshape(repeats,-1)[:,:n]
    sums=a[ids].sum(axis=1);sums=sums[sums[:,0]>0]
    mom=sums[:,1:]/sums[:,:1];mu=mom[:,0]
    var=mom[:,1]-mu**2;keep=var>1e-12;mom=mom[keep];mu=mu[keep];sd=np.sqrt(var[keep])
    skew=(mom[:,2]-3*mu*mom[:,1]+2*mu**3)/sd**3
    kurt=(mom[:,3]-4*mu*mom[:,2]+6*mu**2*mom[:,1]-3*mu**4)/sd**4-3
    return dict(bootstrap_valid_draws=len(mu),bootstrap_dates=n,
                **{key+'_'+tail:float(v) for key,values in [('mean',mu),('sd',sd),('skew',skew),('excess_kurtosis',kurt)]
                   for tail,v in zip(['low','high'],np.quantile(values,[.025,.975]))})


def weighted_km(duration,event,weight):
    """Events at a tied time occur before administrative censoring at that time."""
    f=pd.DataFrame(dict(time=duration,event=event,weight=weight))
    if len(f)==0 or f.time.lt(0).any() or f.weight.le(0).any():raise ValueError('Invalid survival data')
    f['event_weight']=f.weight*f.event.astype(bool)
    g=f.groupby('time',sort=True)[['weight','event_weight']].sum()
    risk=float(g.weight.sum());survival=1.;out=[]
    for t,r in g.iterrows():
        before=risk;survival*=max(0,1-r.event_weight/before)
        out.append(dict(time=float(t),at_risk_weight=before,event_weight=float(r.event_weight),
                        censor_weight=float(r.weight-r.event_weight),survival=survival,cdf=1-survival))
        risk-=r.weight
    return pd.DataFrame(out)
