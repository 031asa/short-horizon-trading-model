"""Small, date-causal three-class prediction experiment. No execution returns."""
from __future__ import annotations
import numpy as np
import pandas as pd
from types import SimpleNamespace

CLASSES = np.array([-1, 0, 1])
SNAPSHOT = ['F03_OFI_latest', 'F04_QuoteShift', 'F07_QuotePosition']
ROLLING = ['F03_OFI', 'A06_PositionDeviation', 'B05_DepletionDifference']
NAMES = ['最新 OFI', '最新报价位移', '末价相对盘口位置', '窗口 OFI', '盘口位置偏离', '卖减买变薄差']
RULES = dict(version='opening-prediction-pilot-1', horizons=[1, 2, 3], windows=[1, 2, 3, 4, 5],
             first_train_days=20, test_block_days=5, inner_validation_days=5, C=[.1, 1., 10.],
             thresholds=[.55, .6, .65, .7, .8], target='sign(y_decision_hs)',
             label_origin='each observation end', labels=[-1, 0, 1],
             train_weights='equal dates, normalized to sample count',
             fit_samples='complete six-factor rows within each W and horizon; no imputation',
             selection='training-only chronological validation; daily equal log loss',
             status='internal diagnostic: factors already screened on all 38 dates',
             confidence_rule='max P(down),P(up) >= threshold; flat counts as wrong',
             paired_bootstrap='date clusters, 4000 resamples, seed 20260923; exploratory intervals')

def factors(w):
    return SNAPSHOT + [f'{f}_h{w}s' for f in ROLLING]

def specs():
    result = [dict(model=f'single_{i}', name=NAMES[i], w=1, features=[f]) for i, f in enumerate(factors(1))]
    result += [dict(model='snapshot', name='最新快照三因子组合', w=1, features=SNAPSHOT)]
    result += [dict(model=f'full_{w}', name=f'{w} 秒六因子组合', w=w, features=factors(w)) for w in range(1, 6)]
    return result

def folds(dates):
    dates = sorted(set(dates))
    return [(dates[:n], dates[n:n+5]) for n in range(20, len(dates), 5)]

def weights(dates):
    s = pd.Series(np.asarray(dates))
    w = 1 / s.map(s.value_counts()).to_numpy(float)
    return w * len(w) / w.sum()

def softmax(z):
    e=np.exp(z-z.max(axis=1,keepdims=True))
    return e/e.sum(axis=1,keepdims=True)

def objective(theta, design, y_index, w, regularization, hessian=False):
    """Full three-class ridge softmax; intercepts unpenalized, sum-zero gauge."""
    logits=design@theta.T
    maximum=logits.max(axis=1)
    logsum=maximum+np.log(np.exp(logits-maximum[:,None]).sum(axis=1))
    penalty=theta.copy();penalty[:,-1]=0
    loss=np.dot(w,logsum-logits[np.arange(len(logits)),y_index])+regularization*np.sum(penalty**2)/2
    p=softmax(logits)
    residual=p-np.eye(3)[y_index]
    gradient=(residual*w[:,None]).T@design+regularization*penalty
    if not hessian:return loss,gradient
    width=design.shape[1];h=np.empty((3*width,3*width))
    mask=np.eye(width);mask[-1,-1]=0
    for a in range(3):
        for b in range(3):
            v=w*p[:,a]*((1 if a==b else 0)-p[:,b])
            block=design.T@(v[:,None]*design)
            if a==b:block+=regularization*mask
            h[a*width:(a+1)*width,b*width:(b+1)*width]=block
    return loss,gradient,h

class RidgeSoftmax:
    def __getitem__(self,name):return getattr(self,name)
    def predict_proba(self,x):
        z=(np.asarray(x,float)-self.scale.mean_)/self.scale.scale_
        return softmax(z@self.model.coef_.T+self.model.intercept_)

def fit_model(x, y, dates, c):
    x=np.asarray(x,float);y=np.asarray(y,int)
    if not np.array_equal(np.unique(y),CLASSES):raise ValueError('Training set must contain all three classes')
    raw_weights=weights(dates);w=raw_weights/raw_weights.sum()
    mean=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale[scale<1e-12]=1
    design=np.column_stack([(x-mean)/scale,np.ones(len(x))])
    theta=np.zeros((3,design.shape[1]));reg=1/(c*raw_weights.sum())
    converged=False
    for iteration in range(100):
        loss,gradient,hessian=objective(theta,design,y+1,w,reg,True)
        if np.max(np.abs(gradient))<1e-8:converged=True;break
        # The intercept common shift is unidentified. Numerical damping handles
        # only that null direction; recentering leaves probabilities unchanged.
        step=np.linalg.solve(hessian+np.eye(hessian.shape[0])*1e-10,gradient.ravel()).reshape(theta.shape)
        rate=1.
        for _ in range(40):
            trial=theta-rate*step;trial-=trial.mean(axis=0,keepdims=True)
            next_loss,_=objective(trial,design,y+1,w,reg)
            if next_loss<=loss-1e-4*rate*np.sum(gradient*step):break
            rate*=.5
        else:raise ValueError('Softmax line search failed')
        theta=trial
    if not converged:raise ValueError('Softmax fit did not converge')
    model=RidgeSoftmax();model.classes_=CLASSES.copy()
    model.scale=SimpleNamespace(mean_=mean,scale_=scale)
    model.model=SimpleNamespace(coef_=theta[:,:-1],intercept_=theta[:,-1],n_iter_=np.array([iteration]),gradient_max=float(np.max(np.abs(gradient))))
    return model

def daily_loss(y, p, dates):
    losses = -np.log(np.clip(p[np.arange(len(y)), np.asarray(y, int)+1], 1e-15, 1))
    return float(pd.DataFrame({'date': dates, 'loss': losses}).groupby('date').loss.mean().mean())

def tune(train, features, target):
    ds = sorted(train.trade_date.unique())
    a = train[train.trade_date.isin(ds[:-5])]
    b = train[train.trade_date.isin(ds[-5:])]
    assert a.trade_date.max() < b.trade_date.min()
    trials = []
    for c in RULES['C']:
        model = fit_model(a[features].to_numpy(), a[target].to_numpy(), a.trade_date.to_numpy(), c)
        loss = daily_loss(b[target].to_numpy(), model.predict_proba(b[features].to_numpy()), b.trade_date.to_numpy())
        trials.append({'C': c, 'validation_logloss': loss})
    winner = min(trials, key=lambda x: x['validation_logloss'])
    return winner, trials

def class_prior(y, dates):
    # Tiny smoothing protects log loss against a class absent from a training set.
    w = weights(dates)
    counts = np.array([w[np.asarray(y) == c].sum() for c in CLASSES]) + 1e-6
    return counts / counts.sum()

def score(y, p):
    y = np.asarray(y, int)
    pred = CLASSES[np.argmax(p, axis=1)]
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (y+1, pred+1), 1)
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.full(3, np.nan), where=cm.sum(1)>0)
    precision = np.divide(np.diag(cm), cm.sum(0), out=np.full(3, np.nan), where=cm.sum(0)>0)
    onehot = np.eye(3)[y+1]
    return dict(n=len(y), accuracy=float(np.mean(pred == y)), balanced_accuracy=float(np.nanmean(recall)),
                logloss=float(-np.log(np.clip(p[np.arange(len(y)), y+1], 1e-15, 1)).mean()),
                brier=float(np.mean(np.sum((p-onehot)**2, axis=1))),
                **{f'{side}_recall': float(recall[i]) for i, side in enumerate(['down','flat','up'])},
                **{f'{side}_precision': float(precision[i]) for i, side in enumerate(['down','flat','up'])},
                flat_share=float(np.mean(y == 0))), cm

def evaluate(predictions, prefix='p'):
    pcols = [f'{prefix}_{s}' for s in ['down','flat','up']]
    daily, summaries, cms, confidence = [], [], [], []
    for (scope, model, h), g in predictions.groupby(['sample', 'model', 'horizon'], sort=False):
        local = []
        cm_total = np.zeros((3,3), int)
        for date, d in g.groupby('trade_date'):
            metrics, cm = score(d.y.to_numpy(), d[pcols].to_numpy())
            row = dict(sample=scope, model=model, horizon=int(h), trade_date=date, **metrics)
            daily.append(row); local.append(row); cm_total += cm
        frame = pd.DataFrame(local)
        row = {k: float(frame[k].mean()) for k in metrics if k != 'n'}
        summaries.append(dict(sample=scope, model=model, horizon=int(h), n=len(g), days=len(local),
                              coverage=len(g)/(18*50), **row))
        cms.append(dict(sample=scope, model=model, horizon=int(h), matrix=cm_total.tolist()))
        for threshold in RULES['thresholds']:
            probs = g[pcols].to_numpy()
            direction = np.where(probs[:,2] >= probs[:,0], 1, -1)
            keep = np.maximum(probs[:,0], probs[:,2]) >= threshold
            z = g.assign(signal=direction, keep=keep, hit=direction == g.y.to_numpy())
            selected = z[z.keep]
            day_acc = selected.groupby('trade_date').hit.mean()
            confidence.append(dict(sample=scope, model=model, horizon=int(h), threshold=threshold,
                 signals=int(keep.sum()), coverage=float(keep.mean()), days=len(day_acc),
                 accuracy=float(day_acc.mean()) if len(day_acc) else np.nan,
                 pooled_accuracy=float(selected.hit.mean()) if len(selected) else np.nan,
                 flat_cases=int((selected.y==0).sum()), up_signals=int((selected.signal==1).sum()),
                 down_signals=int((selected.signal==-1).sum())))
    return pd.DataFrame(summaries), pd.DataFrame(daily), cms, pd.DataFrame(confidence)

def paired_daily(daily, model, reference, horizon, metric, scope='common_windows'):
    d = daily[(daily['sample']==scope) & (daily.horizon==horizon)]
    a = d[d.model==model].set_index('trade_date')[metric]
    b = d[d.model==reference].set_index('trade_date')[metric]
    joined = pd.concat([a,b], axis=1, keys=['a','b']).dropna()
    delta = (joined.a-joined.b).to_numpy()
    rng = np.random.default_rng(20260923)
    boot = delta[rng.integers(0,len(delta),size=(4000,len(delta)))].mean(1)
    lo, hi = np.quantile(boot,[.025,.975])
    return dict(model=model, reference=reference, horizon=horizon, metric=metric, sample=scope,
                difference=float(delta.mean()), ci_low=float(lo), ci_high=float(hi), days=len(delta))
