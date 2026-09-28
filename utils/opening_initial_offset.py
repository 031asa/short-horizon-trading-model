"""Initial-offset sweep with an unchanged T+3 passive limit of 19 ticks."""
import numpy as np
import pandas as pd
from .opening_two_stage import date_block_interval

OFFSETS = np.arange(61, dtype=int)
METHODS = {'early': '提前贡献最大', 'total': '完整成本最低'}
STAGES = {('limit', 'initial'): 0, ('limit', 'signal'): 1, ('market', 'deadline'): 2, ('market', 'signal'): 3}


def safe_initial_sweep(d, t, side, signal, offsets=OFFSETS):
    """Validated full path required. Return fill indexes/prices/stages for each k."""
    offsets = np.asarray(offsets, int)
    origin = d.source_index(t); p0 = d.p[origin]
    initial = p0 - side * offsets
    first = int(np.searchsorted(d.times, t, 'right')) + 1
    arrival = int(np.searchsorted(d.times, t+3, 'right')) + 1
    deadline = int(np.searchsorted(d.times, t+10, 'right')) + 1
    def hit(ids, limits):
        quote = d.a[ids, None] <= limits if side == 1 else d.b[ids, None] >= limits
        cross = d.p[ids, None] < limits if side == 1 else d.p[ids, None] > limits
        return quote | (cross & (d.dv[ids, None] > 0))
    # Old limit can fill on the replacement-arrival snapshot before replacement.
    ids = np.arange(first, arrival+1); hits = hit(ids, initial)
    old_filled = hits.any(axis=0); old_index = ids[hits.argmax(axis=0)]
    if side * signal > 0:
        indexes = np.where(old_filled, old_index, arrival)
        prices = np.where(old_filled, initial, d.a[arrival] if side == 1 else d.b[arrival])
        stages = np.where(old_filled, 0, 3)
    else:
        updated = d.p[d.source_index(t+3)] - side*19
        later_ids = np.arange(arrival, deadline+1)
        later_hits = hit(later_ids, np.array([updated]))[:, 0]
        if later_hits.any():
            later_index = later_ids[np.flatnonzero(later_hits)[0]]; later_price = updated
            later_stage = np.where(initial == updated, 0, 1)
        else:
            later_index = deadline; later_price = d.a[deadline] if side == 1 else d.b[deadline]
            later_stage = np.full(len(offsets), 2)
        indexes = np.where(old_filled, old_index, later_index)
        prices = np.where(old_filled, initial, later_price)
        stages = np.where(old_filled, 0, later_stage)
    return indexes.astype(int), prices.astype(float), stages.astype(int)


def select_offset(curve, method):
    metric = 'early_contribution_bp' if method == 'early' else 'cost_bp'
    if method not in METHODS: raise ValueError('Unknown selection method')
    g = curve[['k', metric]].dropna()
    if g.empty: raise ValueError('No finite candidates')
    optimum = g[metric].max() if method == 'early' else g[metric].min()
    near = g[(g[metric]-optimum).abs() <= 1e-12].copy()
    near['distance'] = (near.k-19).abs()
    return int(near.sort_values(['distance', 'k']).iloc[0].k)


def daily_summary(rows):
    """Start with paired directions; each date/side contributes equal weight."""
    r = rows.copy()
    r['early_fill'] = r.elapsed_seconds.le(3+1e-10)
    r['saving_vs_market_bp'] = r.market_cost_bp-r.cost_bp
    r['early_contribution_bp'] = np.where(r.early_fill, r.saving_vs_market_bp, 0.)
    r['later_contribution_bp'] = np.where(r.early_fill, 0., r.saving_vs_market_bp)
    for code, name in [(0, 'initial_limit'), (1, 'signal_limit'), (2, 'deadline_market'), (3, 'signal_market')]:
        r[name] = r.stage.eq(code)
    metrics = ['cost_bp', 'market_cost_bp', 'elapsed_seconds', 'early_fill', 'saving_vs_market_bp',
               'early_contribution_bp', 'later_contribution_bp', 'initial_limit', 'signal_limit', 'deadline_market', 'signal_market']
    outputs = []
    for minutes in [1, 60]:
        for side, label in [(0, 'both'), (1, 'buy'), (-1, 'sell')]:
            g = r[r.nominal_second.lt(minutes*60)]
            if side: g = g[g.direction.eq(side)]
            q = g.groupby(['trade_date', 'k'])[metrics].mean().reset_index()
            q['orders'] = g.groupby(['trade_date', 'k']).size().to_numpy()
            q['minutes'] = minutes; q['side'] = label
            q['early_conditional_saving_bp'] = q.early_contribution_bp/q.early_fill.replace(0, np.nan)
            outputs.append(q)
    return pd.concat(outputs, ignore_index=True)


def aggregate(daily):
    result = []
    metrics = ['cost_bp', 'market_cost_bp', 'elapsed_seconds', 'early_fill', 'saving_vs_market_bp',
               'early_contribution_bp', 'later_contribution_bp', 'initial_limit', 'signal_limit', 'deadline_market', 'signal_market']
    for (minutes, side), g in daily.groupby(['minutes', 'side']):
        costs = g.pivot(index='trade_date', columns='k', values='cost_bp').sort_index()
        for k, q in g.groupby('k'):
            q = q.sort_values('trade_date')
            row = dict(minutes=int(minutes), side=side, k=int(k), days=len(q), orders=int(q.orders.sum()),
                       tasks=int(q.orders.sum()/(2 if side == 'both' else 1)))
            row.update({name: float(q[name].mean()) for name in metrics})
            row['early_conditional_saving_bp'] = row['early_contribution_bp']/row['early_fill'] if row['early_fill'] else np.nan
            for name in ['saving_vs_market_bp', 'early_contribution_bp', 'later_contribution_bp']:
                lo, hi = date_block_interval(q[name], seed=20260928)
                row[name+'_low'] = float(lo); row[name+'_high'] = float(hi)
            diff = costs[19]-costs[k]; lo, hi = date_block_interval(diff, seed=20260928)
            row.update(saving_vs_19_bp=float(diff.mean()), saving_vs_19_low=float(lo), saving_vs_19_high=float(hi))
            result.append(row)
    return pd.DataFrame(result)


def forward_selection(daily):
    records = []; choices = []
    dates = sorted(daily.trade_date.unique())
    for minutes in [1, 60]:
        scope = daily[daily.minutes.eq(minutes)]
        for train_n in [8, 13]:
            tr, te = dates[:train_n], dates[train_n:train_n+5]
            train = scope[scope.side.eq('both') & scope.trade_date.isin(tr)].groupby('k', as_index=False)[['cost_bp', 'early_contribution_bp']].mean()
            assert max(tr) < min(te)
            for method in METHODS:
                k = select_offset(train, method)
                choices.append(dict(minutes=minutes, training_days=train_n, method=method, selected_k=k,
                                    train_end=tr[-1], test_start=te[0], test_end=te[-1]))
                for side in ['both', 'buy', 'sell']:
                    q = scope[scope.side.eq(side) & scope.trade_date.isin(te)]
                    for date, g in q.groupby('trade_date'):
                        r = g[g.k.eq(k)].iloc[0].to_dict(); base = g[g.k.eq(19)].iloc[0]
                        records.append(dict(**r, method=method, training_days=train_n,
                                            saving_vs_19_bp=float(base.cost_bp-r['cost_bp'])))
    return pd.DataFrame(choices), pd.DataFrame(records)
