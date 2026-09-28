"""Paired summaries for the frozen 19/19 execution horizon comparison."""
import numpy as np
import pandas as pd
from .opening_two_stage import date_block_interval

MODELS = ['legacy_3s', 'fixed_1s', 'fixed_3s']
STRATEGIES = ['M', *MODELS]
NAMES = {'M': '立即市价', 'legacy_3s': '原旧C：原模型预测3秒',
         'fixed_1s': '候选：五因子预测1秒', 'fixed_3s': '对照：五因子预测3秒'}
KEY = ['trade_date', 'nominal_second', 'direction']
TASK = KEY[:2]


def switch_decision(daily_savings):
    x = np.asarray(daily_savings, float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite paired daily savings are required')
    low, high = date_block_interval(x, repeats=5000, block=5, seed=20260928)
    gain = float(x.mean())
    passed = bool(gain > 0 and low > 0)
    return dict(mean_saving_bp=gain, low_bp=float(low), high_bp=float(high),
                passes=passed, selected='fixed_1s' if passed else 'legacy_3s',
                rule='first-hour both-side date-equal saving >0 and paired 95% block lower bound >0',
                days=len(x), bootstrap_repeats=5000, bootstrap_block_days=5, seed=20260928)


def summarize_orders(orders):
    if orders.duplicated(KEY + ['strategy']).any():
        raise ValueError('Duplicate strategy/order key')
    if not orders.groupby(TASK).size().eq(8).all():
        raise ValueError('Each common task must have four strategies and two sides')
    if set(orders.strategy) != set(STRATEGIES):
        raise ValueError('Unexpected strategy set')
    for _, g in orders.groupby(TASK):
        if set(zip(g.strategy, g.direction)) != {(s, d) for s in STRATEGIES for d in [-1, 1]}:
            raise ValueError('Unpaired sides or strategies')
    z = orders.copy()
    z['early_fill'] = z.elapsed_seconds.le(3 + 1e-10)
    for stage, kind, name in [('initial', 'limit', 'initial_limit'), ('signal', 'limit', 'signal_limit'),
                              ('signal', 'market', 'signal_market'), ('deadline', 'market', 'deadline_market'),
                              ('immediate', 'market', 'immediate_market')]:
        z[name] = z.fill_stage.eq(stage) & z.fill_kind.eq(kind)
    mix = ['initial_limit', 'signal_limit', 'signal_market', 'deadline_market', 'immediate_market']
    if not z[mix].sum(axis=1).eq(1).all():
        raise ValueError('Fill categories must be exhaustive and exclusive')
    metrics = ['cost_bp', 'elapsed_seconds', 'early_fill', *mix]
    rows = []
    for minutes in [1, 60]:
        q = z[z.nominal_second.lt(minutes * 60)]
        for side, label in [(0, 'both'), (1, 'buy'), (-1, 'sell')]:
            g = q if side == 0 else q[q.direction.eq(side)]
            a = g.groupby(['trade_date', 'fold', 'strategy'])[metrics].mean().reset_index()
            a['orders'] = g.groupby(['trade_date', 'fold', 'strategy']).size().to_numpy()
            a['minutes'] = minutes; a['side'] = label
            rows.append(a)
    daily = pd.concat(rows, ignore_index=True)
    summary = []; pairs = []
    for (minutes, side), g in daily.groupby(['minutes', 'side']):
        costs = g.pivot(index='trade_date', columns='strategy', values='cost_bp').sort_index()
        for strategy, q in g.groupby('strategy'):
            r = dict(minutes=minutes, side=side, strategy=strategy, days=len(q), orders=int(q.orders.sum()),
                     tasks=int(q.orders.sum() // (2 if side == 'both' else 1)))
            r.update({c: float(q[c].mean()) for c in metrics})
            for ref, label in [('M', 'market'), ('legacy_3s', 'legacy')]:
                diff = costs[ref] - costs[strategy]
                lo, hi = date_block_interval(diff, seed=20260928)
                r.update({f'saving_vs_{label}_bp': float(diff.mean()), f'low_vs_{label}_bp': float(lo),
                          f'high_vs_{label}_bp': float(hi)})
            summary.append(r)
        for ref, candidate in [('legacy_3s', 'fixed_1s'), ('legacy_3s', 'fixed_3s'), ('fixed_3s', 'fixed_1s')]:
            diff = costs[ref] - costs[candidate]; lo, hi = date_block_interval(diff, seed=20260928)
            pairs.append(dict(minutes=minutes, side=side, reference=ref, candidate=candidate,
                              gain_bp=float(diff.mean()), low_bp=float(lo), high_bp=float(hi),
                              positive_days=int((diff > 1e-12).sum()), zero_days=int((diff.abs() <= 1e-12).sum()),
                              negative_days=int((diff < -1e-12).sum())))
    p = daily[daily.minutes.eq(60) & daily.side.eq('both')].pivot(index='trade_date', columns='strategy', values='cost_bp').sort_index()
    return daily, pd.DataFrame(summary), pd.DataFrame(pairs), switch_decision(p.legacy_3s - p.fixed_1s)
