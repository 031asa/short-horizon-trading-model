"""Two-stage unit-order execution proxy on causal L1 snapshots.

Prices are integer ticks. A clock decision is processed after any snapshot at
the same time. An active limit has priority over an arriving atomic replacement.
"""
from __future__ import annotations

import numpy as np
from .opening_ic import limit_fill_evidence

POLICIES = ('A_benchmark', 'B_observe_first', 'C_limit_first')
RULES = dict(
    stages=[[0, 3], [3, 10]], signal_seconds=3, deadline_seconds=10,
    execution_delay_snapshots=2, quantity=1, tick_size=.2,
    sides=[1, -1], task_seconds=list(range(60)),
    model='frozen W3 selected models; argmax down/flat/up; no refit',
    adverse='buy/up or sell/down => market; otherwise LastPrice limit',
    replacement='atomic; old limit remains active until new order arrives',
    priority='existing limit fill, replacement arrival, clock decision',
    fill='quote reaches or positive-volume LastPrice strictly crosses; at limit',
    market='opposite best quote at arrival, best quantity >= 1',
    deadline='T+10 submission; fill may occur after T+10; no clock reset',
    initial_source='as-of T, max age 0.5 seconds; never a future opening quote',
    gaps='all traversed snapshots valid; contiguous source rows; max gap 1s',
    cohort='intersection of all three policies and both sides per task',
    aggregation='equal dates; buy and sell each 50 percent',
    limitations='unit order; no queue, partial fills, impact or fees; L1 proxy',
)


def simulate(d, task: float, direction: int, policy: str, signal: int | None, *, c_limit_offset_ticks: int = 0,
             c_signal_action: str | None = None):
    """Simulate one independent order. Only T+3 consumes the supplied signal."""
    if direction not in (-1, 1) or policy not in POLICIES:
        raise ValueError('Invalid side/policy')
    if signal not in (-1, 0, 1, None):
        raise ValueError('Signal must be -1, 0, 1, or None')
    if c_signal_action not in (None, 'market', 'lastprice', 'passive'):
        raise ValueError('C signal action must be market, lastprice, passive, or None')
    if c_signal_action is not None and policy != 'C_limit_first':
        raise ValueError('Signal action override is only available for C')
    if isinstance(c_limit_offset_ticks, bool) or not isinstance(c_limit_offset_ticks, (int, np.integer)) or c_limit_offset_ticks < 0:
        raise ValueError('C passive offset must be a nonnegative integer number of ticks')
    offset = c_limit_offset_ticks if policy == 'C_limit_first' else 0
    trace = []
    active = None
    pending = None
    submissions = 0
    replacements = 0
    signal_used = False
    signal_action = 'not_reached'
    deadline_submitted = False
    skipped_same_price = False
    p0 = np.nan
    now = float(task)

    def event(event_name, **values):
        trace.append(dict(time=float(now), event=event_name, **values))

    def finish(status, reason='', **values):
        return dict(status=status, reason=reason, initial_ticks=p0,
                    policy=policy, direction=direction, task_seconds=task,
                    submissions=submissions, replacements=replacements,
                    signal_used=signal_used, signal_action=signal_action,
                    deadline_submitted=deadline_submitted,
                    skipped_same_price=skipped_same_price, trace=trace, **values)

    def invalid(reason):
        event('invalid', reason=reason)
        return finish('invalid', reason)

    def source(t):
        i = d.source_index(t)
        if i < 0:
            return i, 'no_source'
        if t - d.times[i] > d.config.maximum_source_age_seconds + 1e-10:
            return i, 'stale_source'
        if not all(d.valid[k][i] for k in ('price', 'book', 'queue', 'volume')):
            return i, 'invalid_source'
        return i, ''

    def submit(kind, price, stage):
        nonlocal pending, submissions, replacements
        assert pending is None
        arrival = int(np.searchsorted(d.times, now, side='right')) + d.config.execution_delay_snapshots - 1
        pending = dict(kind=kind, limit=price, stage=stage, arrival=arrival, submitted=float(now))
        submissions += 1
        replacements += int(active is not None)
        # Store index only: arrivals have no access to future prices at decision time.
        event('submit', kind=kind, limit_ticks=price, stage=stage, arrival_index=arrival,
              replaces_active=active is not None)

    def fill(price, kind, stage, mechanism, index):
        event('fill', kind=kind, stage=stage, mechanism=mechanism,
              price_ticks=float(price), snapshot_index=index,
              pending_cancelled=pending is not None)
        cost_ticks = direction * (price - p0)
        return finish('filled', fill_seconds=float(now), elapsed_seconds=float(now-task),
                      fill_ticks=float(price), cost_ticks=float(cost_ticks),
                      cost_bp=float(cost_ticks / p0 * 10000), fill_kind=kind,
                      fill_stage=stage, fill_mechanism=mechanism,
                      pending_cancelled=pending is not None)

    def match(index):
        if active is None:
            return None
        hit, mechanism = limit_fill_evidence(direction, active['limit'], d.p[index],
                                            d.b[index], d.a[index], d.dv[index])
        if hit:
            return fill(active['limit'], 'limit', active['stage'], mechanism, index)
        return None

    origin, error = source(task)
    if error:
        return invalid('initial_' + error)
    p0 = float(d.p[origin])
    event('reference', price_ticks=p0, source_index=origin,
          source_seconds=float(d.times[origin]))
    first = int(np.searchsorted(d.times, task, side='right'))
    # Includes enough snapshots for the deadline command's arrival, with no
    # synthetic fill when the file ends too early.
    last = min(d.n, int(np.searchsorted(d.times, task+10, side='right'))
               + d.config.execution_delay_snapshots)
    events = [(float(d.times[i]), 0, i) for i in range(first, last)]
    events += [(float(task+s), 1, s) for s in (0, 3, 10)]
    events.sort()
    for now, category, item in events:
        if category == 0:
            i = item
            if not all(d.valid[k][i] for k in ('price', 'book', 'queue', 'volume')):
                return invalid('invalid_path_snapshot')
            if not d.time_edge[i]:
                return invalid('gap_or_nonadjacent_snapshot')
            if not d.edge['volume'][i]:
                return invalid('volume_reset_or_invalid_edge')
            result = match(i)
            if result:
                return result
            if pending is not None and pending['arrival'] == i:
                command, pending = pending, None
                event('arrival', kind=command['kind'], limit_ticks=command['limit'],
                      stage=command['stage'], submitted=command['submitted'],
                      delay_seconds=float(now-command['submitted']),
                      old_limit_ticks=None if active is None else active['limit'])
                active = None
                if command['kind'] == 'market':
                    price, quantity = (d.a[i], d.qa[i]) if direction == 1 else (d.b[i], d.qb[i])
                    if quantity < 1:
                        return invalid('insufficient_opposite_size')
                    return fill(price, 'market', command['stage'], 'opposite_quote', i)
                active = command
                result = match(i)
                if result:
                    return result
        elif item == 0:
            if policy != 'B_observe_first':
                submit('limit', p0-direction*offset, 'initial')
        elif item == 3:
            if policy == 'A_benchmark':
                continue
            i, error = source(now)
            if error:
                return invalid('signal_' + error)
            if signal is None:
                return invalid('missing_signal')
            signal_used = True
            signal_action = 'market' if direction*signal > 0 else 'limit'
            signal_offset = offset
            if c_signal_action is not None:
                signal_action = 'market' if c_signal_action == 'market' else 'limit'
                signal_offset = 0 if c_signal_action == 'lastprice' else offset
            event('signal', signal=int(signal), action=signal_action,
                  lastprice_ticks=float(d.p[i]), source_seconds=float(d.times[i]))
            if pending is not None:
                # With the two-snapshot delay and maximum 1-second gaps the
                # initial limit must have arrived by T+3 on every valid path.
                return invalid('unexpected_pending_at_signal')
            signal_limit = float(d.p[i])-direction*signal_offset
            if signal_action == 'limit' and active is not None and active['limit'] == signal_limit:
                skipped_same_price = True
                event('keep_same_limit', limit_ticks=signal_limit)
            else:
                submit(signal_action, signal_limit if signal_action == 'limit' else None, 'signal')
        else:
            if pending is not None:
                if pending['kind'] == 'market':
                    event('deadline_already_market')
                    continue
                return invalid('unexpected_pending_at_deadline')
            i, error = source(now)
            if error:
                return invalid('deadline_' + error)
            deadline_submitted = True
            submit('market', None, 'deadline')
    return invalid('insufficient_arrival_tail')


def date_block_interval(values, repeats=5000, block=5, seed=20260923):
    """Circular moving date-block bootstrap; inputs are ordered daily means."""
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite daily means required')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), (repeats, int(np.ceil(len(values)/block))))
    index = ((starts[..., None] + np.arange(block)) % len(values)).reshape(repeats, -1)[:, :len(values)]
    return np.quantile(values[index].mean(axis=1), [.025, .975])
