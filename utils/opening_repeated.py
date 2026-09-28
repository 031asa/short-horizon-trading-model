"""Repeated causal decisions with atomic replacements and queued deadline orders.

The two-stage reference engine remains unchanged. This extension permits a T+9
limit replacement still in flight when the T+10 market fallback is submitted.
"""
import numpy as np
from .opening_ic import limit_fill_evidence

STRATEGIES = {'C3': (3,), 'C36': (3, 6), 'C369': (3, 6, 9)}
NAMES = {'M': '立即市价', 'C3': '只在3秒判断', 'C36': '3、6秒判断', 'C369': '3、6、9秒判断'}


def simulate_repeated(d, task, direction, signals, decisions=(3, 6, 9), offset=19):
    if direction not in (-1, 1) or tuple(decisions) not in STRATEGIES.values():
        raise ValueError('Invalid side or decision schedule')
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)) or offset < 0:
        raise ValueError('Invalid passive offset')
    if any(signals.get(s) not in (-1, 0, 1, None) for s in decisions):
        raise ValueError('Invalid signal')
    trace = []; active = None; pending = []; now = float(task); p0 = np.nan
    used = []; missing = []; submissions = 0; replacements = 0
    deadline_submitted = False; same_price = 0

    def event(name, **kw):
        trace.append(dict(time=float(now), event=name, **kw))

    def finish(status, reason='', **kw):
        return dict(status=status, reason=reason, task_seconds=float(task), direction=direction,
                    initial_ticks=p0, submissions=submissions, replacements=replacements,
                    decisions_used=used.copy(), missing_decisions=missing.copy(), same_price_count=same_price,
                    deadline_submitted=deadline_submitted, trace=trace, **kw)

    def invalid(reason):
        event('invalid', reason=reason)
        return finish('invalid', reason)

    def source(t):
        i = d.source_index(t)
        if i < 0: return i, 'no_source'
        if t-d.times[i] > d.config.maximum_source_age_seconds+1e-10: return i, 'stale_source'
        if not all(d.valid[g][i] for g in ('price', 'book', 'queue', 'volume')): return i, 'invalid_source'
        return i, ''

    def submit(kind, limit, stage):
        nonlocal submissions, replacements
        arrival = int(np.searchsorted(d.times, now, 'right'))+d.config.execution_delay_snapshots-1
        command = dict(kind=kind, limit=limit, stage=stage, arrival=arrival, submitted=float(now))
        pending.append(command)
        submissions += 1; replacements += int(active is not None)
        event('submit', kind=kind, limit_ticks=limit, stage=stage, arrival_index=arrival,
              replaces_active=active is not None)

    def fill(price, kind, stage, mechanism, i):
        event('fill', kind=kind, stage=stage, mechanism=mechanism, price_ticks=float(price),
              snapshot_index=i, pending_cancelled=bool(pending))
        ticks = direction*(price-p0)
        return finish('filled', fill_seconds=float(now), elapsed_seconds=float(now-task),
                      fill_ticks=float(price), cost_ticks=float(ticks), cost_bp=float(ticks/p0*10000),
                      fill_kind=kind, fill_stage=stage, fill_mechanism=mechanism, pending_cancelled=bool(pending))

    def match(i):
        if active is None: return None
        hit, mechanism = limit_fill_evidence(direction, active['limit'], d.p[i], d.b[i], d.a[i], d.dv[i])
        if hit: return fill(active['limit'], 'limit', active['stage'], mechanism, i)
        return None

    origin, error = source(task)
    if error: return invalid('initial_'+error)
    p0 = float(d.p[origin])
    event('reference', price_ticks=p0, source_index=origin, source_seconds=float(d.times[origin]))
    first = int(np.searchsorted(d.times, task, 'right'))
    last = min(d.n, int(np.searchsorted(d.times, task+10, 'right'))+d.config.execution_delay_snapshots)
    events = [(float(d.times[i]), 0, i) for i in range(first, last)]
    events += [(float(task+s), 1, s) for s in (0, *decisions, 10)]
    events.sort()
    for now, category, item in events:
        if category == 0:
            i = item
            if not all(d.valid[g][i] for g in ('price', 'book', 'queue', 'volume')):
                return invalid('invalid_path_snapshot')
            if not d.time_edge[i]: return invalid('gap_or_nonadjacent_snapshot')
            if not d.edge['volume'][i]: return invalid('volume_reset_or_invalid_edge')
            result = match(i)
            if result: return result
            # Commands with a common arrival snapshot retain submission order.
            while pending and pending[0]['arrival'] == i:
                command = pending.pop(0)
                event('arrival', kind=command['kind'], limit_ticks=command['limit'], stage=command['stage'],
                      submitted=command['submitted'], delay_seconds=float(now-command['submitted']),
                      old_limit_ticks=None if active is None else active['limit'])
                active = None
                if command['kind'] == 'market':
                    price, quantity = (d.a[i], d.qa[i]) if direction == 1 else (d.b[i], d.qb[i])
                    if quantity < 1: return invalid('insufficient_opposite_size')
                    return fill(price, 'market', command['stage'], 'opposite_quote', i)
                active = command
                result = match(i)
                if result: return result
        elif item == 0:
            submit('limit', p0-direction*offset, 'initial')
        elif item in decisions:
            if pending: return invalid('unexpected_pending_at_signal')
            i, error = source(now)
            if error: return invalid('signal_'+error)
            signal = signals.get(item)
            if signal is None:
                missing.append(item)
                event('missing_signal_keep', decision_second=item)
                continue
            used.append(item)
            kind = 'market' if direction*signal > 0 else 'limit'
            limit = float(d.p[i])-direction*offset
            event('signal', decision_second=item, signal=int(signal), action=kind,
                  lastprice_ticks=float(d.p[i]), source_seconds=float(d.times[i]))
            if kind == 'limit' and active is not None and active['limit'] == limit:
                same_price += 1
                event('keep_same_limit', decision_second=item, limit_ticks=limit)
            else:
                submit(kind, limit if kind == 'limit' else None, f'signal{item}')
        else:
            if any(c['kind'] == 'market' for c in pending):
                event('deadline_already_market')
                continue
            _, error = source(now)
            if error: return invalid('deadline_'+error)
            # A T+9 limit may still be in flight. It is not discarded or made
            # effective early: enqueue fallback using the original T+10 clock.
            event('deadline_pending_limit', count=len(pending))
            deadline_submitted = True
            submit('market', None, 'deadline')
    return invalid('insufficient_arrival_tail')
