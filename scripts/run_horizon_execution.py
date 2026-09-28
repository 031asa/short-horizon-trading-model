"""Only compare old C at W3 with 1s/3s forecast labels; do not scan offsets."""
from pathlib import Path
import os, sys, json, hashlib
sys.path = [p for p in sys.path if not p.replace('\\', '/').endswith('.cache/model-packages')]
for name in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(name, '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / '.cache/python-packages'))
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from utils.opening_ic import ICConfig, SessionData
from utils.opening_schedule import ScheduleConfig
from utils.opening_task_window import bounded_features
from utils.opening_prediction import fit_model, score, class_prior
from utils.opening_anchored import pack, predict
from utils.opening_observation import FEATURES, common_keys
from utils.opening_two_stage import simulate
from utils.opening_horizon import MODELS, STRATEGIES, KEY, TASK, summarize_orders
from scripts.run_second_ranges import independent_fill
from scripts.run_opening_prediction import save_json
from scripts.audit_opening_data import RAW, EXPECTED_SHA256

OUT = ROOT / 'result/opening_execution/initial_offset_selection/horizon_comparison'
OBS = ROOT / 'result/opening_prediction/observation_1s'
OLD = ROOT / 'result/opening_prediction/three_factor_enhancement/模型参数.json'
RANGES = ROOT / 'result/opening_execution/opening_ranges'
BASE = RANGES / 'four_strategy_c19/四策略逐任务配对.parquet'
SIGNALS = RANGES / '首小时逐秒信号.parquet'
PCOLS = ['p_down', 'p_flat', 'p_up']
FIELDS = ['initial_ticks', 'fill_seconds', 'fill_ticks', 'cost_ticks', 'cost_bp',
          'elapsed_seconds', 'fill_kind', 'fill_stage', 'pending_cancelled']


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare_models():
    old = [m for m in json.loads(OLD.read_text(encoding='utf-8')) if m['window'] == 3 and m['method'] == 'selected']
    one = [m for m in json.loads((OBS / '模型参数.json').read_text(encoding='utf-8')) if m['window'] == 3 and m['fold'] != 'final']
    a = pd.read_parquet(OBS / '观察期原子表.parquet')
    keys = common_keys(a, [1, 2, 3, 4, 5]); complete = a[a.window.eq(3)].merge(keys, on=TASK, validate='one_to_one')
    assert complete.cum_3.notna().all()
    saved = pd.read_parquet(OBS / '逐任务预测.parquet'); saved = saved[saved.window.eq(3)]
    models = []; checks = []
    for legacy, candidate in zip(old, one):
        assert legacy['fold'] == candidate['fold'] and legacy['test_dates'] == candidate['test_dates']
        train = complete[complete.trade_date.isin(candidate['train_dates'])]
        assert len(train) == candidate['training_rows'] and max(candidate['train_dates']) < min(candidate['test_dates'])
        x = train[FEATURES].to_numpy()
        for h in [1, 3]:
            fit = fit_model(x, np.sign(train[f'cum_{h}']).to_numpy(int), train.trade_date.to_numpy(), candidate['C'])
            params = pack(fit)
            for k in ['mean', 'scale']:
                np.testing.assert_allclose(params[k], candidate[k], atol=1e-13, rtol=1e-13)
            if h == 1:
                for k in ['coefficients', 'intercept']:
                    np.testing.assert_allclose(params[k], candidate[k], atol=1e-13, rtol=1e-13)
            else:
                control = {**candidate, **params, 'training_horizon': 3,
                           'majority_prior': class_prior(np.sign(train.cum_3).to_numpy(int), train.trade_date.to_numpy()).tolist()}
        test = saved[saved.fold.eq(candidate['fold'])]
        np.testing.assert_allclose(predict(candidate, test[FEATURES].to_numpy()), test[PCOLS], atol=1e-13, rtol=1e-13)
        for code, m in [('legacy_3s', legacy), ('fixed_1s', candidate), ('fixed_3s', control)]:
            models.append({**m, 'strategy': code})
        checks.append(dict(fold=candidate['fold'], C=candidate['C'], train_rows=len(train),
                           saved_prediction_rows=len(test), one_second_exactly_reproduced=True,
                           equal_standardization=True, all_training_three_second_labels_valid=True))
    save_json(OUT / '模型参数.json', models)
    save_json(OUT / '模型复核.json', checks)
    complete[ TASK + ['window', *FEATURES, 'cum_1', 'cum_3']].to_parquet(OUT / '配对训练任务.parquet', index=False)
    return models


def feature_reason(d, t, f):
    reasons = []
    for name in FEATURES:
        if np.isfinite(f[name]): continue
        group = 'ofi' if name == 'A07_OFIDirection' else 'price' if name == 'C05_CountImbalance' else None
        reason = 'INVALID_EDGE_OR_SOURCE'
        lo = int(np.searchsorted(d.times, t, 'left')); i = d.source_index(t + 3)
        if group and i >= lo and d.valid[group][i]:
            start = max(lo, int(d.segment[group][i])); span = d.times[i] - d.times[start]
            reason = 'INSUFFICIENT_80_PERCENT_SPAN' if span / 3 < .8 - 1e-10 else 'ZERO_DENOMINATOR'
        reasons.append(name + ':' + reason)
    return ';'.join(reasons)


def worker(job):
    date, frame, baseline, oldsign, models = job
    cfg = ICConfig(schedule=ScheduleConfig(cold_start_seconds=0, task_range_seconds=3600, observation_seconds=(3.,)), horizons_seconds=(1, 3))
    d = SessionData(frame, pd.Timestamp(date + ' 09:30', tz='Asia/Shanghai'), cfg)
    refs = baseline.set_index(['nominal_second', 'direction', 'strategy'])
    oldsign = oldsign.set_index('nominal_second')
    modelmap = {m['strategy']: m for m in models}
    assert all(max(m['train_dates']) < date and date in m['test_dates'] for m in models)
    records = []; signals = []; excluded = []; feature_rows = []; traces = []
    checked = 0; old_checked = 0; early_checked = 0; causal = 0; opening_exact = 0
    atomic = pd.read_parquet(OBS / '观察期原子表.parquet')
    atomic = atomic[atomic.trade_date.eq(date) & atomic.window.eq(3)].set_index('nominal_second')
    tasks = baseline[baseline.strategy.eq('C') & baseline.direction.eq(1)].sort_values('nominal_second')
    for count, task in enumerate(tasks.itertuples()):
        nominal = int(task.nominal_second); t = float(task.task_seconds); f = bounded_features(d, t, 3)
        i = d.source_index(t); j = int(np.searchsorted(d.times, t, 'right')) + 1
        assert t == float(oldsign.loc[nominal, 'task_seconds'])
        assert 0 <= t - nominal <= .5 + 1e-10 and d.times[i] == t
        probs = {}; sig = {}
        for code in MODELS:
            m = modelmap[code]; x = np.array([f[k] for k in m['features']])
            probs[code] = predict(m, x[None, :])[0] if np.isfinite(x).all() else np.full(3, np.nan)
            sig[code] = int(probs[code].argmax() - 1) if np.isfinite(probs[code]).all() else None
        np.testing.assert_allclose(probs['legacy_3s'], oldsign.loc[nominal, PCOLS].to_numpy(float), atol=1e-13, rtol=1e-13)
        assert sig['legacy_3s'] == int(oldsign.loc[nominal, 'signal'])
        if nominal < 60:
            np.testing.assert_allclose([f[k] for k in FEATURES], atomic.loc[nominal, FEATURES].to_numpy(float), atol=0, rtol=0, equal_nan=True)
            opening_exact += 1
        if count % 180 == 0:
            use = sorted(set(FEATURES + modelmap['legacy_3s']['features']))
            sliced = d.frame[(d.times >= t) & (d.times <= t+3)].copy()
            changed = d.frame.copy(); outside = (d.times < t) | (d.times > t+3)
            for col in ['LastPrice', 'BidPrice1', 'AskPrice1']: changed.loc[outside, col] += 20
            changed.loc[outside, 'BidVolume1'] += 777; changed.loc[outside, 'AskVolume1'] += 333
            for source in [sliced, changed]:
                check = bounded_features(SessionData(source, d.open_time, cfg), t, 3)
                np.testing.assert_allclose([f[k] for k in use], [check[k] for k in use], atol=0, rtol=0, equal_nan=True)
                causal += 1
        label = d.labels(t + 3)  # diagnostics only; never used for eligibility or actions
        feature_rows.append(dict(trade_date=date, nominal_second=nominal, task_seconds=t,
                                 **{k: f[k] for k in FEATURES}, feature_status=feature_reason(d, t, f)))
        taskrows = []; errors = []
        valid_signal = all(v is not None for v in sig.values())
        for code in MODELS:
            signals.append(dict(trade_date=date, nominal_second=nominal, task_seconds=t, fold=modelmap[code]['fold'],
                                strategy=code, signal=sig[code], **dict(zip(PCOLS, probs[code])),
                                cum_1=label['y_decision_1s'], cum_3=label['y_decision_3s'],
                                arrival_delay_seconds=label['arrival_delay_seconds']))
        for side in [1, -1]:
            assert j < d.n and all(d.valid[k][i:j+1].all() for k in ['price', 'book', 'queue', 'volume'])
            assert d.time_edge[i+1:j+1].all() and d.edge['volume'][i+1:j+1].all()
            price = float(d.a[j] if side == 1 else d.b[j]); assert (d.qa[j] if side == 1 else d.qb[j]) >= 1
            cost = side * (price-d.p[i])
            market = dict(initial_ticks=float(d.p[i]), fill_seconds=float(d.times[j]), fill_ticks=price, cost_ticks=float(cost),
                          cost_bp=float(cost/d.p[i]*10000), elapsed_seconds=float(d.times[j]-t),
                          fill_kind='market', fill_stage='immediate', pending_cancelled=False)
            for k in FIELDS: assert market[k] == refs.loc[(nominal, side, 'M'), k], (date, nominal, side, k)
            taskrows.append(dict(trade_date=date, nominal_second=nominal, direction=side, strategy='M',
                                 task_seconds=t, fold=modelmap['legacy_3s']['fold'], **market))
            results = {}
            for code in MODELS:
                if code != 'legacy_3s' and not valid_signal: continue
                r = simulate(d, t, side, 'C_limit_first', sig[code], c_limit_offset_ticks=19, c_signal_offset_ticks=19)
                trace = r.pop('trace')
                if r['status'] != 'filled':
                    errors.append(f'{code}/{side}:{r["reason"]}'); continue
                ii, pp = independent_fill(d, trace, side)
                assert r['fill_seconds'] == d.times[ii] and r['fill_ticks'] == pp; checked += 1
                if code == 'legacy_3s':
                    for k in FIELDS: assert r[k] == refs.loc[(nominal, side, 'C'), k], (date, nominal, side, k)
                    old_checked += 1
                for event in trace:
                    if event['event'] == 'submit' and event['kind'] == 'limit':
                        base = d.p[i] if event['stage'] == 'initial' else d.p[d.source_index(t+3)]
                        assert event['limit_ticks'] == base - side*19
                    if event['event'] == 'submit' and event['stage'] == 'deadline': assert event['time'] == t + 10
                if count < 2: traces.append(dict(nominal_second=nominal, direction=side, strategy=code, trace=trace))
                results[code] = r
                taskrows.append(dict(trade_date=date, nominal_second=nominal, direction=side, strategy=code,
                                     task_seconds=t, fold=modelmap[code]['fold'], signal=sig[code],
                                     signal_used=r['signal_used'], signal_action=r['signal_action'],
                                     **{k: r[k] for k in FIELDS}))
            if len(results) == 3 and any(r['elapsed_seconds'] <= 3 + 1e-10 for r in results.values()):
                for k in FIELDS: assert len({r[k] for r in results.values()}) == 1
                early_checked += 1
        if not valid_signal: errors.append('missing_features:' + feature_reason(d, t, f))
        if errors or len(taskrows) != 8:
            excluded.append(dict(trade_date=date, nominal_second=nominal, task_seconds=t, reason=';'.join(errors)))
        else: records.extend(taskrows)
    folder = OUT / '.replay'
    pd.DataFrame(records).to_parquet(folder / f'{date}_orders.parquet', index=False)
    pd.DataFrame(signals).to_parquet(folder / f'{date}_signals.parquet', index=False)
    pd.DataFrame(feature_rows).to_parquet(folder / f'{date}_features.parquet', index=False)
    save_json(folder / f'{date}_traces.json', traces)
    result = dict(trade_date=date, original_tasks=len(tasks), common_tasks=len(records)//8,
                  independently_checked_orders=checked, old_C_exact=old_checked, market_exact=len(tasks)*2,
                  early_fills_identical=early_checked, causal_checks=causal, opening_features_exact=opening_exact,
                  exclusions=excluded)
    save_json(folder / f'{date}_audit.json', result)
    return {k: v for k, v in result.items() if k != 'exclusions'}


def finish():
    files = sorted((OUT / '.replay').glob('*_audit.json'))
    audits = [json.loads(p.read_text(encoding='utf-8')) for p in files]
    assert len(audits) == 18
    orders = pd.concat([pd.read_parquet(p) for p in sorted((OUT / '.replay').glob('*_orders.parquet'))], ignore_index=True)
    signals = pd.concat([pd.read_parquet(p) for p in sorted((OUT / '.replay').glob('*_signals.parquet'))], ignore_index=True)
    features = pd.concat([pd.read_parquet(p) for p in sorted((OUT / '.replay').glob('*_features.parquet'))], ignore_index=True)
    keys = orders[TASK].drop_duplicates().assign(common=True)
    signals = signals.merge(keys, on=TASK, how='left', validate='many_to_one'); signals['common'] = signals.common.eq(True)
    features = features.merge(keys, on=TASK, how='left', validate='one_to_one'); features['common'] = features.common.eq(True)
    orders.to_parquet(OUT / '四组逐任务成交.parquet', index=False)
    signals.to_parquet(OUT / '三模型逐任务信号.parquet', index=False)
    features.to_parquet(OUT / '逐任务特征与覆盖.parquet', index=False)
    daily, summary, paired, choice = summarize_orders(orders)
    for name, df in [('逐日成本', daily), ('成本与节约汇总', summary), ('配对差异', paired)]:
        df.to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')
    fold = daily.groupby(['minutes', 'side', 'fold', 'strategy'], as_index=False).agg(
        cost_bp=('cost_bp', 'mean'), elapsed_seconds=('elapsed_seconds', 'mean'), days=('trade_date', 'size'), orders=('orders', 'sum'))
    fold.to_csv(OUT / '四折表现.csv', index=False, encoding='utf-8-sig')
    excluded = pd.DataFrame([e for a in audits for e in a['exclusions']], columns=TASK + ['task_seconds', 'reason'])
    excluded.to_csv(OUT / '排除任务.csv', index=False, encoding='utf-8-sig')
    excluded.groupby('reason').size().rename('tasks').reset_index().to_csv(OUT / '排除原因汇总.csv', index=False, encoding='utf-8-sig')
    coverage = pd.DataFrame([{k: a[k] for k in ['trade_date', 'original_tasks', 'common_tasks']} for a in audits])
    coverage['excluded_tasks'] = coverage.original_tasks - coverage.common_tasks
    coverage['coverage'] = coverage.common_tasks / coverage.original_tasks
    coverage.to_csv(OUT / '逐日覆盖.csv', index=False, encoding='utf-8-sig')
    accuracy = []
    for (code, date), g in signals[signals.common].groupby(['strategy', 'trade_date']):
        for minutes in [1, 60]:
            for h in [1, 3]:
                q = g[g.nominal_second.lt(minutes*60) & g[f'cum_{h}'].notna()]
                if len(q):
                    stats, _ = score(np.sign(q[f'cum_{h}']).to_numpy(int), q[PCOLS].to_numpy())
                    accuracy.append(dict(strategy=code, trade_date=date, minutes=minutes, horizon=h, **stats))
    pd.DataFrame(accuracy).to_csv(OUT / '预测诊断逐日.csv', index=False, encoding='utf-8-sig')
    save_json(OUT / '模型切换判定.json', {**choice, 'status': 'research recommendation only; no model replaced; no offset sweep',
              'original_tasks': int(coverage.original_tasks.sum()), 'common_tasks': len(keys),
              'coverage': len(keys)/coverage.original_tasks.sum()})
    save_json(OUT / '撮合验收.json', dict(per_day=audits, original_tasks=int(coverage.original_tasks.sum()),
        common_tasks=len(keys), paired_orders=len(orders), no_future_label_filter=True, no_confidence_filter=True,
        same_initial_and_signal_offset=19, no_offset_scan=True))
    print(summary[summary.minutes.eq(60) & summary.side.eq('both')].to_string(index=False), flush=True)
    print('SELECTION', json.dumps(choice), flush=True)


def audit_saved_outputs():
    """Recompute key results from the saved records, not plotting aggregates."""
    orders = pd.read_parquet(OUT/'四组逐任务成交.parquet')
    signals = pd.read_parquet(OUT/'三模型逐任务信号.parquet')
    features = pd.read_parquet(OUT/'逐任务特征与覆盖.parquet')
    models = json.loads((OUT/'模型参数.json').read_text(encoding='utf-8'))
    common = signals[signals.common]
    counts = common.groupby(TASK).size()
    assert counts.eq(3).all() and common[TASK].drop_duplicates().shape[0] == orders[TASK].drop_duplicates().shape[0]
    side_cost = orders.direction * (orders.fill_ticks-orders.initial_ticks)/orders.initial_ticks*10000
    np.testing.assert_allclose(side_cost, orders.cost_bp, atol=1e-12, rtol=1e-12)
    daily = orders[orders.strategy.isin(['legacy_3s', 'fixed_1s'])].groupby(['trade_date', 'strategy']).cost_bp.mean().unstack()
    from utils.opening_horizon import switch_decision
    independent = switch_decision(daily.legacy_3s-daily.fixed_1s)
    choice = json.loads((OUT/'模型切换判定.json').read_text(encoding='utf-8'))
    for k in ['mean_saving_bp', 'low_bp', 'high_bp', 'selected', 'passes']: assert independent[k] == choice[k]
    replayed = 0
    for m in models:
        assert max(m['train_dates']) < min(m['test_dates'])
        if m['strategy'] == 'legacy_3s': continue  # legacy factors checked directly against raw data in worker
        g = signals[signals.strategy.eq(m['strategy']) & signals.fold.eq(m['fold'])]
        q = g.merge(features[TASK+FEATURES], on=TASK, validate='one_to_one').dropna(subset=FEATURES)
        p = predict(m, q[FEATURES].to_numpy())
        np.testing.assert_allclose(p, q[PCOLS].to_numpy(), atol=1e-13, rtol=1e-13)
        np.testing.assert_array_equal(p.argmax(1)-1, q.signal.to_numpy(int))
        replayed += len(q)
    c = orders[orders.strategy.isin(MODELS)]
    early_keys = c.loc[c.elapsed_seconds.le(3+1e-10), KEY].drop_duplicates()
    early = c.merge(early_keys, on=KEY, validate='many_to_one')
    assert early.groupby(KEY)[['fill_ticks', 'fill_seconds']].nunique().eq(1).all().all()
    delays = common.drop_duplicates(TASK).arrival_delay_seconds
    perday = json.loads((OUT/'撮合验收.json').read_text(encoding='utf-8'))['per_day']
    save_json(OUT/'独立复核.json', dict(saved_cost_formula_exact=True, gate_recomputed=True,
        no_offset_scan=True, saved_parameter_prediction_rows=replayed, training_test_isolated=True,
        common_tasks=len(counts), independently_checked_C_orders=sum(d['independently_checked_orders'] for d in perday),
        legacy_C_exact=sum(d['old_C_exact'] for d in perday), immediate_market_exact=sum(d['market_exact'] for d in perday),
        causal_perturbation_checks=sum(d['causal_checks'] for d in perday),
        common_decision_delay_min=float(delays.min()), common_decision_delay_max=float(delays.max()),
        decision_delay_ge_one_second_ratio=float(delays.ge(1-1e-10).mean()),
        script_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ROOT/'utils/opening_horizon.py']}))


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / '.replay').mkdir(exist_ok=True)
    sources = [RAW, OLD, OBS/'模型参数.json', OBS/'观察期原子表.parquet', OBS/'逐任务预测.parquet', BASE, SIGNALS,
               ROOT/'utils/opening_two_stage.py', ROOT/'utils/opening_task_window.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    assert hashes[str(RAW.relative_to(ROOT))] == EXPECTED_SHA256
    registration = dict(input_sha256=hashes, observation_seconds=3, initial_offset=19, signal_limit_offset=19,
        deadline_seconds=10, execution_delay_snapshots=2, task_scope='18 held-forward days first hour each second; first minute secondary',
        primary='date equal, both sides half; paired common tasks', switch='gain_vs_original_C>0 and 95% block lower>0',
        control='same five features, exact same training tasks and C as saved W3/H1; only label changed to H3; C originally selected for H1',
        gate_reference='legacy_3s', bootstrap=dict(block=5, repeats=5000, seed=20260928),
        no_confidence_filter=True, no_evaluation_label_filter=True, initial_offset_sweep=False,
        status='existing-date exploration; frozen model recommendations only')
    regpath = OUT/'实验口径.json'
    if regpath.exists(): assert json.loads(regpath.read_text(encoding='utf-8')) == registration
    else: save_json(regpath, registration)
    if '--finish-only' not in sys.argv:
        models = prepare_models()
        baseline = pd.read_parquet(BASE); baseline = baseline[baseline.strategy.isin(['M', 'C'])]
        assert baseline[TASK].drop_duplicates().shape[0] == 62982
        original = baseline.groupby(['trade_date', 'strategy']).cost_bp.mean().groupby('strategy').mean()
        np.testing.assert_allclose(original[['C', 'M']], [.663587, .691888], atol=.00000051, rtol=0)
        save_json(OUT/'原底池基准.json', dict(original_cost_bp=original.to_dict(), tasks=62982, dates=18))
        oldsign = pd.read_parquet(SIGNALS)
        raw = pd.read_parquet(RAW); raw['source_row'] = np.arange(len(raw))
        jobs = []
        for date, frame in raw.groupby(raw.Date.astype(str), sort=True):
            ms = [m for m in models if date in m['test_dates']]
            if not ms: continue
            start = pd.Timestamp(date+' 09:30', tz='Asia/Shanghai')
            frame = frame[(frame.Datetime >= start) & (frame.Datetime <= start+pd.Timedelta(seconds=3620))]
            frame = frame[['Datetime', 'LastPrice', 'BidPrice1', 'AskPrice1', 'BidVolume1', 'AskVolume1', 'Volume', 'source_row']].copy()
            if not (OUT/'.replay'/f'{date}_audit.json').exists():
                jobs.append((date, frame, baseline[baseline.trade_date.eq(date)].copy(), oldsign[oldsign.trade_date.eq(date)].copy(), ms))
        with ProcessPoolExecutor(max_workers=2) as pool:
            for start in range(0, len(jobs), 4):
                for receipt in pool.map(worker, jobs[start:start+4]): print(json.dumps(receipt), flush=True)
    finish()
    audit_saved_outputs()
    assert all(sha(p) == hashes[str(p.relative_to(ROOT))] for p in sources)
    save_json(OUT/'输入保持不变.json', dict(all_verified=True, input_sha256=hashes))


if __name__ == '__main__': main()
