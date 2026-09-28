"""Select 1–5 second observation windows for a fixed one-second price forecast."""
from pathlib import Path
import os, sys, json, hashlib
for name in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(name, '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / '.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_ic import ICConfig, SessionData
from utils.opening_schedule import ScheduleConfig
from utils.opening_task_window import bounded_features
from utils.opening_prediction import folds, fit_model, score, class_prior
from utils.opening_anchored import pack, predict
from utils.opening_small_combinations import design_stats
from utils.opening_observation import FEATURES, KEY, RULES, common_keys, select_window, extension_needed, interval
from scripts.audit_opening_data import RAW, EXPECTED_SHA256
from scripts.run_opening_prediction import save_json

OUT = ROOT / 'result/opening_prediction/observation_1s'
OLD = ROOT / 'result/opening_prediction/three_factor_enhancement'
PCOLS = ['p_down', 'p_flat', 'p_up']


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def build_atomic(windows, raw):
    cfg = ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,
                   observation_seconds=tuple(float(w) for w in windows)), horizons_seconds=(1, 2, 3))
    rows = []; causal = 0; label_checks = 0
    for date, frame in raw.groupby(raw.Date.astype(str), sort=True):
        if date in ['2026-07-20', '2026-07-27']:
            continue
        d = SessionData(frame, pd.Timestamp(date + ' 09:30', tz='Asia/Shanghai'), cfg)
        cache = {}
        for nominal in range(60):
            origin = int(np.searchsorted(d.times, nominal, 'left'))
            if origin >= d.n or d.times[origin] - nominal > .5 + 1e-10:
                start_status = 'NO_START_WITHIN_0.5S'; task = np.nan
            elif not all(d.valid[g][origin] for g in ['price', 'book', 'queue', 'volume']):
                start_status = 'INVALID_START'; task = np.nan
            else:
                start_status = 'ok'; task = float(d.times[origin])
            for w in windows:
                r = dict(trade_date=date, nominal_second=nominal, task_seconds=task,
                         window=w, task_offset=task - nominal, start_status=start_status,
                         decision_seconds=task + w)
                if not np.isfinite(task):
                    r.update({f: np.nan for f in FEATURES})
                    r.update(feature_status=start_status, arrival_delay_seconds=np.nan)
                    for h in [1, 2, 3]:
                        r[f'cum_{h}'] = np.nan; r[f'cum_{h}_status'] = start_status
                    rows.append(r); continue
                end = task + w
                x = bounded_features(d, task, w)
                r.update({f: x[f] for f in FEATURES})
                r.update({f: x[f] for f in ['snapshot_count', 'source_seconds', 'used_start', 'source_age', 'min_coverage']})
                missing = []
                for f in FEATURES:
                    if np.isfinite(x[f]):
                        continue
                    group = 'ofi' if f == 'A07_OFIDirection' else 'price' if f == 'C05_CountImbalance' else None
                    reason = 'INVALID_EDGE_OR_SOURCE'
                    i = d.source_index(end)
                    if group and i >= origin and d.valid[group][i]:
                        s = max(origin, int(d.segment[group][i])); span = d.times[i] - d.times[s]
                        reason = 'INSUFFICIENT_80_PERCENT_SPAN' if span / w < .8 - 1e-10 else 'ZERO_DENOMINATOR'
                    missing.append(f + ':' + reason)
                r['feature_status'] = ';'.join(missing) if missing else 'ok'
                if end not in cache:
                    cache[end] = d.labels(end)
                lab = cache[end]
                r['arrival_delay_seconds'] = lab.get('arrival_delay_seconds', np.nan)
                for h in [1, 2, 3]:
                    key = f'y_decision_{h}s'
                    r[f'cum_{h}'] = lab[key]; r[f'cum_{h}_status'] = lab[key + '_status']
                    r[f'label_error_{h}'] = lab[key + '_alignment_error_seconds']
                    if lab[key + '_status'] == 'ok':
                        i = d.source_index(end); j = int(np.searchsorted(d.times, end + h, 'left'))
                        assert d.segment['price'][j] <= i and d.valid['price'][j]
                        assert lab[key] == d.p[j] - d.p[i]
                        assert 0 <= d.times[j] - (end + h) <= .5 + 1e-10
                        label_checks += 1
                if nominal == 17:
                    base = np.array([x[f] for f in FEATURES])
                    sliced = d.frame[(d.times >= task) & (d.times <= end)].copy()
                    changed = d.frame.copy()
                    outside = (d.times < task) | (d.times > end)
                    for col in ['LastPrice', 'BidPrice1', 'AskPrice1']:
                        changed.loc[outside, col] += 20.
                    changed.loc[outside, 'BidVolume1'] += 777
                    changed.loc[outside, 'AskVolume1'] += 333
                    for check_frame in [sliced, changed]:
                        sub = SessionData(check_frame, d.open_time, cfg)
                        check = bounded_features(sub, task, w)
                        np.testing.assert_allclose(base, [check[f] for f in FEATURES], rtol=0, atol=0, equal_nan=True)
                        causal += 1
                rows.append(r)
        print(f'Built {date}, windows={windows}', flush=True)
    a = pd.DataFrame(rows)
    assert len(a) == 38 * 60 * len(windows) and a.trade_date.nunique() == 38
    assert not a.duplicated(KEY + ['window']).any()
    assert a.loc[a.used_start.notna(), 'used_start'].ge(a.loc[a.used_start.notna(), 'task_seconds']).all()
    assert a.loc[a.source_seconds.notna(), 'source_seconds'].le(a.loc[a.source_seconds.notna(), 'decision_seconds']).all()
    old = pd.read_parquet(OLD / '固定三因子原子表.parquet', columns=['trade_date', 'task_elapsed_seconds', 'window', *FEATURES, 'cum_1'])
    old = old.rename(columns={'task_elapsed_seconds': 'nominal_second'})
    matched = a[a.task_offset.eq(0)].merge(old, on=KEY + ['window'], suffixes=('_new', '_old'))
    for f in FEATURES + ['cum_1']:
        np.testing.assert_allclose(matched[f + '_new'], matched[f + '_old'], atol=0, rtol=0, equal_nan=True)
    audit = dict(rows=len(a), dates=38, windows=windows, causal_comparisons=causal,
                 independently_checked_labels=label_checks, same_clock_old_rows_exact=len(matched))
    return a, audit


def metric_daily(g, horizon=1, prefix='p'):
    out = []
    for date, d in g.groupby('trade_date', sort=True):
        d = d[d[f'cum_{horizon}'].notna()]
        if not len(d):
            continue
        y = np.sign(d[f'cum_{horizon}'].to_numpy()).astype(int)
        m, cm = score(y, d[[prefix + '_' + k for k in ['down', 'flat', 'up']]].to_numpy())
        m.update(trade_date=date, down_share=float(np.mean(y == -1)), up_share=float(np.mean(y == 1)),
                 all_three_classes=bool(np.all(cm.sum(1) > 0)))
        out.append(m)
    return pd.DataFrame(out)


def fit_grid(a, windows, include_test=True):
    dates = sorted(a.trade_date.unique())
    keys = common_keys(a, windows)
    complete = a.merge(keys, on=KEY, validate='many_to_one')
    models = []; preds = []; decisions = []; trials = []; validations = []
    for fold, tr, te in [(str(i), tr, te) for i, (tr, te) in enumerate(folds(dates), 1)] + [('final', dates, [])]:
        inner_dates, val_dates = tr[:-5], tr[-5:]
        assert max(inner_dates) < min(val_dates)
        if te:
            assert max(tr) < min(te)
        options = []
        for w in windows:
            g = complete[complete.window.eq(w)]
            train = g[g.trade_date.isin(tr)]
            inside = train[train.trade_date.isin(inner_dates)]
            val = train[train.trade_date.isin(val_dates)]
            assert inside.trade_date.nunique() == len(inner_dates) and val.trade_date.nunique() == 5
            assert inside.groupby('trade_date').size().min() >= 15 and val.groupby('trade_date').size().min() >= 15
            choices = []
            for c in RULES['C']:
                model = fit_model(inside[FEATURES].to_numpy(), np.sign(inside.cum_1).to_numpy(int), inside.trade_date.to_numpy(), c)
                packed = pack(model); p = model.predict_proba(val[FEATURES].to_numpy())
                np.testing.assert_allclose(p, predict(packed, val[FEATURES].to_numpy()), atol=1e-14, rtol=1e-14)
                z = val.copy()
                for j, col in enumerate(PCOLS):
                    z[col] = p[:, j]
                daily = metric_daily(z)
                result = dict(fold=fold, window=w, C=c, accuracy=float(daily.accuracy.mean()),
                              daily_sd=float(daily.accuracy.std(ddof=1)), logloss=float(daily.logloss.mean()),
                              balanced_accuracy=float(daily.balanced_accuracy.mean()), rows=len(val))
                choices.append((result, model, z)); trials.append(result)
            choice, inner_model, valp = min(choices, key=lambda z: (z[0]['logloss'], z[0]['C']))
            options.append(choice)
            vd = metric_daily(valp).assign(fold=fold, window=w, C=choice['C'])
            validations.append(vd)
            # Hold feature definitions and sample keys fixed; no factor reselection.
            refit = fit_model(train[FEATURES].to_numpy(), np.sign(train.cum_1).to_numpy(int), train.trade_date.to_numpy(), choice['C'])
            packed = pack(refit)
            prior = class_prior(np.sign(train.cum_1).to_numpy(int), train.trade_date.to_numpy())
            models.append(dict(fold=fold, window=w, C=choice['C'], features=FEATURES,
                               training_horizon=1, inner_train_dates=inner_dates, validation_dates=val_dates,
                               train_dates=tr, test_dates=te, training_rows=len(train),
                               design_stats=design_stats(train[FEATURES].to_numpy(), train.trade_date.to_numpy()),
                               majority_prior=prior.tolist(), **packed))
            if te and include_test:
                test = a[a.window.eq(w) & a.trade_date.isin(te)].dropna(subset=FEATURES + ['cum_1']).copy()
                p = refit.predict_proba(test[FEATURES].to_numpy())
                np.testing.assert_allclose(p, predict(packed, test[FEATURES].to_numpy()), atol=1e-14, rtol=1e-14)
                np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-14)
                for j, side in enumerate(['down', 'flat', 'up']):
                    test['p_' + side] = p[:, j]; test['prior_' + side] = prior[j]
                test['fold'] = fold; test['C'] = choice['C']; test['signal'] = p.argmax(axis=1) - 1
                test = test.merge(keys.assign(common=True), on=KEY, how='left', validate='many_to_one')
                test['common'] = test['common'].eq(True)
                preds.append(test)
            print(f'Fit fold={fold}, W={w}, C={choice["C"]}, validation_n={len(val)}', flush=True)
        selected = select_window(options)
        decisions.append(dict(fold=fold, train_dates=tr, test_dates=te, choice=selected, options=options))
    return dict(predictions=pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(), models=models,
                decisions=decisions, trials=trials, validations=pd.concat(validations, ignore_index=True), keys=keys)


def summarize(a, result, audit):
    p = result['predictions']; daily_all = []; summaries = []; cm_rows = []
    for sample in ['common_windows', 'native']:
        base = p[p.common] if sample == 'common_windows' else p
        for w, g in base.groupby('window'):
            for h in [1, 2, 3]:
                for method, prefix in [('model', 'p'), ('training_majority', 'prior')]:
                    daily = metric_daily(g, h, prefix)
                    daily['sample'] = sample; daily['window'] = w; daily['horizon'] = h; daily['method'] = method
                    daily_all.append(daily)
                    low, high = interval(daily.accuracy)
                    row = dict(sample=sample, window=int(w), horizon=h, method=method, n=int(daily.n.sum()),
                               days=len(daily), coverage=float(daily.n.sum() / (18 * 60)),
                               accuracy_low=low, accuracy_high=high, daily_sd=float(daily.accuracy.std(ddof=1)))
                    for metric in ['accuracy', 'balanced_accuracy', 'logloss', 'brier', 'flat_share', 'down_share', 'up_share',
                                   'down_recall', 'flat_recall', 'up_recall']:
                        row[metric] = float(daily[metric].mean())
                    summaries.append(row)
                gh = g.dropna(subset=[f'cum_{h}'])
                y = np.sign(gh[f'cum_{h}'].to_numpy()).astype(int)
                _, cm = score(y, gh[PCOLS].to_numpy())
                for i, true in enumerate([-1, 0, 1]):
                    for j, pred in enumerate([-1, 0, 1]):
                        cm_rows.append(dict(sample=sample, window=w, horizon=h, true=true, predicted=pred, count=int(cm[i, j])))
    daily = pd.concat(daily_all, ignore_index=True); summary = pd.DataFrame(summaries)
    main = daily[(daily['sample'] == 'common_windows') & daily.method.eq('model') & daily.horizon.eq(1)]
    paired = []
    wide = main.pivot(index='trade_date', columns='window', values='accuracy').sort_index()
    for w in wide.columns:
        delta = wide[w] - wide[3]; lo, hi = interval(delta)
        paired.append(dict(window=int(w), reference_window=3, gain=float(delta.mean()), low=lo, high=hi,
                           positive_days=int((delta > 0).sum()), zero_days=int((delta == 0).sum()), negative_days=int((delta < 0).sum())))
    adaptive = []
    fold_stats = []
    for d in result['decisions']:
        if d['fold'] == 'final':
            continue
        w = d['choice']['window']; z = main[main.trade_date.isin(d['test_dates']) & main.window.eq(w)].copy()
        z['selected_window'] = w; z['fold'] = d['fold']; adaptive.append(z)
        for ww, gd in main[main.trade_date.isin(d['test_dates'])].groupby('window'):
            fold_stats.append(dict(fold=d['fold'], window=int(ww), accuracy=float(gd.accuracy.mean()),
                                   n=int(gd.n.sum()), days=len(gd), chosen_window=w))
    adaptive = pd.concat(adaptive, ignore_index=True)
    coverage = []; exclusions = []
    for w, g in a.groupby('window'):
        valid = g.dropna(subset=FEATURES + ['cum_1'])
        test = p[p.window.eq(w)]
        coverage.append(dict(window=int(w), scheduled=len(g), all_dates_valid=len(valid),
                             test_native=len(test), test_common=int(test.common.sum()),
                             test_common_loss_vs_native=len(test) - int(test.common.sum()),
                             all_dates_coverage=len(valid) / len(g)))
        for status, count in g.feature_status.value_counts().items():
            if status != 'ok':
                exclusions.append(dict(window=int(w), category='feature', reason=status, count=int(count)))
        for status, count in g.cum_1_status.value_counts().items():
            if status != 'ok':
                exclusions.append(dict(window=int(w), category='label', reason=status, count=int(count)))
    delay = []
    for w, g in p[p.common].groupby('window'):
        x = g.arrival_delay_seconds.dropna()
        delay.append(dict(window=int(w), n=len(x), median=float(x.median()), p10=float(x.quantile(.1)),
                          p90=float(x.quantile(.9)), fraction_ge_1s=float((x >= 1 - 1e-10).mean()),
                          remaining_1s_mean=float(np.maximum(1 - x, 0).mean())))
    for name, frame in [('观察期汇总', summary), ('逐日表现', daily), ('相对3秒配对差异', pd.DataFrame(paired)),
                         ('逐折表现', pd.DataFrame(fold_stats)), ('按折选窗的后续表现', adaptive),
                         ('覆盖率', pd.DataFrame(coverage)), ('缺失原因', pd.DataFrame(exclusions)),
                         ('三分类混淆矩阵', pd.DataFrame(cm_rows)), ('执行延迟诊断', pd.DataFrame(delay)),
                         ('内部验证逐日', result['validations']), ('正则选择记录', pd.DataFrame(result['trials']))]:
        frame.to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')
    p.to_parquet(OUT / '逐任务预测.parquet', index=False)
    save_json(OUT / '模型参数.json', result['models']); save_json(OUT / '窗口选择.json', result['decisions'])
    final = next(d for d in result['decisions'] if d['fold'] == 'final')
    model = next(m for m in result['models'] if m['fold'] == 'final' and m['window'] == final['choice']['window'])
    save_json(OUT / '下一步候选配置.json', dict(choice=final['choice'], model=model,
              status='prediction candidate only; not installed in execution; final all-history fit has no independent test'))
    audit.update(common_all_date_tasks=len(result['keys']), test_dates=p.trade_date.nunique(),
                 saved_prediction_rows=len(p), paired_test_tasks=int(p[p.common].groupby(KEY).ngroups),
                 train_test_dates_disjoint=True, saved_parameters_reproduce_probabilities=True,
                 adaptive_accuracy=float(adaptive.accuracy.mean()), adaptive_ci=interval(adaptive.accuracy),
                 predicted_flat_not_dropped=True, fixed_features=FEATURES)
    return summary, daily, audit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [RAW, OLD / '固定三因子原子表.parquet', OLD / '模型参数.json',
                 ROOT / 'utils/opening_two_stage.py', ROOT / 'utils/opening_task_window.py']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    assert before[str(RAW.relative_to(ROOT))] == EXPECTED_SHA256
    freeze = dict(rules=RULES, input_sha256=before)
    path = OUT / '实验口径.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == freeze
    else:
        save_json(path, freeze)
    raw = pd.read_parquet(RAW); raw['source_row'] = np.arange(len(raw))
    windows = RULES['windows']
    a, audit = build_atomic(windows, raw)
    # Eligibility/extension decisions are based on training and inner validation only.
    result = fit_grid(a, windows)
    extension = extension_needed(result['decisions'])
    save_json(OUT / '延长窗口判定.json', extension)
    if extension['extend']:
        windows = windows + RULES['extension_windows']
        a, audit = build_atomic(windows, raw); result = fit_grid(a, windows)
    a.to_parquet(OUT / '观察期原子表.parquet', index=False)
    summary, daily, audit = summarize(a, result, audit)
    audit['protected_files_unchanged'] = all(sha(p) == before[str(p.relative_to(ROOT))] for p in protected)
    assert audit['protected_files_unchanged']
    audit['atomic_sha256'] = sha(OUT / '观察期原子表.parquet')
    save_json(OUT / '验收.json', audit)
    print(summary[(summary['sample'] == 'common_windows') & summary.method.eq('model') & summary.horizon.eq(1)].to_string(index=False), flush=True)
    print(json.dumps(dict(extension=extension, final_choice=result['decisions'][-1]['choice'], audit=audit), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
