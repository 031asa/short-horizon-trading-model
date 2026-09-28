"""Observation-only selection, keeping the forecast horizon and factors fixed."""
import numpy as np
import pandas as pd

FEATURES = ['F03_OFI_latest', 'F04_QuoteShift', 'F07_QuotePosition',
            'A07_OFIDirection', 'C05_CountImbalance']
KEY = ['trade_date', 'nominal_second']
RULES = dict(version=1, windows=[1, 2, 3, 4, 5], extension_windows=[6, 7, 8],
             training_horizon=1, diagnostic_horizons=[1, 2, 3], C=[.1, 1., 10.],
             features=FEATURES, cold_start=0, nominal_task_seconds=list(range(60)),
             task_origin='first snapshot at/after nominal second, offset <= 0.5s',
             interval='actual task origin <= all feature sources <= origin+window',
             training_scope='38 valid opening first minutes, one task each nominal second',
             windows_equal_lookback=True, minimum_coverage=.8, no_imputation=True,
             outer_train_days=[20, 25, 30, 35], outer_test_days=[5, 5, 5, 3],
             inner_validation_days=5, tune_C='minimum date-equal one-second log loss; tie smaller C',
             select_window='within 0.005 accuracy of validation maximum; shorter window; lower daily SD; lower logloss; smaller C',
             training_samples='same complete task keys across compared windows, one-second labels only',
             evaluation_samples=['common_windows', 'native'],
             no_abstention=True, flat_retained=True, train_only_standardization=True,
             extension_rule='on four inner-validation folds only: W5 within 0.005 of highest mean accuracy, W5-W4 mean >0.005, positive in >=3 folds; then add 6/7/8 and repeat on common tasks',
             bootstrap='5000 circular blocks of 5 dates, paired across windows, seed 20260928',
             status='exploratory on dates already used in earlier factor research; no execution result')


def common_keys(frame, windows, target='cum_1'):
    valid = frame[frame.window.isin(windows)].dropna(subset=FEATURES + [target])
    counts = valid.groupby(KEY).window.nunique()
    return counts[counts.eq(len(windows))].reset_index()[KEY]


def select_window(options):
    if not options:
        raise ValueError('No observation-window candidates')
    best = max(r['accuracy'] for r in options)
    near = [r for r in options if r['accuracy'] >= best - .005 - 1e-12]
    return min(near, key=lambda r: (r['window'], r['daily_sd'], r['logloss'], r['C']))


def extension_needed(decisions):
    records = [{**o, 'fold': d['fold']} for d in decisions if d['fold'] != 'final'
               for o in d['options']]
    table = pd.DataFrame(records).pivot(index='fold', columns='window', values='accuracy')
    delta = table[5] - table[4]
    means = table.mean()
    trigger = bool(means[5] >= means.max() - .005 - 1e-12
                   and delta.mean() > .005 and (delta > 0).sum() >= 3)
    return dict(extend=trigger, validation_mean_accuracy=means.to_dict(),
                mean_w5_minus_w4=float(delta.mean()), positive_folds=int((delta > 0).sum()),
                source='inner validation only; outer test labels are not consulted')


def block_indices(n, repeats=5000, seed=20260928):
    if n < 1:
        raise ValueError('No dates to resample')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(repeats, int(np.ceil(n / 5))))
    return ((starts[..., None] + np.arange(5)) % n).reshape(repeats, -1)[:, :n]


def interval(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return [np.nan, np.nan]
    return np.quantile(x[block_indices(len(x))].mean(axis=1), [.025, .975]).tolist()
