"""Freeze W3/H1 signals; compare early-fill contribution and total-cost k0 choices."""
from pathlib import Path
import os, sys, json
sys.path = [p for p in sys.path if not p.replace('\\', '/').endswith('.cache/model-packages')]
for name in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']: os.environ.setdefault(name, '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from utils.opening_ic import ICConfig, SessionData
from utils.opening_schedule import ScheduleConfig
from utils.opening_two_stage import simulate, date_block_interval
from utils.opening_initial_offset import OFFSETS, STAGES, METHODS, safe_initial_sweep, daily_summary, aggregate, select_offset, forward_selection
from scripts.run_four_strategy_all_dates import full_path_safe
from scripts.run_second_ranges import independent_fill
from scripts.run_horizon_execution import OUT as SOURCE, RAW, sha, save_json

OUT = SOURCE.parent/'initial_offset_sweep'
TASK = ['trade_date', 'nominal_second']
KEY = TASK+['direction']


def worker(job):
    date, frame, baseline, signals = job
    cfg = ICConfig(schedule=ScheduleConfig(cold_start_seconds=0, task_range_seconds=3600, observation_seconds=(3.,)))
    d = SessionData(frame, pd.Timestamp(date+' 09:30', tz='Asia/Shanghai'), cfg)
    refs = baseline.set_index(['nominal_second', 'direction', 'strategy'])
    signals = signals.set_index('nominal_second')
    source = baseline[baseline.strategy.eq('fixed_1s')].sort_values(['nominal_second', 'direction'])
    blocks = []; errors = []; checked = 0; vector = 0; k19checks = 0
    safe_cache = {}
    for n, r in enumerate(source.itertuples(index=False)):
        t = float(r.task_seconds); side = int(r.direction); nominal = int(r.nominal_second)
        signal = int(signals.loc[nominal, 'signal'])
        assert signals.loc[nominal, 'task_seconds'] == t
        if t not in safe_cache: safe_cache[t] = full_path_safe(d, t)
        if safe_cache[t]:
            ix, prices, stages = safe_initial_sweep(d, t, side, signal)
            times = d.times[ix]; vector += len(OFFSETS)
        else:
            times = np.full(len(OFFSETS), np.nan); prices = times.copy(); stages = np.full(len(OFFSETS), -1)
        # All offsets on deterministic samples and unsafe paths. Every k19 result
        # is also compared below to the previously independently audited baseline.
        verify = range(len(OFFSETS)) if n % 101 == 0 or not safe_cache[t] else []
        for j in verify:
            k = int(OFFSETS[j])
            z = simulate(d, t, side, 'C_limit_first', signal, c_limit_offset_ticks=k, c_signal_offset_ticks=19)
            if z['status'] != 'filled':
                assert not np.isfinite(times[j])
                errors.append(dict(trade_date=date, nominal_second=nominal, direction=side, k=k, reason=z['reason']))
                continue
            idx, price = independent_fill(d, z['trace'], side)
            assert d.times[idx] == z['fill_seconds'] and price == z['fill_ticks']
            method = STAGES[(z['fill_kind'], z['fill_stage'])]
            if np.isfinite(times[j]):
                assert times[j] == z['fill_seconds'] and prices[j] == z['fill_ticks'] and stages[j] == method, (date, nominal, side, k)
            times[j] = z['fill_seconds']; prices[j] = z['fill_ticks']; stages[j] = method; checked += 1
            for e in z['trace']:
                if e['event'] == 'submit' and e['kind'] == 'limit':
                    expected = r.initial_ticks-side*k if e['stage'] == 'initial' else d.p[d.source_index(t+3)]-side*19
                    assert e['limit_ticks'] == expected
        assert times[19] == r.fill_seconds and prices[19] == r.fill_ticks and stages[19] == STAGES[(r.fill_kind, r.fill_stage)]
        k19checks += 1
        early = times-t <= 3+1e-10
        if np.isfinite(times).all(): assert (np.diff(early.astype(int)) <= 0).all()
        market = refs.loc[(nominal, side, 'M')]
        i = d.source_index(t); j = int(np.searchsorted(d.times, t, 'right'))+1
        market_price = d.a[j] if side == 1 else d.b[j]
        assert market.fill_ticks == market_price and market.fill_seconds == d.times[j]
        costs = side*(prices-r.initial_ticks)/r.initial_ticks*10000
        assert costs[19] == r.cost_bp
        blocks.append(pd.DataFrame(dict(nominal_second=nominal, direction=side, k=OFFSETS,
            task_seconds=t, initial_ticks=r.initial_ticks, fill_seconds=times, fill_ticks=prices,
            cost_bp=costs, elapsed_seconds=times-t, stage=stages, market_cost_bp=float(market.cost_bp))))
    rows = pd.concat(blocks, ignore_index=True); rows['trade_date'] = date
    good = rows.assign(valid=np.isfinite(rows.cost_bp)).groupby('nominal_second').valid.all()
    valid_keys = good[good].index; rows = rows[rows.nominal_second.isin(valid_keys)].copy()
    assert rows.groupby('nominal_second').size().eq(122).all()
    assert not rows.duplicated(['nominal_second','direction','k']).any()
    rows.to_parquet(OUT/'逐任务'/f'{date}.parquet', index=False)
    daily_summary(rows).to_csv(OUT/'.checks'/f'{date}_daily.csv', index=False, encoding='utf-8-sig')
    receipt = dict(trade_date=date, original_tasks=len(source)//2, common_tasks=len(valid_keys),
                   excluded_nominal_seconds=good[~good].index.tolist(), event_engine_checks=checked,
                   vector_orders=vector, k19_exact=k19checks, errors=errors)
    save_json(OUT/'.checks'/f'{date}.json', receipt)
    return {k:v for k,v in receipt.items() if k not in ['errors', 'excluded_nominal_seconds']}


def finish():
    receipts = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/'.checks').glob('*.json'))]
    assert len(receipts) == 18
    daily = pd.concat([pd.read_csv(p) for p in sorted((OUT/'.checks').glob('*_daily.csv'))], ignore_index=True)
    daily.to_csv(OUT/'逐日档位结果.csv', index=False, encoding='utf-8-sig')
    summary = aggregate(daily); summary.to_csv(OUT/'全部档位汇总.csv', index=False, encoding='utf-8-sig')
    selected = []
    for (minutes, side), g in summary.groupby(['minutes', 'side']):
        for method in METHODS:
            k = select_offset(g, method); r = g[g.k.eq(k)].iloc[0].to_dict()
            selected.append(dict(**r, method=method, at_upper_boundary=k==60, descriptive=True))
        r = g[g.k.eq(19)].iloc[0].to_dict(); selected.append(dict(**r, method='baseline19', at_upper_boundary=False, descriptive=True))
    selections = pd.DataFrame(selected); selections.to_csv(OUT/'两种方法选档对照.csv', index=False, encoding='utf-8-sig')
    choices, forward = forward_selection(daily)
    choices.to_csv(OUT/'前推选档记录.csv', index=False, encoding='utf-8-sig')
    forward.to_csv(OUT/'前推逐日结果.csv', index=False, encoding='utf-8-sig')
    forward_summary = []
    for (minutes, side, method), g in forward.groupby(['minutes','side','method']):
        g = g.sort_values('trade_date'); r = dict(minutes=int(minutes), side=side, method=method, days=len(g), orders=int(g.orders.sum()))
        for metric in ['cost_bp','saving_vs_market_bp','saving_vs_19_bp','early_contribution_bp','later_contribution_bp','early_fill']:
            r[metric] = float(g[metric].mean())
            if 'saving' in metric:
                lo,hi = date_block_interval(g[metric], seed=20260928); r[metric+'_low']=float(lo); r[metric+'_high']=float(hi)
        forward_summary.append(r)
    pd.DataFrame(forward_summary).to_csv(OUT/'前推表现汇总.csv', index=False, encoding='utf-8-sig')
    coverage = pd.DataFrame([{k:r[k] for k in ['trade_date','original_tasks','common_tasks']} for r in receipts])
    coverage['excluded_tasks'] = coverage.original_tasks-coverage.common_tasks
    coverage.to_csv(OUT/'覆盖率.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([dict(trade_date=r['trade_date'],nominal_second=n) for r in receipts for n in r['excluded_nominal_seconds']],columns=TASK).to_csv(OUT/'新增排除任务.csv',index=False,encoding='utf-8-sig')
    errors = pd.DataFrame([e for r in receipts for e in r['errors']],columns=KEY+['k','reason'])
    errors.to_csv(OUT/'不可完成回放明细.csv',index=False,encoding='utf-8-sig')
    # Validate algebra with daily values; pooled conditional statistics are derived, not multiplied after averaging.
    np.testing.assert_allclose(daily.early_contribution_bp+daily.later_contribution_bp, daily.saving_vs_market_bp, atol=1e-12)
    np.testing.assert_allclose(summary.early_contribution_bp+summary.later_contribution_bp, summary.saving_vs_market_bp, atol=1e-12)
    np.testing.assert_allclose(summary[['initial_limit','signal_limit','deadline_market','signal_market']].sum(axis=1), 1, atol=1e-12)
    valid = daily.early_fill.gt(0)
    np.testing.assert_allclose(daily.loc[valid,'early_fill']*daily.loc[valid,'early_conditional_saving_bp'], daily.loc[valid,'early_contribution_bp'], atol=1e-12)
    for _,g in daily.groupby(['trade_date','minutes','side']):
        assert (np.diff(g.sort_values('k').early_fill) <= 1e-12).all()
    for _,g in summary.groupby('minutes'):
        by={s:q.set_index('k') for s,q in g.groupby('side')}
        for metric in ['cost_bp','early_contribution_bp','later_contribution_bp','early_fill','elapsed_seconds']:
            np.testing.assert_allclose((by['buy'][metric]+by['sell'][metric])/2, by['both'][metric],atol=1e-12)
    total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in (OUT/'逐任务').glob('*.parquet'))
    common_count = int(coverage.common_tasks.sum()); assert total_rows == common_count*122
    save_json(OUT/'验收.json', dict(dates=18, offsets=OFFSETS.tolist(), original_tasks=int(coverage.original_tasks.sum()),
        common_tasks=common_count, rows=total_rows, per_day=receipts, decomposition_exact=True,
        early_fill_monotonic=True, sides_reproduce_both=True, model_not_retrained=True, frozen_signal_limit=19,
        historical_selection='within0..60 only; upper boundary is not a certified global optimum',
        forward_status='exploratory conditional on forecast horizon previously selected using all18 dates'))
    print(selections[selections.side.eq('both')][['minutes','method','k','cost_bp','early_fill','early_contribution_bp','later_contribution_bp','saving_vs_market_bp','saving_vs_19_bp']].to_string(index=False),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'逐任务').mkdir(exist_ok=True); (OUT/'.checks').mkdir(exist_ok=True)
    sources = [RAW, SOURCE/'模型参数.json', SOURCE/'模型切换判定.json', SOURCE/'三模型逐任务信号.parquet',
               SOURCE/'四组逐任务成交.parquet', ROOT/'utils/opening_two_stage.py']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    decision = json.loads((SOURCE/'模型切换判定.json').read_text(encoding='utf-8'))
    assert decision['selected']=='fixed_1s' and decision['passes']
    rules = dict(input_sha256=hashes, offsets=OFFSETS.tolist(), selected_model='fixed_1s', observation=3, forecast=1,
        signal_limit_offset=19, deadline=10, execution_delay_snapshots=2, quantity=1, dates=18,
        primary='09:30..10:29:59 nominal seconds, actual first snapshot at/after nominal <=0.5s; opening first minute separately',
        cohort='intersection across61 offsets and both sides, starting with previous62899 task cohort',
        early='filled on or before T+3, before signal decision; paired saving vs T immediate market, unfilled early contribution0',
        aggregate='equal dates and equal sides; conditional saving computed using consistent date/side measure',
        selection='max early contribution or min total cost; ties1e-12 closest19 then smaller k',
        bootstrap=dict(block_days=5,repeats=5000,seed=20260928),
        forward='first8 then5, first13 then5; separately eachscope, select shared k on both sides',
        boundary='no automatic expansion beyond60; report boundary optimum as unresolved')
    reg=OUT/'实验口径.json'
    if reg.exists(): assert json.loads(reg.read_text(encoding='utf-8'))==rules
    else: save_json(reg,rules)
    if '--finish-only' not in sys.argv:
        base = pd.read_parquet(SOURCE/'四组逐任务成交.parquet'); base=base[base.strategy.isin(['M','fixed_1s'])]
        signals=pd.read_parquet(SOURCE/'三模型逐任务信号.parquet'); signals=signals[signals.strategy.eq('fixed_1s') & signals.common]
        assert len(signals)==62899
        raw=pd.read_parquet(RAW);raw['source_row']=np.arange(len(raw));jobs=[]
        for date,b in base.groupby('trade_date',sort=True):
            if (OUT/'.checks'/f'{date}.json').exists():continue
            start=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai')
            frame=raw[(raw.Datetime>=start)&(raw.Datetime<=start+pd.Timedelta(seconds=3620))]
            frame=frame[['Datetime','LastPrice','BidPrice1','AskPrice1','BidVolume1','AskVolume1','Volume','source_row']].copy()
            jobs.append((date,frame,b.copy(),signals[signals.trade_date.eq(date)].copy()))
        with ProcessPoolExecutor(max_workers=2) as pool:
            for start in range(0,len(jobs),4):
                for receipt in pool.map(worker,jobs[start:start+4]):print(json.dumps(receipt),flush=True)
    finish()
    assert all(sha(p)==hashes[str(p.relative_to(ROOT))] for p in sources)
    save_json(OUT/'输入保持不变.json',dict(all_verified=True,input_sha256=hashes))


if __name__=='__main__':main()
