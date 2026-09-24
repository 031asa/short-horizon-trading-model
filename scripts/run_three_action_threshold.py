"""Replay three T+3 actions, then evaluate frozen score thresholds on all 40 dates."""
import sys
# Child workers inherit paths added by legacy model imports. Arrow must load
# with the bundled cloudpickle before that optional legacy package directory.
sys.path = [p for p in sys.path if not p.replace('\\', '/').endswith('.cache/model-packages')]
from scripts.run_second_ranges import *
from utils.opening_three_action import ACTIONS, THRESHOLDS, select_actions
from concurrent.futures import ProcessPoolExecutor

SOURCE = OUT/'four_strategy_c19_all40'
DEST = OUT/'three_action_threshold'
KEY = ['trade_date', 'nominal_second', 'direction']
TASK = KEY[:2]
FIELDS = ['task_seconds', 'initial_ticks', 'fill_seconds', 'fill_ticks', 'cost_ticks',
          'cost_bp', 'elapsed_seconds', 'fill_kind', 'fill_stage', 'pending_cancelled']
EXPECTED = {(1, 'minute'): 37, (1, 'second'): 2258, (19, 'minute'): 720,
            (19, 'second'): 43233, (30, 'minute'): 1146, (30, 'second'): 68473,
            (60, 'minute'): 2279, (60, 'second'): 135318}


def replay_date(job):
    date, frame, cohort, signals = job
    start = pd.Timestamp(date+' 09:30', tz='Asia/Shanghai')
    cfg = ICConfig(schedule=ScheduleConfig(cold_start_seconds=0, task_range_seconds=3600, observation_seconds=(3.,)))
    d = SessionData(frame, start, cfg)
    lookup = signals.set_index('nominal_second')
    rows = []; errors = []; checked = 0; replayed = 0; early = 0
    for old in cohort.itertuples(index=False):
        signal = int(lookup.loc[old.nominal_second, 'signal'])
        original_action = 'market' if old.direction*signal > 0 else 'passive'
        for action in ACTIONS:
            r = simulate(d, old.task_seconds, old.direction, 'C_limit_first', signal,
                         c_limit_offset_ticks=19, c_signal_action=action)
            trace = r.pop('trace')
            if r['status'] != 'filled':
                errors.append(dict(trade_date=date, nominal_second=old.nominal_second,
                                   direction=old.direction, action=action, reason=r['reason']))
                continue
            ix, price = independent_fill(d, trace, old.direction)
            assert price == r['fill_ticks'] and d.times[ix] == r['fill_seconds']
            checked += 1
            if action == original_action or old.elapsed_seconds <= 3:
                for name in FIELDS:
                    assert r[name] == getattr(old, name), (date, old.nominal_second, name)
                replayed += int(action == original_action)
                early += int(old.elapsed_seconds <= 3)
            orders = [e for e in trace if e['event'] == 'submit']
            assert orders[0]['limit_ticks'] == old.initial_ticks-old.direction*19
            assert orders[0]['time'] == old.task_seconds
            for e in orders:
                assert e['time'] in (old.task_seconds, old.task_seconds+3, old.task_seconds+10)
                if e['stage'] == 'signal' and e['kind'] == 'limit':
                    expected = d.p[d.source_index(old.task_seconds+3)]-old.direction*(19 if action == 'passive' else 0)
                    assert e['limit_ticks'] == expected
            rows.append(dict(trade_date=date, nominal_second=old.nominal_second,
                             direction=old.direction, action=action, **{k:r[k] for k in FIELDS}))
    path = DEST/'.replay'/f'{date}.parquet'
    pd.DataFrame(rows).to_parquet(path, index=False)
    return dict(trade_date=date, fills_checked=checked, original_rows_replayed=replayed,
                early_fill_checks=early, errors=errors, rows=len(rows))


def run():
    DEST.mkdir(parents=True, exist_ok=True); (DEST/'.replay').mkdir(exist_ok=True)
    inputs = [RAW, MODELS, SOURCE/'四策略逐任务配对.parquet', SOURCE/'全部日期信号.parquet']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in inputs}
    save_json(DEST/'预登记口径.json', dict(input_sha256=hashes, thresholds=list(THRESHOLDS),
        model='all40 latest frozen W3 fold4, no refit', initial_offset_ticks=19,
        score='direction*(p_down-p_up); confidence ranking, not predicted price magnitude',
        rule='adverse argmax => market; flat => LastPrice; favorable => passive19 iff score>=threshold',
        control='C19 original and adverse-market/otherwise-LastPrice control',
        selection='single threshold minimizing date-equal cost, first60min eachsecond ALL tasks; exact ties smaller threshold',
        source_dates=40, bootstrap='5000 circular 5-date blocks; seed20260923; descriptive, selection unadjusted',
        cohort='existing all40 cohort intersect all3 actions and both sides; report additional losses',
        replay='initial live order preserved through replacement delay; fills at/before3 unchanged',
        aggregation='date equal; buy/sell half after within-date-side means; subset requires both sides per date',
        limitations='existing researched dates; 35 model training dates; no independent out-of-sample claim'))
    old = pd.read_parquet(inputs[2]); signals = pd.read_parquet(inputs[3])
    raw = pd.read_parquet(RAW); c = old[old.strategy.eq('C')]
    jobs = []
    for date, frame in raw.groupby(raw.Date.astype(str), sort=True):
        cohort = c[c.trade_date.eq(date)]
        if cohort.empty: continue
        start = pd.Timestamp(date+' 09:30', tz='Asia/Shanghai')
        frame = frame[(frame.Datetime >= start) & (frame.Datetime <= start+pd.Timedelta(seconds=3620))]
        jobs.append((date, frame, cohort, signals[signals.trade_date.eq(date)]))
    checks = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for check in pool.map(replay_date, jobs):
            checks.append(check)
            print(check['trade_date'], 'three actions checked:', check['fills_checked'], flush=True)
    allrows = pd.concat([pd.read_parquet(DEST/'.replay'/f'{c["trade_date"]}.parquet') for c in checks], ignore_index=True)
    valid = allrows.groupby(TASK).size().eq(6)
    allrows = allrows.merge(valid.rename('common'), on=TASK, validate='many_to_one')
    allrows = allrows[allrows.common].drop(columns='common')
    assert not allrows.duplicated(KEY+['action']).any()
    allrows.to_parquet(DEST/'三种动作逐单回放.parquet', index=False)
    errors = [e for c in checks for e in c['errors']]
    pd.DataFrame(errors, columns=KEY+['action', 'reason']).to_csv(DEST/'新增排除任务.csv', index=False, encoding='utf-8-sig')
    for p in inputs: assert sha(p) == hashes[str(p.relative_to(ROOT))]
    save_json(DEST/'回放验收.json', dict(checks=checks, source_tasks=len(old)//8,
        common_tasks=len(allrows)//6, extra_removed_tasks=len(old)//8-len(allrows)//6,
        original_inputs_unchanged=True, verified_fills=sum(c['fills_checked'] for c in checks),
        exact_C_replay=sum(c['original_rows_replayed'] for c in checks),
        early_fill_checks=sum(c['early_fill_checks'] for c in checks)))
    analyze()


def write_csv(frame, name):
    frame.to_csv(DEST/name, index=False, encoding='utf-8-sig', float_format='%.10g')


def analyze():
    branch = pd.read_parquet(DEST/'三种动作逐单回放.parquet')
    old = pd.read_parquet(SOURCE/'四策略逐任务配对.parquet')
    signals = pd.read_parquet(SOURCE/'全部日期信号.parquet')
    cohort = branch[KEY].drop_duplicates().sort_values(KEY).reset_index(drop=True)
    info = cohort.merge(old[old.strategy.eq('C')][KEY+['elapsed_seconds','phase']], on=KEY, validate='one_to_one')
    info = info.rename(columns={'elapsed_seconds':'C_elapsed'})
    info = info.merge(signals[TASK+['signal','p_down','p_flat','p_up']], on=TASK, validate='many_to_one')
    info['survivor3'] = info.C_elapsed > 3
    info['favorable_score'] = info.direction*(info.p_down-info.p_up)
    info.to_parquet(DEST/'任务与信号.parquet', index=False)
    arrays = {}
    for action in ACTIONS:
        frame = cohort.merge(branch[branch.action.eq(action)], on=KEY, validate='one_to_one')
        arrays[action] = frame
    legacy = {s:cohort.merge(old[old.strategy.eq(s)], on=KEY, validate='one_to_one') for s in ['M','A','B','C']}
    fields = ['cost_bp', 'cost_ticks', 'elapsed_seconds', 'fill_kind', 'fill_stage', 'fill_seconds', 'fill_ticks']
    cube = {k:np.column_stack([arrays[a][k].to_numpy() for a in ACTIONS]) for k in fields}
    ids = np.arange(len(info)); policies = {}
    for label, threshold in [('L0', np.inf)]+[(f'T{t:g}',t) for t in THRESHOLDS]:
        choice = select_actions(info.direction, info.signal, info.p_down, info.p_up, threshold)
        f = info[KEY+['survivor3']].copy()
        for k in fields: f[k] = cube[k][ids, choice]
        f['chosen_action'] = np.array(ACTIONS)[choice]
        f.loc[~f.survivor3, 'chosen_action'] = 'already_filled'
        policies[label] = f
    for s, frame in legacy.items():
        f = info[KEY+['survivor3']].copy()
        for k in fields: f[k] = frame[k].to_numpy()
        f['chosen_action'] = 'legacy'
        policies['C19' if s=='C' else s] = f
    # Diagnostic ablations: same initial19 and deadline, one fixed T+3 action
    # for every survivor. These are not additional threshold candidates.
    for action, frame in arrays.items():
        f = info[KEY+['survivor3']].copy()
        for k in fields: f[k] = frame[k].to_numpy()
        f['chosen_action'] = np.where(info.survivor3, action, 'already_filled')
        policies['R_'+action] = f
    daily = []; summary = []; perorders = []
    numeric = ['cost_bp','elapsed_seconds','initial_limit','signal_limit','signal_market','deadline_market','immediate_market',
               'saving_vs_C19','saving_vs_L0','saving_vs_M','win_vs_C19','tie_vs_C19','loss_vs_C19',
               'action_market','action_lastprice','action_passive','action_already_filled']
    for label, frame in policies.items():
        frame = frame.copy(); frame['policy'] = label
        frame['initial_limit'] = (frame.fill_kind.eq('limit') & frame.fill_stage.eq('initial')).astype(float)
        frame['signal_limit'] = (frame.fill_kind.eq('limit') & frame.fill_stage.eq('signal')).astype(float)
        frame['signal_market'] = (frame.fill_kind.eq('market') & frame.fill_stage.eq('signal')).astype(float)
        frame['deadline_market'] = (frame.fill_kind.eq('market') & frame.fill_stage.eq('deadline')).astype(float)
        frame['immediate_market'] = frame.fill_stage.eq('immediate').astype(float)
        assert np.allclose(frame[['initial_limit','signal_limit','signal_market','deadline_market','immediate_market']].sum(axis=1), 1)
        for control in ['C19','L0','M']: frame['saving_vs_'+control] = policies[control].cost_bp-frame.cost_bp
        diff = policies['C19'].cost_ticks-frame.cost_ticks
        frame['win_vs_C19'] = (diff>1e-10).astype(float)
        frame['tie_vs_C19'] = (abs(diff)<=1e-10).astype(float)
        frame['loss_vs_C19'] = (diff< -1e-10).astype(float)
        for action in ['market','lastprice','passive','already_filled']:
            frame['action_'+action] = frame.chosen_action.eq(action).astype(float)
        perorders.append(frame[KEY+['policy','chosen_action']+fields])
        for minutes in [1,19,30,60]:
            for grid in ['minute','second']:
                base = frame[(frame.nominal_second < minutes*60) & ((frame.nominal_second%60==0) if grid=='minute' else True)]
                if json.loads((DEST/'回放验收.json').read_text())['extra_removed_tasks']==0:
                    assert len(base)//2 == EXPECTED[(minutes,grid)]
                for subset in ['all','survivor3']:
                    sub = base if subset=='all' else base[base.survivor3]
                    for side, sides in [('buy',[1]),('sell',[-1]),('both',[1,-1])]:
                        g = sub[sub.direction.isin(sides)]
                        ds = g.groupby(['trade_date','direction'])[numeric].mean()
                        present = ds.groupby(level=0).size()
                        valid_dates = present[present.eq(len(sides))].index
                        dm = ds.loc[ds.index.get_level_values(0).isin(valid_dates)].groupby(level=0).mean()
                        if dm.empty: continue
                        used = g[g.trade_date.isin(valid_dates)]
                        meta = dict(policy=label,minutes=minutes,grid=grid,subset=subset,side=side)
                        dr = dm.reset_index(); dr['orders'] = used.groupby('trade_date').size().reindex(dm.index).values
                        for k,v in meta.items(): dr[k] = v
                        daily.append(dr)
                        result = dict(**meta, days=len(dm), tasks=used[TASK].drop_duplicates().shape[0], orders=len(used),
                                      excluded_side_incomplete_orders=len(g)-len(used), **dm.mean().to_dict())
                        for metric in ['saving_vs_C19','saving_vs_L0','saving_vs_M']:
                            lo,hi = date_block_interval(dm[metric].values)
                            result[metric+'_low'],result[metric+'_high'] = lo,hi
                        result['days_better_C19'] = int((dm.saving_vs_C19>1e-10).sum())
                        result['days_equal_C19'] = int((abs(dm.saving_vs_C19)<=1e-10).sum())
                        result['days_worse_C19'] = int((dm.saving_vs_C19< -1e-10).sum())
                        summary.append(result)
        print('summarized', label, flush=True)
    s = pd.DataFrame(summary); day = pd.concat(daily, ignore_index=True)
    write_csv(s, '全策略8组分方向汇总.csv'); write_csv(day,'全策略逐日结果.csv')
    pd.concat(perorders, ignore_index=True).to_parquet(DEST/'阈值策略逐单结果.parquet', index=False)
    primary = s[(s.minutes==60)&(s.grid=='second')&(s.subset=='all')&(s.side=='both')]
    candidates = [f'T{t:g}' for t in THRESHOLDS]
    best = min(candidates, key=lambda k:(float(primary.set_index('policy').loc[k,'cost_bp']), float(k[1:])))
    selected = s[s.policy.isin(['M','A','B','C19','L0',best]) & s.side.eq('both')]
    write_csv(selected, '主要策略8组比较.csv')
    # LOO is selection sensitivity, not independent validation of the frozen model.
    dd = day[(day.minutes==60)&day.grid.eq('second')&day.subset.eq('all')&day.side.eq('both')]
    wide = dd.pivot(index='trade_date', columns='policy', values='cost_bp')
    loo = []
    for date in wide.index:
        scores = wide.drop(index=date)[candidates].mean()
        chosen = min(candidates, key=lambda k:(scores[k],float(k[1:])))
        loo.append(dict(trade_date=date, chosen_policy=chosen, cost_bp=wide.loc[date,chosen],
                        saving_vs_C19=wide.loc[date,'C19']-wide.loc[date,chosen],
                        saving_vs_L0=wide.loc[date,'L0']-wide.loc[date,chosen]))
    write_csv(pd.DataFrame(loo),'留一日期阈值敏感性.csv')
    # Shared date-block draws keep paired threshold comparisons intact.
    rng = np.random.default_rng(20260923); n=len(wide)
    starts = rng.integers(0,n,(5000,int(np.ceil(n/5))))
    index = ((starts[...,None]+np.arange(5))%n).reshape(5000,-1)[:,:n]
    boot = wide[candidates].to_numpy()[index].mean(axis=1)
    winners = boot.argmin(axis=1)
    write_csv(pd.DataFrame(dict(policy=candidates, selections=np.bincount(winners,minlength=len(candidates)),
                               proportion=np.bincount(winners,minlength=len(candidates))/5000)), '日期块阈值选择频率.csv')
    # Fixed score bins; action comparisons use exactly the same survivor orders.
    edges = [-1,-.3,-.2,-.15,-.1,-.075,-.05,-.025,0,.025,.05,.075,.1,.15,.2,.3,1]
    bins = pd.cut(info.favorable_score,edges,include_lowest=True,right=False)
    bin_rows = []; bin_daily = []
    for action in ACTIONS:
        f = info[KEY+['survivor3','favorable_score']].copy()
        f['bin'] = bins; f['cost_bp'] = arrays[action].cost_bp
        f['saving_vs_3s_market'] = arrays['market'].cost_bp-arrays[action].cost_bp
        f['elapsed_seconds'] = arrays[action].elapsed_seconds
        f['deadline_rate'] = (arrays[action].fill_kind.eq('market') & arrays[action].fill_stage.eq('deadline')).astype(float)
        for minutes in [1,19,30,60]:
            for grid in ['minute','second']:
                g=f[f.survivor3&(f.nominal_second<minutes*60)&((f.nominal_second%60==0) if grid=='minute' else True)]
                for binid,h in g.groupby('bin',observed=True):
                    # Conditional-bin composition need not be 50/50. Use actual surviving sides within each day.
                    dm=h.groupby('trade_date')[['cost_bp','saving_vs_3s_market','elapsed_seconds','deadline_rate']].mean()
                    lo,hi=date_block_interval(dm.saving_vs_3s_market)
                    meta=dict(action=action,minutes=minutes,grid=grid,bin=str(binid),left=binid.left,right=binid.right)
                    bin_rows.append(dict(**meta,days=len(dm),orders=len(h),buy_orders=int(h.direction.eq(1).sum()),
                        **dm.mean().to_dict(), saving_low=lo,saving_high=hi))
                    dr=dm.reset_index()
                    for k,v in meta.items():dr[k]=v
                    bin_daily.append(dr)
    write_csv(pd.DataFrame(bin_rows),'信号强弱分箱_三动作成本.csv')
    write_csv(pd.concat(bin_daily,ignore_index=True),'信号强弱分箱_逐日.csv')
    save_json(DEST/'筛选与汇总验收.json',dict(selected_threshold=float(best[1:]), selected_policy=best,
        selection_metric='first60min second all dateequal buy/sellhalf cost',
        threshold_scope='same selected threshold across all8 groups, no per-group optimum',
        summary_rows=len(s),daily_rows=len(day),orders_per_policy=len(info),
        early_orders=int((~info.survivor3).sum()),survivor_orders=int(info.survivor3.sum()),
        predicted_flat_orders=int(info.signal.eq(0).sum()),
        LOO_choices=pd.Series([x['chosen_policy'] for x in loo]).value_counts().to_dict(),
        main_table=primary.to_dict('records'),
        fixed_actions_used_no_ex_post_oracle=True,matched_action_cohort=True))


if __name__ == '__main__':
    analyze() if '--analyze' in sys.argv else run()
