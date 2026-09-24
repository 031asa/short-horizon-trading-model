"""Threshold experiment with initial-only C19 (LastPrice at T+3) as baseline."""
import sys
sys.path=[p for p in sys.path if not p.replace('\\','/').endswith('.cache/model-packages')]
from scripts.run_second_ranges import *
from scripts import run_three_action_threshold as legacy
from scripts.run_initial_only_c19 import SOURCES,SIGNALS,COUNTS
from concurrent.futures import ProcessPoolExecutor
import shutil

DEST=OUT/'rebased_three_action_threshold'
CORRECT=OUT/'initial_only_c19'
OLD40=OUT/'three_action_threshold'
KEY=legacy.KEY;TASK=legacy.TASK


def worker(job):
    legacy.DEST=DEST/'18d'/'.analysis'
    r=legacy.replay_date(job)
    save_json(legacy.DEST/'.replay'/f'{r["trade_date"]}.json',r)
    return r


def main():
    DEST.mkdir(exist_ok=True)
    inputs=[RAW,MODELS,*SIGNALS.values(),OLD40/'三种动作逐单回放.parquet',OLD40/'阈值策略逐单结果.parquet']
    inputs += [p/'四策略逐任务配对.parquet' for p in SOURCES.values()]
    inputs += [CORRECT/s/'四策略逐任务配对.parquet' for s in ['18d','40d']]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    save_json(DEST/'预登记口径.json',dict(input_sha256=hashes,
        primary_baseline='C_new: initial passive19; adverse direction market at3; favorable/flat LastPrice at3',
        threshold_rule='adverse argmax => market; flat=>LastPrice; favorable score>=tau => passive19 else LastPrice',
        favorable_score='direction*(p_down-p_up)',thresholds=list(legacy.THRESHOLDS),
        selection='one tau per scope, lowest first60min second-grid dateequal all-order cost; no refit',
        chart_example_threshold=.05,old_strategy='C_dual19 remains a separate reference, not primary baseline',
        rerun18=True,reuse_verified40_branches=True,old_outputs_unchanged=True))
    for scope in ['18d','40d']:
        folder=DEST/scope;folder.mkdir(exist_ok=True)
        internal=folder/'.analysis';internal.mkdir(exist_ok=True);(internal/'.replay').mkdir(exist_ok=True)
        source=folder/'.source';source.mkdir(exist_ok=True)
        old=pd.read_parquet(SOURCES[scope]/'四策略逐任务配对.parquet')
        if 'phase' not in old:old['phase']='original_forward_fold_research'
        old.to_parquet(source/'四策略逐任务配对.parquet',index=False)
        signals=pd.read_parquet(SIGNALS[scope]);signals.to_parquet(source/'全部日期信号.parquet',index=False)
        if scope=='18d':
            raw=pd.read_parquet(RAW);jobs=[]
            for date,frame in raw.groupby(raw.Date.astype(str),sort=True):
                cohort=old[old.trade_date.eq(date)&old.strategy.eq('C')]
                if cohort.empty:continue
                start=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai')
                frame=frame[(frame.Datetime>=start)&(frame.Datetime<=start+pd.Timedelta(seconds=3620))]
                frame=frame[[k for k in ['Datetime','LastPrice','BidPrice1','AskPrice1','BidVolume1','AskVolume1','Volume','source_row'] if k in frame]].copy()
                jobs.append((date,frame,cohort[KEY+legacy.FIELDS].copy(),signals[signals.trade_date.eq(date)][['nominal_second','signal']].copy()))
            checks=[]
            with ProcessPoolExecutor(max_workers=2) as pool:
                for begin in range(0,len(jobs),4):
                    for r in pool.map(worker,jobs[begin:begin+4]):
                        checks.append(r);assert not r['errors'];print('18d actions verified',r['trade_date'],flush=True)
            branches=pd.concat([pd.read_parquet(internal/'.replay'/f'{r["trade_date"]}.parquet') for r in checks],ignore_index=True)
            assert len(branches)==len(old)//8*6
            branches.to_parquet(internal/'三种动作逐单回放.parquet',index=False)
            save_json(folder/'动作回放验收.json',dict(checks=checks,verified_fills=sum(r['fills_checked'] for r in checks),extra_removed_tasks=0))
        else:
            shutil.copyfile(OLD40/'三种动作逐单回放.parquet',internal/'三种动作逐单回放.parquet')
            save_json(folder/'动作回放验收.json',dict(mode='reuse verified same-model same-cohort all40 branches',
                source_sha256=sha(OLD40/'三种动作逐单回放.parquet'),extra_removed_tasks=0))
        save_json(internal/'回放验收.json',dict(extra_removed_tasks=0))
        legacy.DEST=internal;legacy.SOURCE=source
        legacy.EXPECTED={(m,g):COUNTS[scope][g][m] for g in ['minute','second'] for m in [1,19,30,60]}
        legacy.analyze()
        export(scope,folder,internal)
    for p in inputs:assert sha(p)==hashes[str(p.relative_to(ROOT))]
    save_json(DEST/'输入复核.json',dict(all_input_hashes_unchanged=True))


def rename(frame):
    frame=frame.copy()
    mapping={c:c.replace('vs_L0','vs_C_new').replace('vs_C19','vs_C_dual19') for c in frame.columns}
    frame=frame.rename(columns=mapping)
    if 'policy' in frame:frame['policy']=frame.policy.replace({'L0':'C_new','C19':'C_dual19'})
    return frame


def export(scope,folder,internal):
    summary=rename(pd.read_csv(internal/'全策略8组分方向汇总.csv'))
    daily=rename(pd.read_csv(internal/'全策略逐日结果.csv'))
    orders=rename(pd.read_parquet(internal/'阈值策略逐单结果.parquet'))
    base=pd.read_parquet(CORRECT/scope/'四策略逐任务配对.parquet')
    c=orders[orders.policy.eq('C_new')].set_index(KEY).sort_index()
    expected=base[base.strategy.eq('C')].set_index(KEY).loc[c.index]
    fields=['cost_bp','cost_ticks','fill_ticks','fill_seconds','elapsed_seconds']
    np.testing.assert_array_equal(c[fields],expected[fields])
    if scope=='40d':
        previous=rename(pd.read_parquet(OLD40/'阈值策略逐单结果.parquet')).set_index(KEY+['policy']).sort_index()
        current=orders.set_index(KEY+['policy']).loc[previous.index]
        np.testing.assert_array_equal(current[fields],previous[fields])
    # Recheck each saved threshold decision with an independent scalar expression.
    info=pd.read_parquet(internal/'任务与信号.parquet').set_index(KEY).sort_index()
    branch=pd.read_parquet(internal/'三种动作逐单回放.parquet')
    byaction={a:branch[branch.action.eq(a)].set_index(KEY).loc[info.index] for a in legacy.ACTIONS}
    for tau in legacy.THRESHOLDS:
        label=f'T{tau:g}';actual=orders[orders.policy.eq(label)].set_index(KEY).loc[info.index]
        actions=np.asarray(['market' if r.direction*r.signal>0 else 'passive' if r.signal!=0 and r.direction*(r.p_down-r.p_up)>=tau else 'lastprice' for r in info.reset_index().itertuples()])
        for action in legacy.ACTIONS:
            use=actions==action;np.testing.assert_array_equal(actual.loc[use,fields],byaction[action].loc[use,fields])
    for name,frame in [('阈值与基准_分方向汇总.csv',summary),('阈值与基准_逐日结果.csv',daily)]:
        frame.to_csv(folder/name,index=False,encoding='utf-8-sig')
    orders.to_parquet(folder/'阈值与基准_逐单结果.parquet',index=False)
    for name in ['任务与信号.parquet','信号强弱分箱_三动作成本.csv','信号强弱分箱_逐日.csv','留一日期阈值敏感性.csv','日期块阈值选择频率.csv']:
        if name=='留一日期阈值敏感性.csv':
            rename(pd.read_csv(internal/name)).to_csv(folder/name,index=False,encoding='utf-8-sig')
        else:
            shutil.copyfile(internal/name,folder/name)
    primary=summary[summary.minutes.eq(60)&summary.grid.eq('second')&summary.side.eq('both')&summary.subset.eq('all')]
    candidates=[f'T{t:g}' for t in legacy.THRESHOLDS]
    best=min(candidates,key=lambda k:(float(primary.set_index('policy').loc[k,'cost_bp']),float(k[1:])))
    # Paired daily means and date-block intervals must use the corrected C baseline.
    for row in summary[summary.side.eq('both')&summary.subset.eq('all')&summary.policy.isin(candidates)].itertuples():
        z=daily[daily.minutes.eq(row.minutes)&daily.grid.eq(row.grid)&daily.side.eq('both')&daily.subset.eq('all')]
        pivot=z.pivot(index='trade_date',columns='policy',values='cost_bp')
        delta=pivot.C_new-pivot[row.policy]
        assert abs(delta.mean()-row.saving_vs_C_new)<1e-8
        lo,hi=date_block_interval(delta)
        np.testing.assert_allclose([lo,hi],[row.saving_vs_C_new_low,row.saving_vs_C_new_high],atol=1e-8,rtol=0)
    primary.to_csv(folder/'首小时逐秒阈值总表.csv',index=False,encoding='utf-8-sig')
    save_json(folder/'验收与选择.json',dict(scope=scope,selected_policy=best,selected_threshold=float(best[1:]),
        primary_baseline='C_new',no_threshold_means='C_new (equivalent to infinite passive threshold), NOT tau0',
        tau0_equals_dual19=bool(np.array_equal(orders[orders.policy.eq('T0')].cost_bp.to_numpy(),orders[orders.policy.eq('C_dual19')].cost_bp.to_numpy())),
        corrected_C_replayed_exactly=len(c),threshold_decisions_replayed=True,paired_C_new_intervals_checked=64,
        all40_old_threshold_paths_unchanged=scope=='40d',summary_rows=len(summary),old_outputs_unchanged=True))


if __name__=='__main__':main()
