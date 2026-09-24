"""Independent saved-output aggregation, pairing and branch-selection checks."""
from scripts.run_three_action_threshold import *


def main():
    info=pd.read_parquet(DEST/'任务与信号.parquet').set_index(KEY).sort_index()
    orders=pd.read_parquet(DEST/'阈值策略逐单结果.parquet')
    branch=pd.read_parquet(DEST/'三种动作逐单回放.parquet')
    summary=pd.read_csv(DEST/'全策略8组分方向汇总.csv')
    source=pd.read_parquet(SOURCE/'四策略逐任务配对.parquet')
    baseline=source[source.strategy.eq('C')].set_index(KEY).loc[info.index]
    assert not orders.duplicated(KEY+['policy']).any()
    assert len(info)==270636 and info.index.droplevel('direction').nunique()==135318
    branch_arrays={a:branch[branch.action.eq(a)].set_index(KEY).loc[info.index] for a in ACTIONS}
    wide=orders.pivot(index=KEY,columns='policy',values='cost_bp').reindex(info.index)
    assert wide.notna().all().all()
    assert np.array_equal(wide.T0.values,wide.C19.values)
    fields=['fill_seconds','fill_ticks','cost_ticks','cost_bp','elapsed_seconds']
    checked_cells=0; checked_early=0
    for policy,g in orders.groupby('policy',sort=False):
        g=g.set_index(KEY).loc[info.index]
        if policy in ['C19','M','A','B']:
            expected=source[source.strategy.eq('C' if policy=='C19' else policy)].set_index(KEY).loc[info.index]
            np.testing.assert_array_equal(g[fields],expected[fields])
        else:
            early=~info.survivor3
            np.testing.assert_array_equal(g.loc[early,fields],baseline.loc[early,fields]);checked_early+=int(early.sum())
            if policy.startswith('T') or policy=='L0':
                threshold=float(policy[1:]) if policy!='L0' else np.inf
                # Independent scalar decision, deliberately not calling the selection helper.
                decisions=[]
                for r in info.reset_index().itertuples():
                    if r.direction*r.signal>0:decisions.append('market')
                    elif r.signal==0:decisions.append('lastprice')
                    elif (r.p_down-r.p_up)*r.direction>=threshold:decisions.append('passive')
                    else:decisions.append('lastprice')
            else:decisions=[policy[2:]]*len(info)
            choice=np.asarray(decisions)
            for action in ACTIONS:
                use=choice==action
                np.testing.assert_array_equal(g.loc[use,fields],branch_arrays[action].loc[use,fields])
        np.testing.assert_allclose(g.cost_bp,g.cost_ticks/baseline.initial_ticks*10000,atol=1e-12,rtol=0)
        np.testing.assert_allclose(g.elapsed_seconds,g.fill_seconds-baseline.task_seconds,atol=0,rtol=0)
        frame=g.reset_index()
        for row in summary[summary.policy.eq(policy)].itertuples():
            sub=frame[frame.nominal_second.lt(row.minutes*60)]
            if row.grid=='minute':sub=sub[sub.nominal_second.mod(60).eq(0)]
            if row.subset=='survivor3':sub=sub[sub.set_index(KEY).index.isin(info[info.survivor3].index)]
            sides=[1] if row.side=='buy' else [-1] if row.side=='sell' else [1,-1]
            sub=sub[sub.direction.isin(sides)]
            day=[];valid_dates=[]
            for date,h in sub.groupby('trade_date'):
                if set(h.direction)!=set(sides):continue
                values=[h[h.direction.eq(side)].cost_bp.mean() for side in sides]
                day.append(float(np.mean(values)));valid_dates.append(date)
            assert len(day)==row.days
            use=sub[sub.trade_date.isin(valid_dates)]
            assert len(use)==row.orders and len(use[TASK].drop_duplicates())==row.tasks
            assert abs(np.mean(day)-row.cost_bp)<1e-8
            assert abs(row.initial_limit+row.signal_limit+row.signal_market+row.deadline_market+row.immediate_market-1)<1e-8
            assert abs(row.win_vs_C19+row.tie_vs_C19+row.loss_vs_C19-1)<1e-8
            checked_cells+=1
    b=pd.read_csv(DEST/'信号强弱分箱_三动作成本.csv')
    for action in ACTIONS:
        assert b[b.action.eq(action)&b.minutes.eq(60)&b.grid.eq('second')].orders.sum()==info.survivor3.sum()
    cfg=json.loads((DEST/'预登记口径.json').read_text(encoding='utf-8'))
    for rel,digest in cfg['input_sha256'].items():assert sha(ROOT/rel)==digest
    save_json(DEST/'交付验收.json',dict(summary_rows_independently_checked=checked_cells,
        early_rows_unchanged=checked_early, matched_orders_per_policy=len(info),
        decision_replay=True, cost_and_elapsed_identities=True, stage_and_outcome_sum100=True,
        all_source_hashes_unchanged=True, bins_cover_all_survivors=True,
        original_C_and_MAB_exact=True, zero_threshold_exactly_C19=True))
    print('Verified',checked_cells,'summary rows; all saved decisions and all original strategy fills match',flush=True)


if __name__=='__main__':main()
