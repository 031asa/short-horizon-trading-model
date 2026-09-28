"""Censored passive-fill timing and opening-minute execution attribution.

Read-only analysis of frozen C3 orders. No new execution rule is selected.
"""
from scripts.run_repeated_decisions import ROOT,OUT as SOURCE,SOURCE as HORIZON,RAW,TASK,KEY,sha,save_json
from utils.opening_ic import ICConfig,SessionData
from utils.opening_schedule import ScheduleConfig
from utils.opening_two_stage import date_block_interval
from utils.opening_fill_distribution import normal_diagnostics,block_moment_intervals,weighted_quantile,weighted_km,NORMAL
import json
import numpy as np
import pandas as pd

OUT=SOURCE.parent/'limit_fill_timing'
SCOPES={'first_minute':(0,60),'rest_59':(60,3600),'first_hour':(0,3600)}
NAMES={'first_minute':'开盘首分钟','rest_59':'随后59分钟','first_hour':'开盘后首小时'}
STAGES=['initial_limit','limit3','market3','market10']


def select_scope(frame,scope,side='both'):
    lo,hi=SCOPES[scope];g=frame[frame.nominal_second.ge(lo)&frame.nominal_second.lt(hi)].copy()
    if side!='both':g=g[g.direction.eq(1 if side=='buy' else -1)]
    g['weight']=1/g.groupby('trade_date').nominal_second.transform('size')
    return g


def enrich():
    allorders=pd.read_parquet(SOURCE/'全部逐任务成交.parquet')
    o=allorders[allorders.strategy.eq('C3')].copy()
    m=allorders[allorders.strategy.eq('M')][KEY+['fill_seconds','fill_ticks']].rename(columns={'fill_seconds':'m_fill_seconds','fill_ticks':'m_fill_ticks'})
    o=o.merge(m,on=KEY,validate='one_to_one')
    sig=pd.read_parquet(HORIZON/'三模型逐任务信号.parquet')
    sig=sig[sig.strategy.eq('fixed_1s')&sig.common]
    o=o.merge(sig[TASK+['signal','p_down','p_flat','p_up','cum_1','cum_3']],on=TASK,validate='many_to_one')
    commands=pd.read_parquet(SOURCE/'全部下单记录.parquet')
    c=commands[commands.strategy.eq('C3')&commands.decision_second.eq(3)&commands.kind.eq('limit')]
    c=c[KEY+['arrival_seconds','realized_arrival','limit_ticks']].rename(columns={'arrival_seconds':'limit3_arrival','realized_arrival':'limit3_activated','limit_ticks':'limit3_ticks'})
    o=o.merge(c,on=KEY,how='left',validate='one_to_one')
    o['limit3_submitted']=o.limit3_arrival.notna()
    o['limit3_activated']=o.limit3_activated.eq(True)
    o['limit_branch']=o.elapsed_seconds.gt(3+1e-10)&(o.direction*o.signal).le(0)
    o['limit_event']=o.fill_kind.eq('limit')
    assert (~o.limit3_submitted | o.limit_branch).all()
    assert o[o.limit3_activated].stage.isin(['limit3','market10']).all()
    assert o[o.limit_branch].stage.isin(['initial_limit','limit3','market10']).all()
    o['wait_from3']=o.elapsed_seconds-3
    o['active_wait']=o.fill_seconds-o.limit3_arrival
    assert o.loc[o.limit3_activated,'active_wait'].ge(-1e-10).all()
    o['saving_bp']=o.market_cost_bp-o.cost_bp
    o['extra_cost_bp']=-o.saving_bp
    raw=pd.read_parquet(RAW);raw['source_row']=np.arange(len(raw));enriched=[]
    for date,g in o.groupby('trade_date',sort=True):
        start=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai')
        f=raw[(raw.Datetime>=start)&(raw.Datetime<=start+pd.Timedelta(seconds=3620))]
        d=SessionData(f.copy(),start,ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,task_range_seconds=3600,observation_seconds=(3.,))))
        t=g.task_seconds.to_numpy();side=g.direction.to_numpy();p0=g.initial_ticks.to_numpy()
        i0=np.searchsorted(d.times,t,'right')-1
        i3=np.searchsorted(d.times,t+3,'right')-1
        i10=np.searchsorted(d.times,t+10,'right')-1
        im=np.searchsorted(d.times,g.m_fill_seconds,'left')
        iff=np.searchsorted(d.times,g.fill_seconds,'left')
        np.testing.assert_allclose(d.p[i0],p0,atol=0,rtol=0)
        def quote(ids):return np.where(side==1,d.a[ids],d.b[ids])
        q3=quote(i3);q10=quote(i10);qm=quote(im);qfill=quote(iff)
        np.testing.assert_allclose(qm,g.m_fill_ticks,atol=0,rtol=0)
        mask=g.fill_kind.eq('market').to_numpy()
        np.testing.assert_allclose(qfill[mask],g.fill_ticks.to_numpy()[mask],atol=0,rtol=0)
        g=g.copy()
        g['wait_to3_cost_bp']=side*(q3-qm)/p0*10000
        g['signal3_delay_cost_bp']=side*(qfill-q3)/p0*10000
        g['from3_to10_cost_bp']=side*(q10-q3)/p0*10000
        g['fallback_delay_cost_bp']=side*(qfill-q10)/p0*10000
        g['spread0_ticks']=d.a[i0]-d.b[i0];g['spread3_ticks']=d.a[i3]-d.b[i3]
        g['abs_observed_move_ticks']=np.abs(d.p[i3]-p0)
        g['observed_move_ticks']=d.p[i3]-p0
        z=g.stage.eq('market3')
        np.testing.assert_allclose(g.loc[z,'wait_to3_cost_bp']+g.loc[z,'signal3_delay_cost_bp'],g.loc[z,'extra_cost_bp'],atol=1e-12)
        z=g.stage.eq('market10')
        np.testing.assert_allclose(g.loc[z,'wait_to3_cost_bp']+g.loc[z,'from3_to10_cost_bp']+g.loc[z,'fallback_delay_cost_bp'],g.loc[z,'extra_cost_bp'],atol=1e-12)
        enriched.append(g)
    o=pd.concat(enriched,ignore_index=True)
    o.to_parquet(OUT/'原方案成交及时间分解.parquet',index=False)
    return o


def distributions(o):
    counts=[];stats=[];curves=[];qq=[];hist=[];landmarks=[];daily_landmarks=[]
    for scope in SCOPES:
        for side in ['both','buy','sell']:
            g=select_scope(o,scope,side)
            cohorts={'limit_branch':g[g.limit_branch], 'activated_new_limit':g[g.limit3_activated]}
            for cohort,b in cohorts.items():
                value='wait_from3' if cohort=='limit_branch' else 'active_wait'
                event=b.limit_event.to_numpy(bool);w=b.weight.to_numpy();values=b[value].to_numpy()
                weight=float(w.sum());frac=float(w[event].sum()/weight)
                counts.append(dict(scope=scope,side=side,cohort=cohort,all_orders=len(g),all_tasks=g[TASK].drop_duplicates().shape[0],
                    cohort_orders=len(b),limit_fills=int(event.sum()),censored=int((~event).sum()),days=b.trade_date.nunique(),
                    cohort_share=float(weight/g.weight.sum()),limit_fill_rate=frac,censor_rate=1-frac,
                    submitted3=int(g.limit3_submitted.sum()),old_fill_before_activation=int((g.limit3_submitted&~g.limit3_activated).sum())))
                km=weighted_km(np.round(values,9),event,w)
                for row in km.to_dict('records'):curves.append(dict(scope=scope,side=side,cohort=cohort,**row))
                ev=b[b.limit_event].copy()
                for metric in [value,'elapsed_seconds']:
                    stat=normal_diagnostics(ev[metric],ev.weight)
                    intervals=block_moment_intervals(ev,metric,dates=g.trade_date.unique())
                    stats.append(dict(scope=scope,side=side,cohort=cohort,metric=metric,days=ev.trade_date.nunique(),**stat,**intervals))
                if cohort=='activated_new_limit':
                    p=np.linspace(.01,.99,99);x=weighted_quantile(ev[value],ev.weight,p)
                    for prob,actual in zip(p,x):qq.append(dict(scope=scope,side=side,probability=prob,theoretical_z=NORMAL.inv_cdf(prob),actual_seconds=float(actual)))
                    edges=np.arange(-.25,9.26,.5);bins=np.histogram(ev[value],bins=edges,weights=ev.weight)[0]/ev.weight.sum()
                    nd=normal_diagnostics(ev[value],ev.weight)
                    for a,z,v in zip(edges[:-1],edges[1:],bins):
                        ref=NORMAL.cdf((z-nd['mean'])/nd['sd'])-NORMAL.cdf((a-nd['mean'])/nd['sd'])
                        hist.append(dict(scope=scope,side=side,left=a,right=z,observed_probability=float(v),normal_probability=ref))
                else:
                    for clock in [4,5,6,7,8,9,10]:
                        filled=b.limit_event & b.elapsed_seconds.le(clock+1e-10)
                        remaining=b.elapsed_seconds.gt(clock+1e-10)
                        late=remaining & b.limit_event
                        final_censor=remaining & ~b.limit_event
                        denom=float(b.weight[remaining].sum())
                        landmarks.append(dict(scope=scope,side=side,clock=clock,
                            filled_fraction_of_branch=float(b.weight[filled].sum()/weight),
                            retained_eventual_limit_fills=float(b.weight[filled].sum()/b.weight[b.limit_event].sum()),
                            still_waiting_fraction=float(b.weight[remaining].sum()/weight),
                            later_limit_fraction_among_waiting=float(b.weight[late].sum()/denom),
                            eventual_fallback_fraction_among_waiting=float(b.weight[final_censor].sum()/denom)))
                        for date,z in b.groupby('trade_date'):
                            good=z.limit_event&z.elapsed_seconds.le(clock+1e-10)
                            daily_landmarks.append(dict(scope=scope,side=side,clock=clock,trade_date=date,
                                cohort_mass=float(z.weight.sum()),filled_mass=float(z.weight[good].sum()),
                                eventual_fill_mass=float(z.weight[z.limit_event].sum())))
    files={'群体与删失比例':counts,'正态形状与分位统计':stats,'含删失成交分布':curves,'正态QQ数据':qq,
           '成交时间直方图':hist,'等待时点保留比例':landmarks,'逐日等待保留比例':daily_landmarks}
    for name,rows in files.items():pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    return files


def minute_attribution(o):
    daily=[];stages=[];timing=[];bins=[];accuracy=[]
    for scope in SCOPES:
        for side in ['both','buy','sell']:
            g=select_scope(o,scope,side)
            for date,z in g.groupby('trade_date',sort=True):
                daily.append(dict(scope=scope,side=side,trade_date=date,orders=len(z),cost_bp=float(z.cost_bp.mean()),
                    market_cost_bp=float(z.market_cost_bp.mean()),saving_bp=float(z.saving_bp.mean()),
                    elapsed_seconds=float(z.elapsed_seconds.mean())))
                for stage in STAGES:
                    mask=z.stage.eq(stage)
                    stages.append(dict(scope=scope,side=side,trade_date=date,stage=stage,rate=float(mask.mean()),
                        saving_contribution_bp=float(z.saving_bp.where(mask,0).mean())))
                for stage,metrics in [('market3',['wait_to3_cost_bp','signal3_delay_cost_bp']),
                                      ('market10',['wait_to3_cost_bp','from3_to10_cost_bp','fallback_delay_cost_bp'])]:
                    mask=z.stage.eq(stage)
                    for metric in metrics:
                        timing.append(dict(scope=scope,side=side,trade_date=date,stage=stage,component=metric,
                            rate=float(mask.mean()),extra_contribution_bp=float(z[metric].where(mask,0).mean())))
            # Prediction stats count each task once; no fill- or label-based execution filtering.
            if side=='both':
                for group_name,q in [('all_tasks',g.drop_duplicates(TASK)),('market3_orders',g[g.stage.eq('market3')])]:
                    for date,z in q.groupby('trade_date',sort=True):
                        valid=z.cum_1.notna();v=z[valid];truth=np.sign(v.cum_1);hit=v.signal.eq(truth)
                        accuracy.append(dict(scope=scope,group=group_name,trade_date=date,available=len(z),valid=len(v),
                            accuracy=float(hit.mean()),flat_rate=float(truth.eq(0).mean()),
                            abs_move1_ticks=float(v.cum_1.abs().mean()),abs_move3_observed_ticks=float(z.abs_observed_move_ticks.mean()),
                            spread0_ticks=float(z.spread0_ticks.mean()),spread3_ticks=float(z.spread3_ticks.mean())))
    g=o[o.nominal_second.lt(60)].copy();g['bin_start']=(g.nominal_second//10)*10
    for side in ['both','buy','sell']:
        q=g if side=='both' else g[g.direction.eq(1 if side=='buy' else -1)]
        for (date,start),z in q.groupby(['trade_date','bin_start'],sort=True):
            bins.append(dict(trade_date=date,side=side,bin_start=int(start),orders=len(z),cost_bp=float(z.cost_bp.mean()),
                market_cost_bp=float(z.market_cost_bp.mean()),saving_bp=float(z.saving_bp.mean()),
                **{stage+'_saving_bp':float(z.saving_bp.where(z.stage.eq(stage),0).mean()) for stage in STAGES}))
    daily=pd.DataFrame(daily);stages=pd.DataFrame(stages);timing=pd.DataFrame(timing);bins=pd.DataFrame(bins);accuracy=pd.DataFrame(accuracy)
    for name,f in [('逐日执行范围',daily),('逐日成交分支贡献',stages),('逐日市价时间成本分解',timing),('首分钟10秒段逐日结果',bins),('逐日预测与行情诊断',accuracy)]:
        f.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    summary=[]
    for (scope,side),g in daily.groupby(['scope','side']):
        r=dict(scope=scope,side=side,days=len(g),orders=int(g.orders.sum()))
        for k in ['cost_bp','market_cost_bp','saving_bp','elapsed_seconds']:r[k]=float(g[k].mean())
        lo,hi=date_block_interval(g.sort_values('trade_date').saving_bp,seed=20260928)
        r.update(low_bp=float(lo),high_bp=float(hi));summary.append(r)
    pd.DataFrame(summary).to_csv(OUT/'范围成本汇总.csv',index=False,encoding='utf-8-sig')
    contrasts=[]
    for side in ['both','buy','sell']:
        wide=daily[daily.side.eq(side)].pivot(index='trade_date',columns='scope',values='saving_bp').sort_index()
        delta=wide.first_minute-wide.rest_59
        lo,hi=date_block_interval(delta,seed=20260928)
        contrasts.append(dict(side=side,first_minus_rest_saving_bp=float(delta.mean()),low_bp=float(lo),high_bp=float(hi),
                              negative_dates=int(delta.lt(0).sum()),dates=len(delta)))
    pd.DataFrame(contrasts).to_csv(OUT/'首分钟相对随后59分钟_配对差异.csv',index=False,encoding='utf-8-sig')
    st=stages.groupby(['scope','side','stage'],as_index=False)[['rate','saving_contribution_bp']].mean()
    st['conditional_saving_bp']=st.saving_contribution_bp/st.rate
    st.to_csv(OUT/'成交分支贡献汇总.csv',index=False,encoding='utf-8-sig')
    ti=timing.groupby(['scope','side','stage','component'],as_index=False)[['rate','extra_contribution_bp']].mean()
    ti['conditional_extra_cost_bp']=ti.extra_contribution_bp/ti.rate
    ti.to_csv(OUT/'市价时间成本分解.csv',index=False,encoding='utf-8-sig')
    bin_summary=[]
    for (side,start),g in bins.groupby(['side','bin_start']):
        lo,hi=date_block_interval(g.sort_values('trade_date').saving_bp,seed=20260928)
        bin_summary.append(dict(side=side,bin_start=int(start),orders=int(g.orders.sum()),
            **{k:float(g[k].mean()) for k in ['cost_bp','market_cost_bp','saving_bp']+[s+'_saving_bp' for s in STAGES]},low_bp=float(lo),high_bp=float(hi)))
    pd.DataFrame(bin_summary).to_csv(OUT/'首分钟10秒段汇总.csv',index=False,encoding='utf-8-sig')
    accuracy.groupby(['scope','group'],as_index=False).agg(available=('available','sum'),valid=('valid','sum'),
        accuracy=('accuracy','mean'),flat_rate=('flat_rate','mean'),abs_move1_ticks=('abs_move1_ticks','mean'),
        abs_move3_observed_ticks=('abs_move3_observed_ticks','mean'),spread0_ticks=('spread0_ticks','mean'),spread3_ticks=('spread3_ticks','mean')).to_csv(OUT/'预测与行情诊断.csv',index=False,encoding='utf-8-sig')
    # Exact symmetric mix/severity decomposition, descriptive rather than causal.
    mix=[]
    for side in ['both','buy','sell']:
        a=st[st.scope.eq('first_minute')&st.side.eq(side)].set_index('stage')
        b=st[st.scope.eq('rest_59')&st.side.eq(side)].set_index('stage')
        for stage in STAGES:
            p1,p2=a.loc[stage,'rate'],b.loc[stage,'rate'];m1,m2=a.loc[stage,'conditional_saving_bp'],b.loc[stage,'conditional_saving_bp']
            mixpart=(p1-p2)*(m1+m2)/2;severity=(m1-m2)*(p1+p2)/2
            diff=a.loc[stage,'saving_contribution_bp']-b.loc[stage,'saving_contribution_bp']
            np.testing.assert_allclose(mixpart+severity,diff,atol=1e-12)
            mix.append(dict(side=side,stage=stage,first_rate=p1,rest_rate=p2,first_conditional_saving_bp=m1,rest_conditional_saving_bp=m2,
                            mix_difference_bp=mixpart,severity_difference_bp=severity,total_difference_bp=diff))
    pd.DataFrame(mix).to_csv(OUT/'首分钟差异_占比与单笔代价分解.csv',index=False,encoding='utf-8-sig')
    for (scope,side),g in st.groupby(['scope','side']):
        ref=next(r for r in summary if r['scope']==scope and r['side']==side)
        np.testing.assert_allclose([g.rate.sum(),g.saving_contribution_bp.sum()],[1,ref['saving_bp']],atol=1e-12)
    return pd.DataFrame(summary)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[RAW,SOURCE/'全部逐任务成交.parquet',SOURCE/'全部下单记录.parquet',HORIZON/'三模型逐任务信号.parquet',SOURCE/'冻结模型.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    save_json(OUT/'分析口径.json',dict(input_sha256=hashes,baseline='C3 W3H1 19/19 ticks',scopes=SCOPES,
        primary='activated T3 replacement limits: actual activation origin; event=passive fill, right-censor=market fallback',
        supplementary='all surviving T3 passive decisions, including kept initial limits and fills before replacement activation',
        weights='equal day and equal sides on full task cohort; then condition on the studied subgroup',
        shape='conditional successful-fill histogram/QQ/moments; no IID normality p value; censored distribution KM',
        tail='no extrapolation past observation; 10s market fallback is not a passive fill',
        bootstrap='5000 circular date blocks of5; descriptive, prior research not selection adjusted',
        strategy_change=False))
    o=enrich();distributions(o);s=minute_attribution(o)
    old=pd.read_csv(SOURCE/'成本与节约汇总.csv')
    for scope,minutes in [('first_minute',1),('first_hour',60)]:
        for side in ['both','buy','sell']:
            a=s[s.scope.eq(scope)&s.side.eq(side)].iloc[0]
            b=old[old.minutes.eq(minutes)&old.side.eq(side)&old.strategy.eq('C3')].iloc[0]
            np.testing.assert_allclose([a.cost_bp,a.saving_bp,a.elapsed_seconds],
                [b.cost_bp,b.saving_vs_market_bp,b.elapsed_seconds],atol=1e-12,rtol=0)
    assert o[TASK].drop_duplicates().shape[0]==62899 and len(o)==125798 and o.trade_date.nunique()==18
    assert len(o[o.nominal_second.lt(60)])==2160
    assert all(sha(p)==hashes[str(p.relative_to(ROOT))] for p in inputs)
    save_json(OUT/'验收.json',dict(orders=len(o),tasks=o[TASK].drop_duplicates().shape[0],dates=o.trade_date.nunique(),
        costs_reproduced=True,market_time_decomposition_exact=True,stage_identity_exact=True,input_sha256=hashes,inputs_unchanged=True))
    print(s.to_string(index=False),flush=True)
    print(pd.read_csv(OUT/'群体与删失比例.csv').query('side == "both"').to_string(index=False),flush=True)
    print(pd.read_csv(OUT/'正态形状与分位统计.csv').query('side == "both" and metric == "active_wait"').to_string(index=False),flush=True)


if __name__=='__main__':main()
