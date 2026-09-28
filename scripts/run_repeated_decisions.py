"""Frozen W3/H1 model: add T+6 and T+9 decisions on trailing three seconds."""
from pathlib import Path
import os, sys, json
sys.path = [p for p in sys.path if not p.replace('\\','/').endswith('.cache/model-packages')]
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ.setdefault(name,'1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from utils.opening_ic import ICConfig,SessionData
from utils.opening_schedule import ScheduleConfig
from utils.opening_task_window import bounded_features
from utils.opening_anchored import predict
from utils.opening_observation import FEATURES
from utils.opening_two_stage import date_block_interval
from utils.opening_repeated import STRATEGIES,NAMES,simulate_repeated
from scripts.run_horizon_execution import OUT as SOURCE,RAW,sha,save_json,feature_reason,PCOLS,FIELDS
from scripts.run_second_ranges import independent_fill

OUT=SOURCE.parent/'repeated_decisions'
TASK=['trade_date','nominal_second']
KEY=TASK+['direction']
STAGES=['immediate_market','initial_limit','limit3','limit6','limit9','market3','market6','market9','market10']


def stage_name(r):
    if r['fill_stage']=='initial':return 'initial_limit'
    if r['fill_stage']=='deadline':return 'market10'
    return r['fill_kind']+r['fill_stage'].replace('signal','')


def worker(job):
    date,frame,base,oldsign,model=job
    cfg=ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,task_range_seconds=3600,observation_seconds=(3.,)))
    d=SessionData(frame,pd.Timestamp(date+' 09:30',tz='Asia/Shanghai'),cfg)
    assert max(model['train_dates'])<date and date in model['test_dates']
    refs=base.set_index(['nominal_second','direction','strategy'])
    oldsign=oldsign.set_index('nominal_second')
    tasks=base[base.strategy.eq('fixed_1s') & base.direction.eq(1)].sort_values('nominal_second')
    feature_cache={}; signal_rows=[]; order_rows=[]; invalid=[]; traces=[]; command_rows=[]
    causal=0; checked=0; single_exact=0; market_exact=0; prefix_checks=0
    for count,r in enumerate(tasks.itertuples(index=False)):
        t=float(r.task_seconds); nominal=int(r.nominal_second); signals={}
        for s in [3,6,9]:
            # Preserve the exact observed task origin for the first window.
            # (t+3)-3 can differ from t by floating-point rounding and move a
            # boundary snapshot outside searchsorted's inclusive interval.
            start=t+(s-3)
            if start not in feature_cache:
                feature_cache[start]=bounded_features(d,start,3)
            f=feature_cache[start];x=np.array([f[k] for k in FEATURES])
            p=predict(model,x[None,:])[0] if np.isfinite(x).all() else np.full(3,np.nan)
            signals[s]=int(p.argmax()-1) if np.isfinite(p).all() else None
            if s==3:
                np.testing.assert_allclose(p,oldsign.loc[nominal,PCOLS].to_numpy(float),atol=1e-13,rtol=1e-13,
                                           err_msg=f'{date}/{nominal}/T={t}/start={start}')
                assert signals[s]==int(oldsign.loc[nominal,'signal'])
            signal_rows.append(dict(trade_date=date,nominal_second=nominal,task_seconds=t,decision_second=s,
                window_start=start,window_end=t+s,fold=model['fold'],signal=signals[s],
                **dict(zip(PCOLS,p)),**{k:f[k] for k in FEATURES},missing_reason=feature_reason(d,start,f)))
            if count%1000==0:
                sliced=d.frame[(d.times>=start)&(d.times<=start+3)].copy()
                changed=d.frame.copy();outside=(d.times<start)|(d.times>start+3)
                for col in ['LastPrice','BidPrice1','AskPrice1']:changed.loc[outside,col]+=20
                changed.loc[outside,'BidVolume1']+=777;changed.loc[outside,'AskVolume1']+=333
                for sub in [sliced,changed]:
                    f2=bounded_features(SessionData(sub,d.open_time,cfg),start,3)
                    np.testing.assert_allclose(x,[f2[k] for k in FEATURES],atol=0,rtol=0,equal_nan=True)
                    causal+=1
        taskrows=[]; taskerrors=[]
        for side in [1,-1]:
            ref=refs.loc[(nominal,side,'M')]
            i=d.source_index(t);j=int(np.searchsorted(d.times,t,'right'))+1
            assert j<d.n and all(d.valid[g][i:j+1].all() for g in ['price','book','queue','volume'])
            assert d.time_edge[i+1:j+1].all() and d.edge['volume'][i+1:j+1].all()
            price=float(d.a[j] if side==1 else d.b[j])
            assert (d.qa[j] if side==1 else d.qb[j])>=1
            assert ref.fill_seconds==d.times[j] and ref.fill_ticks==price
            assert ref.cost_bp==side*(price-d.p[i])/d.p[i]*10000
            market_exact+=1
            common=dict(trade_date=date,nominal_second=nominal,task_seconds=t,direction=side,fold=model['fold'],market_cost_bp=float(ref.cost_bp))
            taskrows.append(dict(**common,strategy='M',**{k:ref[k] for k in FIELDS},stage='immediate_market',
                submissions=1,replacements=0,decisions_used='',missing_decisions='',same_price_count=0,deadline_submitted=False))
            outcomes={}
            for code,decisions in STRATEGIES.items():
                z=simulate_repeated(d,t,side,signals,decisions)
                trace=z.pop('trace')
                if z['status']!='filled':
                    taskerrors.append(f'{code}/{side}:{z["reason"]}')
                    continue
                ii,pp=independent_fill(d,trace,side)
                assert d.times[ii]==z['fill_seconds'] and pp==z['fill_ticks']
                checked+=1
                if code=='C3':
                    old=refs.loc[(nominal,side,'fixed_1s')]
                    for k in FIELDS:
                        actual=z[k].replace('signal3','signal') if k=='fill_stage' else z[k]
                        assert actual==old[k],(date,nominal,side,k)
                    single_exact+=1
                for e in trace:
                    if e['event']=='submit':
                        if e['kind']=='limit':
                            assert e['limit_ticks']==d.p[d.source_index(e['time'])]-side*19
                        if e['stage']=='deadline':assert e['time']==t+10
                        command_rows.append(dict(**common,strategy=code,submitted_seconds=e['time'],
                            decision_second=0 if e['stage']=='initial' else 10 if e['stage']=='deadline' else int(e['stage'][6:]),
                            kind=e['kind'],command_stage=e['stage'],
                            limit_ticks=e['limit_ticks'],arrival_seconds=float(d.times[e['arrival_index']]),
                            realized_arrival=any(a['event']=='arrival' and a['submitted']==e['time'] for a in trace)))
                if count<2 or (len(traces)<40 and any(e['event']=='deadline_pending_limit' and e['count']>0 for e in trace)):
                    traces.append(dict(nominal_second=nominal,direction=side,strategy=code,trace=trace))
                outcomes[code]=z
                taskrows.append(dict(**common,strategy=code,**{k:z[k] for k in FIELDS},stage=stage_name(z),
                    **{k:z[k] for k in ['submissions','replacements','same_price_count','deadline_submitted']},
                    decisions_used=','.join(map(str,z['decisions_used'])),missing_decisions=','.join(map(str,z['missing_decisions']))))
            for left,right,end in [('C3','C36',6),('C36','C369',9)]:
                if left in outcomes and right in outcomes and min(outcomes[left]['elapsed_seconds'],outcomes[right]['elapsed_seconds'])<=end:
                    for k in ['fill_seconds','fill_ticks','fill_stage','fill_kind','cost_bp']:assert outcomes[left][k]==outcomes[right][k]
                    prefix_checks+=1
        if taskerrors or len(taskrows)!=8:
            invalid.append(dict(trade_date=date,nominal_second=nominal,reason=';'.join(taskerrors)))
        else:order_rows.extend(taskrows)
    folder=OUT/'逐日回放'
    pd.DataFrame(order_rows).to_parquet(folder/f'{date}_orders.parquet',index=False)
    pd.DataFrame(signal_rows).to_parquet(folder/f'{date}_signals.parquet',index=False)
    pd.DataFrame(command_rows).to_parquet(folder/f'{date}_commands.parquet',index=False)
    save_json(folder/f'{date}_traces.json',traces)
    receipt=dict(trade_date=date,original_tasks=len(tasks),common_tasks=len(order_rows)//8,
        independently_checked_orders=checked,single_decision_exact=single_exact,market_exact=market_exact,
        causal_checks=causal,same_prefix_fills=prefix_checks,exclusions=invalid)
    save_json(folder/f'{date}_audit.json',receipt)
    return {k:v for k,v in receipt.items() if k!='exclusions'}


def analyze(orders):
    metric=['cost_bp','market_cost_bp','elapsed_seconds','submissions','replacements']
    daily=[];decomp=[];windows=[]
    for minutes in [1,60]:
        for side,name in [(0,'both'),(1,'buy'),(-1,'sell')]:
            scope=orders[orders.nominal_second.lt(minutes*60)]
            if side:scope=scope[scope.direction.eq(side)]
            for (date,strategy),g in scope.groupby(['trade_date','strategy'],sort=True):
                row=dict(trade_date=date,strategy=strategy,minutes=minutes,side=name,fold=int(g.fold.iloc[0]),
                         orders=len(g),tasks=g.nominal_second.nunique())
                row.update({k:float(g[k].mean()) for k in metric})
                row.update(saving_vs_market_bp=float((g.market_cost_bp-g.cost_bp).mean()),
                           early_fill=float(g.elapsed_seconds.le(3+1e-10).mean()),
                           fallback_rate=float(g.stage.eq('market10').mean()))
                daily.append(row)
                saving=g.market_cost_bp-g.cost_bp
                for stage in STAGES:
                    mask=g.stage.eq(stage)
                    decomp.append(dict(trade_date=date,strategy=strategy,minutes=minutes,side=name,stage=stage,
                        rate=float(mask.mean()),cost_contribution_bp=float(g.cost_bp.where(mask,0).mean()),
                        saving_contribution_bp=float(saving.where(mask,0).mean())))
                for label,mask in [('0_3',g.elapsed_seconds.le(3+1e-10)),
                                   ('3_6',g.elapsed_seconds.gt(3+1e-10)&g.elapsed_seconds.le(6+1e-10)),
                                   ('6_9',g.elapsed_seconds.gt(6+1e-10)&g.elapsed_seconds.le(9+1e-10)),
                                   ('after9',g.elapsed_seconds.gt(9+1e-10))]:
                    windows.append(dict(trade_date=date,strategy=strategy,minutes=minutes,side=name,time_bin=label,
                        rate=float(mask.mean()),saving_contribution_bp=float(saving.where(mask,0).mean())))
    daily=pd.DataFrame(daily);decomp=pd.DataFrame(decomp);windows=pd.DataFrame(windows)
    summary=[];pairs=[]
    for (minutes,side),g in daily.groupby(['minutes','side']):
        pivot=g.pivot(index='trade_date',columns='strategy',values='cost_bp').sort_index()
        for strategy,z in g.groupby('strategy'):
            row=dict(minutes=int(minutes),side=side,strategy=strategy,days=len(z),orders=int(z.orders.sum()),tasks=int(z.tasks.sum()))
            row.update({k:float(z[k].mean()) for k in metric+['saving_vs_market_bp','early_fill','fallback_rate']})
            for ref,key in [('M','saving_vs_market'),('C3','saving_vs_C3')]:
                delta=pivot[ref]-pivot[strategy];lo,hi=date_block_interval(delta,seed=20260928)
                row.update({key+'_bp':float(delta.mean()),key+'_low':float(lo),key+'_high':float(hi)})
            summary.append(row)
        for candidate,reference in [('C36','C3'),('C369','C3'),('C369','C36')]:
            delta=pivot[reference]-pivot[candidate];lo,hi=date_block_interval(delta,seed=20260928)
            pairs.append(dict(minutes=int(minutes),side=side,candidate=candidate,reference=reference,
                saving_bp=float(delta.mean()),low_bp=float(lo),high_bp=float(hi),
                positive_days=int((delta>1e-12).sum()),negative_days=int((delta< -1e-12).sum()),zero_days=int((delta.abs()<=1e-12).sum())))
    return daily,pd.DataFrame(summary),pd.DataFrame(pairs),decomp,windows


def finish():
    folder=OUT/'逐日回放'
    orders=pd.concat([pd.read_parquet(p) for p in sorted(folder.glob('*_orders.parquet'))],ignore_index=True)
    signals=pd.concat([pd.read_parquet(p) for p in sorted(folder.glob('*_signals.parquet'))],ignore_index=True)
    commands=pd.concat([pd.read_parquet(p) for p in sorted(folder.glob('*_commands.parquet'))],ignore_index=True)
    clock=commands.command_stage.map(lambda s:0 if s=='initial' else 10 if s=='deadline' else int(s[6:]))
    np.testing.assert_allclose(commands.decision_second,clock,atol=1e-9,rtol=0)
    commands['decision_second']=clock.astype(int)
    receipts=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('*_audit.json'))]
    assert len(receipts)==18
    assert not orders.duplicated(KEY+['strategy']).any()
    assert orders.groupby(TASK).size().eq(8).all()
    np.testing.assert_allclose(orders.direction*(orders.fill_ticks-orders.initial_ticks)/orders.initial_ticks*10000,orders.cost_bp,atol=1e-12)
    common=orders[TASK].drop_duplicates().assign(common=True)
    signals=signals.merge(common,on=TASK,how='left',validate='many_to_one');signals['common']=signals.common.eq(True)
    np.testing.assert_array_equal(signals.window_start,signals.task_seconds+(signals.decision_second-3))
    np.testing.assert_array_equal(signals.window_start+3,signals.window_end)
    save_json(OUT/'时间边界验收.json',dict(signal_rows=len(signals),start_exact=True,end_exact=True,
        repair='Preserve origin as t+(s-3), not (t+s)-3; all stored boundaries checked'))
    for m in json.loads((OUT/'冻结模型.json').read_text(encoding='utf-8')):
        z=signals[signals.fold.eq(m['fold'])&signals.signal.notna()]
        np.testing.assert_allclose(predict(m,z[FEATURES].to_numpy()),z[PCOLS],atol=1e-13,rtol=1e-13)
        np.testing.assert_array_equal(z[PCOLS].to_numpy().argmax(axis=1)-1,z.signal)
    for name,f in [('全部逐任务成交',orders),('三时点信号与特征',signals),('全部下单记录',commands)]:f.to_parquet(OUT/(name+'.parquet'),index=False)
    daily,summary,pairs,decomp,windows=analyze(orders)
    for name,f in [('逐日成本',daily),('成本与节约汇总',summary),('配对改善',pairs),('逐日成交方式贡献',decomp),('逐日成交时段贡献',windows)]:
        f.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    dc=decomp.groupby(['minutes','side','strategy','stage'],as_index=False)[['rate','cost_contribution_bp','saving_contribution_bp']].mean()
    dc['conditional_cost_bp']=dc.cost_contribution_bp/dc.rate.replace(0,np.nan)
    dc['conditional_saving_bp']=dc.saving_contribution_bp/dc.rate.replace(0,np.nan)
    dc.to_csv(OUT/'成交方式贡献汇总.csv',index=False,encoding='utf-8-sig')
    windows.groupby(['minutes','side','strategy','time_bin'],as_index=False)[['rate','saving_contribution_bp']].mean().to_csv(OUT/'成交时段贡献汇总.csv',index=False,encoding='utf-8-sig')
    for (minutes,side,strategy),g in dc.groupby(['minutes','side','strategy']):
        ref=summary[summary.minutes.eq(minutes)&summary.side.eq(side)&summary.strategy.eq(strategy)].iloc[0]
        np.testing.assert_allclose([g.rate.sum(),g.cost_contribution_bp.sum(),g.saving_contribution_bp.sum()],[1,ref.cost_bp,ref.saving_vs_market_bp],atol=1e-12)
    for (minutes,strategy),g in summary.groupby(['minutes','strategy']):
        g=g.set_index('side')
        for k in ['cost_bp','elapsed_seconds','fallback_rate','saving_vs_market_bp']:
            np.testing.assert_allclose((g.loc['buy',k]+g.loc['sell',k])/2,g.loc['both',k],atol=1e-12)
    excluded=pd.DataFrame([r for a in receipts for r in a['exclusions']],columns=TASK+['reason'])
    excluded.to_csv(OUT/'新增排除任务.csv',index=False,encoding='utf-8-sig')
    missing=signals[signals.signal.isna()].groupby(['decision_second','missing_reason']).size().rename('tasks').reset_index()
    missing.to_csv(OUT/'信号缺失原因.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{k:r[k] for k in ['trade_date','original_tasks','common_tasks']} for r in receipts]).to_csv(OUT/'任务覆盖.csv',index=False,encoding='utf-8-sig')
    daily.groupby(['minutes','side','strategy','fold'],as_index=False)[['cost_bp','saving_vs_market_bp','fallback_rate']].mean().to_csv(OUT/'四折结果.csv',index=False,encoding='utf-8-sig')
    save_json(OUT/'撮合验收.json',dict(per_day=receipts,original_tasks=sum(r['original_tasks'] for r in receipts),
        common_tasks=len(common),orders=len(orders),no_future_label_filter=True,missing_signal='keep current limit; next decision or T10 fallback',
        cost_from_prices_verified=True,signal_parameters_reproduced=True,contribution_identity=True))
    print(summary[summary.side.eq('both')][['minutes','strategy','tasks','cost_bp','saving_vs_market_bp','saving_vs_C3_bp','elapsed_seconds','fallback_rate']].to_string(index=False),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'逐日回放').mkdir(exist_ok=True)
    sources=[RAW,SOURCE/'模型参数.json',SOURCE/'三模型逐任务信号.parquet',SOURCE/'四组逐任务成交.parquet',ROOT/'utils/opening_two_stage.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    rules=dict(observation=3,forecast=1,initial_offset=19,signal_offset=19,deadline=10,delay_snapshots=2,
        strategies=STRATEGIES,model='previous frozen fixed_1s per date; no refit',
        window='T..T3, T3..T6, T6..T9; no earlier history',missing='keep current order and revisit next decision; no imputation or exclusion',
        pending_at10='retain in-flight T9 limit, queue fallback at T10; if market pending do not send duplicate',
        cohort='previous62899 tasks; paired both sides and all strategies, no future label filter',
        scopes=[1,60],aggregation='equal18dates; equalbuy/sell',bootstrap=dict(block=5,repeats=5000,seed=20260928),input_sha256=hashes)
    reg=OUT/'实验口径.json'
    if reg.exists():assert json.loads(reg.read_text(encoding='utf-8'))==json.loads(json.dumps(rules))
    else:save_json(reg,rules)
    models=[m for m in json.loads((SOURCE/'模型参数.json').read_text(encoding='utf-8')) if m['strategy']=='fixed_1s']
    save_json(OUT/'冻结模型.json',models)
    if '--finish-only' not in sys.argv:
        base=pd.read_parquet(SOURCE/'四组逐任务成交.parquet');base=base[base.strategy.isin(['M','fixed_1s'])]
        sig=pd.read_parquet(SOURCE/'三模型逐任务信号.parquet');sig=sig[sig.strategy.eq('fixed_1s')&sig.common]
        assert len(sig)==62899
        modelmap={date:m for m in models for date in m['test_dates']}
        raw=pd.read_parquet(RAW);raw['source_row']=np.arange(len(raw));jobs=[]
        for date,b in base.groupby('trade_date',sort=True):
            if (OUT/'逐日回放'/f'{date}_audit.json').exists():continue
            start=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai')
            f=raw[(raw.Datetime>=start)&(raw.Datetime<=start+pd.Timedelta(seconds=3620))]
            f=f[['Datetime','LastPrice','BidPrice1','AskPrice1','BidVolume1','AskVolume1','Volume','source_row']].copy()
            jobs.append((date,f,b.copy(),sig[sig.trade_date.eq(date)].copy(),modelmap[date]))
        with ProcessPoolExecutor(max_workers=2) as pool:
            for r in pool.map(worker,jobs):print(json.dumps(r),flush=True)
    finish()
    assert all(sha(p)==hashes[str(p.relative_to(ROOT))] for p in sources)
    save_json(OUT/'输入保持不变.json',dict(all_verified=True,input_sha256=hashes))


if __name__=='__main__':main()
