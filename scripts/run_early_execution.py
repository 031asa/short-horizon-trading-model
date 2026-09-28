"""Pre-3s adverse-signal checks on opening-minute tasks, frozen W1/W2/W3 models."""
from scripts.run_repeated_decisions import ROOT,OUT as BASE,RAW,TASK,KEY,FIELDS,FEATURES,PCOLS,sha,save_json,stage_name
from scripts.run_horizon_execution import OBS
from scripts.run_second_ranges import independent_fill
from utils.opening_ic import ICConfig,SessionData
from utils.opening_schedule import ScheduleConfig
from utils.opening_task_window import bounded_features
from utils.opening_anchored import predict
from utils.opening_repeated import simulate_repeated
from utils.opening_two_stage import date_block_interval
import json,sys,zipfile,hashlib
import numpy as np
import pandas as pd

OUT=BASE.parent/'early_decisions'
CODES=['M','C3','E1','E2','E1_10','E2_10','M10']
NAMES={'M':'全部立即市价','C3':'原3秒方案','E1':'全分钟加1秒检查','E2':'全分钟加2秒检查',
       'E1_10':'仅前10秒任务加1秒检查','E2_10':'仅前10秒任务加2秒检查','M10':'前10秒任务立即市价'}
SCOPES={'first10':(0,10),'rest50':(10,60),'first_minute':(0,60)}
SCOPE_NAMES={'first10':'最前10秒任务','rest50':'随后50秒任务','first_minute':'开盘首分钟任务'}
NUMERIC=['initial_ticks','fill_seconds','fill_ticks','cost_ticks','cost_bp','elapsed_seconds']


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[RAW,OBS/'模型参数.json',OBS/'逐任务预测.parquet',BASE/'全部逐任务成交.parquet',ROOT/'utils/opening_two_stage.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    rules=dict(strategies=NAMES,primary='opening first60 nominal task seconds,18dates,1080tasks,both sides',
        early_checks='E1 at T+1 using W1/H1 model; E2 at T+2 using W2/H1 model; adverse argmax only sends market, otherwise keep initial order',
        continuation='if still unfilled and no market pending, original W3/H1 decision at T+3; deadline T+10 unchanged',
        early_missing='keep current order then original T+3; never exclude task or fill missing features',
        first10='known nominal task seconds 0..9; policy chosen at task creation, later tasks remain original C3',
        offsets='initial and T+3 passive19; no early limit repricing',delay_snapshots=2,
        pending='old limits remain fillable until market arrival; pending initial limit and subsequent market retain submission order',
        selection='fixed candidates, no threshold scan, no automatic replacement; first10 period is historical hypothesis',
        bootstrap='equal18dates,5day circular blocks,5000replicates,seed20260928',input_sha256=hashes)
    save_json(OUT/'实验口径.json',rules)
    models=[m for m in json.loads((OBS/'模型参数.json').read_text(encoding='utf-8')) if m['window'] in [1,2,3] and m['fold']!='final']
    save_json(OUT/'冻结模型.json',models)
    baseline=pd.read_parquet(BASE/'全部逐任务成交.parquet')
    baseline=baseline[baseline.nominal_second.lt(60)&baseline.strategy.isin(['M','C3'])].copy()
    original=baseline.set_index(KEY+['strategy'])
    saved=pd.read_parquet(OBS/'逐任务预测.parquet');saved=saved[saved.window.isin([1,2,3])].set_index(TASK+['window'])
    raw=pd.read_parquet(RAW,columns=['Datetime','LastPrice','BidPrice1','AskPrice1','BidVolume1','AskVolume1','Volume'])
    raw['source_row']=np.arange(len(raw))
    rows=[];signals=[];commands=[];traces=[];checks=0;causal=0;reproduced=0;prefix=0;future_signal=0
    cfg=ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,task_range_seconds=60,observation_seconds=(1.,2.,3.)))
    for date,z in baseline.groupby('trade_date',sort=True):
        start=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai')
        frame=raw[raw.Datetime.ge(start)&raw.Datetime.le(start+pd.Timedelta(seconds=80))].copy()
        d=SessionData(frame,start,cfg)
        bywindow={w:next(m for m in models if m['window']==w and date in m['test_dates']) for w in [1,2,3]}
        for m in bywindow.values():assert max(m['train_dates'])<date
        tasks=z[z.strategy.eq('C3')&z.direction.eq(1)].sort_values('nominal_second')
        for r in tasks.itertuples(index=False):
            t=float(r.task_seconds);nominal=int(r.nominal_second);ss={}
            for w,m in bywindow.items():
                f=bounded_features(d,t,w);x=np.array([f[k] for k in FEATURES]);valid=np.isfinite(x).all()
                p=predict(m,x[None,:])[0] if valid else np.full(3,np.nan)
                ss[w]=int(p.argmax()-1) if valid else None
                sr=dict(trade_date=date,nominal_second=nominal,task_seconds=t,window=w,fold=m['fold'],signal=ss[w],
                    **dict(zip(PCOLS,p)),**{k:f[k] for k in FEATURES},
                    missing_features=','.join(k for k in FEATURES if not np.isfinite(f[k])))
                signals.append(sr)
                if (date,nominal,w) in saved.index:
                    old=saved.loc[(date,nominal,w)]
                    np.testing.assert_allclose(p,old[PCOLS].to_numpy(float),atol=1e-13,rtol=1e-13)
                    np.testing.assert_allclose(x,old[FEATURES].to_numpy(float),atol=0,rtol=0)
                    reproduced+=1
                if nominal%15==0:
                    clipped=frame[(d.times>=t)&(d.times<=t+w)].copy()
                    changed=frame.copy();outside=(d.times<t)|(d.times>t+w)
                    for col in ['LastPrice','BidPrice1','AskPrice1']:changed.loc[outside,col]+=20
                    changed.loc[outside,'BidVolume1']+=777;changed.loc[outside,'AskVolume1']+=333
                    for altered in [clipped,changed]:
                        f2=bounded_features(SessionData(altered,start,cfg),t,w)
                        np.testing.assert_allclose(x,[f2[k] for k in FEATURES],atol=0,rtol=0,equal_nan=True);causal+=1
            assert ss[3] is not None
            for side in [-1,1]:
                ref=original.loc[(date,nominal,side,'C3')]
                common=dict(trade_date=date,nominal_second=nominal,task_seconds=t,direction=side,fold=int(r.fold),market_cost_bp=ref.market_cost_bp)
                for code,early in [('C3',()),('E1',(1,)),('E2',(2,))]:
                    out=simulate_repeated(d,t,side,ss,decisions=(3,),early_market_seconds=early)
                    assert out['status']=='filled',(date,nominal,side,code,out['reason'])
                    idx,price=independent_fill(d,out['trace'],side)
                    assert d.times[idx]==out['fill_seconds'] and price==out['fill_ticks'];checks+=1
                    assert out['elapsed_seconds']<=ref.elapsed_seconds+1e-10
                    if code=='C3':
                        for k in FIELDS:assert out[k]==ref[k],(date,nominal,side,k)
                    if early and (ss[early[0]] is None or side*ss[early[0]]<=0 or ref.elapsed_seconds<=early[0]+1e-10):
                        for k in NUMERIC:assert out[k]==ref[k]
                        prefix+=1
                    if early and out['elapsed_seconds']<=3+1e-10:
                        altered=dict(ss);altered[3]=-ss[3]
                        alt=simulate_repeated(d,t,side,altered,decisions=(3,),early_market_seconds=early)
                        for k in NUMERIC:assert alt[k]==out[k]
                        future_signal+=1
                    for e in out['trace']:
                        if e['event']=='submit':
                            assert e['arrival_index']==int(np.searchsorted(d.times,e['time'],'right'))+1
                            commands.append(dict(**common,strategy=code,time=e['time'],kind=e['kind'],stage=e['stage'],
                                limit_ticks=e['limit_ticks'],arrival_seconds=float(d.times[e['arrival_index']]),
                                realized_arrival=any(a['event']=='arrival' and a['submitted']==e['time'] for a in out['trace'])))
                            if e['kind']=='limit':assert e['limit_ticks']==d.p[d.source_index(e['time'])]-side*19
                    if nominal<2:traces.append(dict(**common,strategy=code,trace=out['trace']))
                    rows.append(dict(**common,strategy=code,**{k:out[k] for k in FIELDS},stage=stage_name(out),
                        early_checks_used=','.join(map(str,[s for s in out['decisions_used'] if s<3])),
                        early_triggered=any(e['event']=='early_signal' and e['action']=='market' for e in out['trace']),
                        missing_decisions=','.join(map(str,out['missing_decisions']))))
        print(f'{date}: 60 tasks, 360 C3/E1/E2 orders verified',flush=True)
    baseout=pd.DataFrame(rows)
    m=baseline[baseline.strategy.eq('M')].copy();m['early_triggered']=False;m['early_checks_used']='';m['missing_decisions']=''
    cols=baseout.columns;allrows=[baseout,m[cols]]
    for code,parent in [('E1_10','E1'),('E2_10','E2'),('M10','M')]:
        source=pd.concat([baseout,m[cols]],ignore_index=True)
        v=source[(source.nominal_second.lt(10)&source.strategy.eq(parent))|(source.nominal_second.ge(10)&source.strategy.eq('C3'))].copy()
        v['strategy']=code;allrows.append(v)
    orders=pd.concat(allrows,ignore_index=True)
    orders['fold']=orders.fold.astype(int)
    assert not orders.duplicated(KEY+['strategy']).any() and orders.groupby(TASK).size().eq(14).all()
    assert len(orders)==15120 and orders[TASK].drop_duplicates().shape[0]==1080
    np.testing.assert_allclose(orders.cost_bp,orders.direction*(orders.fill_ticks-orders.initial_ticks)/orders.initial_ticks*10000,atol=1e-12)
    orders.to_parquet(OUT/'全部逐任务成交.parquet',index=False)
    sig=pd.DataFrame(signals);sig.to_parquet(OUT/'短窗冻结信号.parquet',index=False)
    pd.DataFrame(commands).to_parquet(OUT/'全部下单记录.parquet',index=False);save_json(OUT/'示例事件轨迹.json',traces)
    for w,g in sig.groupby('window'):
        for fold,h in g.groupby('fold'):
            model=next(m for m in models if m['window']==w and m['fold']==fold);h=h[h.signal.notna()]
            np.testing.assert_allclose(predict(model,h[FEATURES].to_numpy()),h[PCOLS],atol=1e-13,rtol=1e-13)
    missing=sig[sig.signal.isna()].groupby(['window','missing_features']).size().rename('tasks').reset_index()
    missing.to_csv(OUT/'早期信号缺失.csv',index=False,encoding='utf-8-sig')
    assert all(sha(p)==hashes[str(p.relative_to(ROOT))] for p in inputs)
    save_json(OUT/'回放验收.json',dict(tasks=1080,orders=len(orders),dates=18,independently_checked_orders=checks,
        saved_prediction_matches=reproduced,causal_checks=causal,no_early_action_matches=prefix,future_signal_invariance=future_signal,
        baseline_orders_exact=2160,model_parameters_reproduced=True,excluded_tasks=0,inputs_unchanged=True,input_sha256=hashes))
    return orders


def analyze(o):
    daily=[];stages=[];attribution=[];bins=[]
    old=o[o.strategy.eq('C3')][KEY+['cost_bp','stage']].rename(columns={'cost_bp':'c3_cost','stage':'c3_stage'})
    o=o.merge(old,on=KEY,validate='many_to_one');o['gain_vs_c3']=o.c3_cost-o.cost_bp
    for scope,(lo,hi) in SCOPES.items():
        for side in ['both','buy','sell']:
            g=o[o.nominal_second.ge(lo)&o.nominal_second.lt(hi)]
            if side!='both':g=g[g.direction.eq(1 if side=='buy' else -1)]
            for (date,code),z in g.groupby(['trade_date','strategy']):
                daily.append(dict(scope=scope,side=side,trade_date=date,strategy=code,fold=int(z.fold.iloc[0]),orders=len(z),
                    cost_bp=float(z.cost_bp.mean()),saving_vs_market_bp=float((z.market_cost_bp-z.cost_bp).mean()),
                    saving_vs_C3_bp=float(z.gain_vs_c3.mean()),elapsed_seconds=float(z.elapsed_seconds.mean()),
                    fallback_rate=float(z.stage.eq('market10').mean()),early_trigger_rate=float(z.early_triggered.mean())))
                for stage in ['initial_limit','market1','market2','limit3','market3','market10','immediate_market']:
                    mask=z.stage.eq(stage);stages.append(dict(scope=scope,side=side,trade_date=date,strategy=code,stage=stage,
                        rate=float(mask.mean()),saving_contribution_bp=float((z.market_cost_bp-z.cost_bp).where(mask,0).mean())))
                for stage in ['initial_limit','limit3','market3','market10']:
                    mask=z.c3_stage.eq(stage);attribution.append(dict(scope=scope,side=side,trade_date=date,strategy=code,c3_stage=stage,
                        rate=float(mask.mean()),gain_contribution_bp=float(z.gain_vs_c3.where(mask,0).mean())))
    daily=pd.DataFrame(daily);rows=[]
    for (scope,side,code),g in daily.groupby(['scope','side','strategy']):
        g=g.sort_values('trade_date');r=dict(scope=scope,side=side,strategy=code,days=len(g),orders=int(g.orders.sum()))
        for metric in ['cost_bp','saving_vs_market_bp','saving_vs_C3_bp','elapsed_seconds','fallback_rate','early_trigger_rate']:r[metric]=float(g[metric].mean())
        for ref in ['market','C3']:
            lo,hi=date_block_interval(g[f'saving_vs_{ref}_bp'],seed=20260928);r.update({f'vs_{ref}_low':float(lo),f'vs_{ref}_high':float(hi)})
        r.update(positive_dates=int(g.saving_vs_C3_bp.gt(1e-12).sum()),negative_dates=int(g.saving_vs_C3_bp.lt(-1e-12).sum()))
        rows.append(r)
    summary=pd.DataFrame(rows)
    for (scope,code),g in summary.groupby(['scope','strategy']):
        g=g.set_index('side')
        np.testing.assert_allclose((g.loc['buy','cost_bp']+g.loc['sell','cost_bp'])/2,g.loc['both','cost_bp'],atol=1e-12)
    for name,f in [('逐日成本',daily),('成本与节约汇总',summary),('逐日成交方式',pd.DataFrame(stages)),('逐日原群体配对贡献',pd.DataFrame(attribution))]:
        f.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    for name,f,group in [('成交方式汇总',pd.DataFrame(stages),'stage'),('原群体配对贡献',pd.DataFrame(attribution),'c3_stage')]:
        metrics=['rate','saving_contribution_bp'] if group=='stage' else ['rate','gain_contribution_bp']
        a=f.groupby(['scope','side','strategy',group],as_index=False)[metrics].mean()
        for key,z in a.groupby(['scope','side','strategy']):
            ref=summary[summary.scope.eq(key[0])&summary.side.eq(key[1])&summary.strategy.eq(key[2])].iloc[0]
            np.testing.assert_allclose(z[metrics].sum(),[1,ref.saving_vs_market_bp if group=='stage' else ref.saving_vs_C3_bp],atol=1e-12)
        a.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    o['bin_start']=(o.nominal_second//10)*10
    for (date,code,start),g in o.groupby(['trade_date','strategy','bin_start']):
        bins.append(dict(trade_date=date,strategy=code,bin_start=int(start),saving_bp=float((g.market_cost_bp-g.cost_bp).mean()),gain_bp=float(g.gain_vs_c3.mean())))
    b=pd.DataFrame(bins);b.to_csv(OUT/'逐日10秒段结果.csv',index=False,encoding='utf-8-sig')
    b.groupby(['strategy','bin_start'],as_index=False)[['saving_bp','gain_bp']].mean().to_csv(OUT/'10秒段结果.csv',index=False,encoding='utf-8-sig')
    daily.groupby(['scope','side','strategy','fold'],as_index=False)[['cost_bp','saving_vs_C3_bp','saving_vs_market_bp']].mean().to_csv(OUT/'四折结果.csv',index=False,encoding='utf-8-sig')
    print(summary[summary.scope.eq('first_minute')&summary.side.eq('both')].to_string(index=False),flush=True)
    return summary


def main():
    orders=run() if '--analyze-only' not in sys.argv else pd.read_parquet(OUT/'全部逐任务成交.parquet')
    analyze(orders)


if __name__=='__main__':main()
