"""One first actionable opening task per test date, using frozen W3 models."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_ic import SessionData,ICConfig
from utils.opening_task_window import bounded_features
from utils.opening_anchored import predict
from utils.opening_two_stage import POLICIES,simulate,date_block_interval
from scripts.run_two_stage_execution import MODELS,NAMES,RAW,sha,verify
from scripts.run_opening_prediction import save_json

OUT=ROOT/'result/opening_execution/first_order_ablation'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    original={str(p.relative_to(ROOT)):sha(p) for p in [RAW,MODELS]}
    models=[m for m in json.loads(MODELS.read_text(encoding='utf-8')) if m['window']==3 and m['method']=='selected']
    date_model={date:m for m in models for date in m['test_dates']}
    assert len(date_model)==18
    config=dict(task_origin='first observed opening snapshot; no pre-opening/future quote substitution',
                model='frozen selected W3; re-evaluate features at actual first-order origin',
                observation_seconds=3,deadline_seconds=10,execution_delay_snapshots=2,
                input_sha256=original,test_dates=sorted(date_model),
                benchmark='initial LastPrice limit; unchanged until T+10 market',
                alternative_B='observe 3 seconds, adverse signal market, otherwise current LastPrice limit',
                alternative_C='initial limit then one decision at T+3, same rule as B',
                win='strictly lower side-signed cost than A on same task; ties not wins',
                cohort='all policies and both sides valid on same date')
    save_json(OUT/'实验口径.json',config)
    raw=pd.read_parquet(RAW)
    raw=raw[raw.Date.astype(str).isin(date_model)]
    records=[];signals=[];traces=[];causal=0;sessions={}
    for date,g in raw.groupby(raw.Date.astype(str),sort=True):
        d=SessionData(g,pd.Timestamp(date+' 09:30',tz='Asia/Shanghai'),ICConfig())
        sessions[date]=d
        # The FIRST observed snapshot is the task origin. If invalid, simulate
        # excludes the day; do not skip bad rows to obtain a better entry quote.
        t=float(d.times[0]);m=date_model[date]
        assert max(m['train_dates'])<date
        f=bounded_features(d,t,3)
        x=np.array([f[name] for name in m['features']])
        p=predict(m,x[None,:])[0] if np.isfinite(x).all() else np.full(3,np.nan)
        signal=int(p.argmax()-1) if np.isfinite(p).all() else None
        lab=d.labels(t+3)
        sliced=SessionData(d.frame[(d.times>=t)&(d.times<=t+3)],d.open_time,d.config)
        local=bounded_features(sliced,t,3)
        np.testing.assert_allclose(x,[local[name] for name in m['features']],rtol=0,atol=0,equal_nan=True)
        causal+=1
        row=dict(trade_date=date,fold=m['fold'],task_seconds=t,decision_seconds=t+3,deadline_seconds=t+10,
                 initial_price=float(d.p[0]*.2),signal=signal,p_down=p[0],p_flat=p[1],p_up=p[2],
                 snapshot_count=f['snapshot_count'],**{name:f[name] for name in m['features']})
        for h in range(1,11):
            row[f'future_{h}s_ticks']=lab[f'y_decision_{h}s']
            row[f'future_{h}s_status']=lab[f'y_decision_{h}s_status']
        signals.append(row)
        for direction in [1,-1]:
            for policy in POLICIES:
                r=simulate(d,t,direction,policy,signal)
                trace=r.pop('trace')
                records.append(dict(trade_date=date,fold=m['fold'],signal=signal,**r))
                traces.append(dict(trade_date=date,task_elapsed_seconds=t,policy=policy,direction=direction,events=trace))
    f=pd.DataFrame(records)
    good=f.assign(valid=f.status.eq('filled')).groupby('trade_date').valid.all()
    f['common_valid']=f.trade_date.map(good)
    common=f[f.common_valid].copy()
    base=common[common.policy==POLICIES[0]][['trade_date','direction','cost_bp','cost_ticks']].rename(columns={'cost_bp':'benchmark_bp','cost_ticks':'benchmark_ticks'})
    common=common.merge(base,on=['trade_date','direction'],validate='many_to_one')
    common['saving_bp']=common.benchmark_bp-common.cost_bp
    common['saving_ticks']=common.benchmark_ticks-common.cost_ticks
    common['win']=common.saving_ticks>0
    common['loss']=common.saving_ticks<0
    common['tie']=common.saving_ticks==0
    common['limit_fill']=common.fill_kind.eq('limit')
    common['signal_market']=common.fill_kind.eq('market')&common.fill_stage.eq('signal')
    common['deadline_market']=common.fill_kind.eq('market')&common.fill_stage.eq('deadline')
    stats=[]
    for side,d in [('买入',common[common.direction==1]),('卖出',common[common.direction==-1]),('买卖各半',common)]:
        for policy,g in d.groupby('policy'):
            daily=g.groupby('trade_date').saving_bp.mean().sort_index()
            ci=date_block_interval(daily.to_numpy())
            stats.append(dict(side=side,policy=policy,name=NAMES[policy],n=len(g),days=len(daily),
                wins=int(g.win.sum()),losses=int(g.loss.sum()),ties=int(g.tie.sum()),win_rate=float(g.win.mean()),
                loss_rate=float(g.loss.mean()),tie_rate=float(g.tie.mean()),
                cost_bp=float(g.cost_bp.mean()),saving_bp=float(daily.mean()),
                ci_low=float(ci[0]),ci_high=float(ci[1]),elapsed_seconds=float(g.elapsed_seconds.mean()),
                limit_fill=float(g.limit_fill.mean()),signal_market=float(g.signal_market.mean()),deadline_market=float(g.deadline_market.mean())))
    stats=pd.DataFrame(stats)
    s=pd.DataFrame(signals)
    accuracy=[]
    for h in range(1,11):
        y=s[f'future_{h}s_ticks']
        valid=y.notna()&s.signal.notna()
        hit=s.loc[valid,'signal'].eq(np.sign(y[valid]))
        accuracy.append(dict(horizon=h,n=int(valid.sum()),correct=int(hit.sum()),accuracy=float(hit.mean()),
                             actual_up=int((y[valid]>0).sum()),actual_flat=int((y[valid]==0).sum()),actual_down=int((y[valid]<0).sum())))
    compare=common.pivot(index=['trade_date','direction'],columns='policy',values='cost_bp')
    compare['C_vs_B_saving_bp']=compare[POLICIES[1]]-compare[POLICIES[2]]
    paired=compare.groupby('trade_date').mean()
    ci=date_block_interval(paired.C_vs_B_saving_bp.to_numpy())
    ablation=dict(C_vs_B_saving_bp=float(paired.C_vs_B_saving_bp.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),
                  wins=int((compare.C_vs_B_saving_bp>0).sum()),losses=int((compare.C_vs_B_saving_bp<0).sum()),
                  ties=int((compare.C_vs_B_saving_bp==0).sum()))
    execution_audit=verify(f.assign(task_elapsed_seconds=f.task_seconds),
        common.rename(columns={'benchmark_bp':'benchmark_cost_bp'}),traces,sessions,original,[RAW,MODELS])
    for p in [RAW,MODELS]:assert sha(p)==original[str(p.relative_to(ROOT))]
    audit=dict(nominal_days=len(good),valid_days=int(good.sum()),execution_rows=len(common),
               feature_causality_checks=causal,input_files_unchanged=True,
               start_min_seconds=float(s.task_seconds.min()),start_max_seconds=float(s.task_seconds.max()),
               ablation=ablation,execution_audit=execution_audit,
               invalid_reasons=f[f.status!='filled'].reason.value_counts().to_dict())
    save_json(OUT/'验收与结果.json',audit)
    f.to_parquet(OUT/'全部模拟结果.parquet',index=False)
    common.to_csv(OUT/'逐日首笔成交.csv',index=False,encoding='utf-8-sig')
    s.to_csv(OUT/'逐日首笔信号.csv',index=False,encoding='utf-8-sig')
    stats.to_csv(OUT/'首笔执行胜率.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(accuracy).to_csv(OUT/'首笔方向预测准确率.csv',index=False,encoding='utf-8-sig')
    compare.reset_index().to_csv(OUT/'首笔消融配对.csv',index=False,encoding='utf-8-sig')
    save_json(OUT/'逐日首笔事件.json',traces)
    lines=['# 只看每天最开始发单的一笔任务','',
        '本表胜率指同一天、同方向的执行成本严格低于基准A的比例，相同成本单独计为持平。它不是方向预测准确率。','',
        '09:30:00整没有可用报价，因此以当天第一条实际开盘快照作为可发单起点T（09:30:00.1～00.5）。',
        '所有策略共用此T和初始LastPrice；T+3决策，T+10兜底，保留两条未来快照的执行延迟。',
        '这与名义09:30:00任务不同，不把首条未来报价倒填到09:30:00；冻结的W=3模型在新起点重算特征，未重训。','',
        f"18个检验日中共同有效{audit['valid_days']}日；每日期买卖各模拟1手，共{len(common)//3}个方向任务。",'',
        '|策略|平均成本 bp|较A节约 bp|胜／负／平|严格胜率|平均完成 s|',
        '|---|---:|---:|---|---:|---:|']
    for r in stats[stats.side=='买卖各半'].itertuples():
        lines.append(f'|{r.name}|{r.cost_bp:.3f}|{r.saving_bp:+.3f}|{r.wins}/{r.losses}/{r.ties}|{r.win_rate:.1%}|{r.elapsed_seconds:.3f}|')
    lines+=['','逐买卖方向见首笔执行胜率.csv，逐日路径见逐日首笔事件.json。',
        f"C较B平均节约{ablation['C_vs_B_saving_bp']:+.3f}bp，5日日期块重采样95%区间[{ablation['ci_low']:.3f}, {ablation['ci_high']:.3f}]。",'',
        '只有18个日期；买卖两方向来自相同行情，不能当成36个独立日期。该结果不能支持稳定高胜率的结论。',
        '沿用L1代理撮合，未包含排队、手续费、冲击与部分成交。所有成交价格和时钟均由已有撮合引擎计算。']
    (OUT/'首笔执行结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(stats.to_string(index=False))
    print(pd.DataFrame(accuracy).to_string(index=False))
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':
    main()
