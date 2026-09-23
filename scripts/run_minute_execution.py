"""Frozen opening model transferred to one task per intraday minute."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_ic import SessionData,ICConfig
from utils.opening_schedule import ScheduleConfig
from utils.opening_task_window import bounded_features
from utils.opening_anchored import predict
from utils.opening_two_stage import POLICIES,simulate,date_block_interval
from scripts.run_two_stage_execution import MODELS,RAW,NAMES,sha,verify
from scripts.run_opening_prediction import save_json

OUT=ROOT/'result/opening_execution/every_minute'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [RAW,MODELS]}
    models=[m for m in json.loads(MODELS.read_text(encoding='utf-8')) if m['window']==3 and m['method']=='selected']
    mapping={day:m for m in models for day in m['test_dates']}
    assert len(mapping)==18
    save_json(OUT/'实验口径.json',dict(input_sha256=hashes,test_dates=sorted(mapping),
        model='frozen opening W3 selected models; no retraining or threshold selection',
        minutes=['09:30–11:29','13:00–14:59'],nominal_minutes_per_day=240,
        origin='first snapshot at or after each minute, maximum offset 0.5s; never backfill quote to the minute',
        observation_seconds=3,deadline_seconds=10,delay_snapshots=2,quantity=1,
        sides=[1,-1],common_cohort='all three policies and both sides valid',
        bootstrap='5000 circular date-block samples, block=5; pointwise, no multiple-testing correction'))
    raw=pd.read_parquet(RAW);raw=raw[raw.Date.astype(str).isin(mapping)]
    rows=[];signals=[];traces=[];excluded=[];sessions={};causal=0
    cfg=ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,task_range_seconds=7200,observation_seconds=(3.,)))
    for date,g in raw.groupby(raw.Date.astype(str),sort=True):
        m=mapping[date];assert max(m['train_dates'])<date
        for part,clock,end in [('AM','09:30','11:30'),('PM','13:00','15:00')]:
            origin=pd.Timestamp(date+' '+clock,tz='Asia/Shanghai')
            close=pd.Timestamp(date+' '+end,tz='Asia/Shanghai')
            d=SessionData(g[(g.Datetime>=origin)&(g.Datetime<=close)],origin,cfg)
            # Verification keys need a unique session rather than only the date.
            session_key=date+'_'+part;sessions[session_key]=d
            for minute in range(120):
                nominal=minute*60
                stamp=(origin+pd.Timedelta(seconds=nominal)).strftime('%H:%M')
                i=int(np.searchsorted(d.times,nominal,'left'))
                if i>=d.n or d.times[i]-nominal>.5+1e-10:
                    excluded.append(dict(trade_date=date,clock=stamp,reason='no_snapshot_within_half_second'))
                    continue
                t=float(d.times[i]);feature=bounded_features(d,t,3)
                x=np.array([feature[f] for f in m['features']])
                p=predict(m,x[None,:])[0] if np.isfinite(x).all() else np.full(3,np.nan)
                signal=int(p.argmax()-1) if np.isfinite(p).all() else None
                lab=d.labels(t+3)
                signals.append(dict(trade_date=date,session_key=session_key,part=part,clock=stamp,
                    fold=m['fold'],task_seconds=t,offset_seconds=t-nominal,signal=signal,
                    p_down=p[0],p_flat=p[1],p_up=p[2],y3=lab['y_decision_3s'],
                    label_status=lab['y_decision_3s_status'],snapshot_count=feature['snapshot_count']))
                if minute in [0,30,60,90,119]:
                    sliced=SessionData(d.frame[(d.times>=t)&(d.times<=t+3)],d.open_time,cfg)
                    f=bounded_features(sliced,t,3)
                    np.testing.assert_allclose(x,[f[k] for k in m['features']],rtol=0,atol=0,equal_nan=True)
                    causal+=1
                for direction in [1,-1]:
                    for policy in POLICIES:
                        r=simulate(d,t,direction,policy,signal);events=r.pop('trace')
                        rows.append(dict(trade_date=date,session_key=session_key,clock=stamp,part=part,
                            fold=m['fold'],offset_seconds=t-nominal,**r))
                        traces.append(dict(trade_date=session_key,task_elapsed_seconds=t,policy=policy,
                                           direction=direction,clock=stamp,events=events))
        print(date+' complete',flush=True)
    f=pd.DataFrame(rows);key=['trade_date','clock']
    valid=f.assign(valid=f.status.eq('filled')).groupby(key).valid.all()
    f=f.merge(valid.rename('common_valid'),on=key,validate='many_to_one')
    common=f[f.common_valid].copy()
    base=common[common.policy==POLICIES[0]][key+['direction','cost_bp','cost_ticks']].rename(columns={'cost_bp':'benchmark_cost_bp','cost_ticks':'benchmark_cost_ticks'})
    common=common.merge(base,on=key+['direction'],validate='many_to_one')
    common['saving_bp']=common.benchmark_cost_bp-common.cost_bp
    common['win']=common.benchmark_cost_ticks>common.cost_ticks
    common['loss']=common.benchmark_cost_ticks<common.cost_ticks
    common['tie']=common.benchmark_cost_ticks==common.cost_ticks
    common['limit_fill']=common.fill_kind.eq('limit')
    # Use session ids only within the independent execution replay.
    audit_frame=f.copy();audit_frame['trade_date']=audit_frame.session_key
    audit_frame['task_elapsed_seconds']=audit_frame.task_seconds
    execution_checks=verify(audit_frame,common,traces,sessions,hashes,[RAW,MODELS])
    f.to_parquet(OUT/'全部分钟执行.parquet',index=False)
    common.to_parquet(OUT/'共同任务成交.parquet',index=False)
    sf=pd.DataFrame(signals)
    sf['hit3']=np.where(sf.y3.notna()&sf.signal.notna(),sf.signal.eq(np.sign(sf.y3)),np.nan)
    sf.to_csv(OUT/'逐分钟信号.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(excluded,columns=['trade_date','clock','reason']).to_csv(OUT/'无行情分钟.csv',index=False,encoding='utf-8-sig')
    f[~f.common_valid].to_csv(OUT/'无效执行路径.csv',index=False,encoding='utf-8-sig')
    with (OUT/'逐任务事件.jsonl').open('w',encoding='utf-8') as out:
        for trace in traces:out.write(json.dumps(trace,ensure_ascii=False)+'\n')
    cols=['cost_bp','saving_bp','win','loss','tie','elapsed_seconds','limit_fill']
    daily=common.groupby(['trade_date','part','direction','policy'])[cols].mean().reset_index()
    daily.to_csv(OUT/'逐日分时段结果.csv',index=False,encoding='utf-8-sig')
    tables=[]
    scopes=[('全天',common),('上午',common[common.part=='AM']),('下午',common[common.part=='PM']),
            ('09:30首笔',common[common.clock=='09:30']),('其余分钟',common[common.clock!='09:30'])]
    for scope,g in scopes:
        for side,sub in [('买卖各半',g),('买入',g[g.direction==1]),('卖出',g[g.direction==-1])]:
            for policy,z in sub.groupby('policy'):
                day=z.groupby('trade_date')[cols].mean().sort_index()
                lo,hi=date_block_interval(day.saving_bp)
                tables.append(dict(scope=scope,side=side,policy=policy,n=len(z),days=len(day),
                    **{c:float(day[c].mean()) for c in cols},low=float(lo),high=float(hi),
                    wins=int(z.win.sum()),losses=int(z.loss.sum()),ties=int(z.tie.sum())))
    summary=pd.DataFrame(tables);summary.to_csv(OUT/'分钟策略总表.csv',index=False,encoding='utf-8-sig')
    clock=[]
    for (stamp,policy),g in common.groupby(['clock','policy']):
        day=g.groupby('trade_date')[cols].mean().sort_index();lo,hi=date_block_interval(day.saving_bp)
        clock.append(dict(clock=stamp,policy=policy,days=len(day),n=len(g),
                          **{c:float(day[c].mean()) for c in cols},low=float(lo),high=float(hi)))
    clock=pd.DataFrame(clock);clock.to_csv(OUT/'每分钟结果.csv',index=False,encoding='utf-8-sig')
    hourly=[]
    common['half_hour']=common.clock.str[:3]+np.where(common.clock.str[3:].astype(int)<30,'00','30')
    for (stamp,policy),g in common.groupby(['half_hour','policy']):
        day=g.groupby('trade_date')[cols].mean().sort_index();lo,hi=date_block_interval(day.saving_bp)
        hourly.append(dict(half_hour=stamp,policy=policy,days=len(day),n=len(g),
                           **{c:float(day[c].mean()) for c in cols},low=float(lo),high=float(hi)))
    half=pd.DataFrame(hourly);half.to_csv(OUT/'半小时时段汇总.csv',index=False,encoding='utf-8-sig')
    # The 09:30 observation must reproduce the previously delivered first-order experiment.
    old=pd.read_csv(ROOT/'result/opening_execution/first_order_ablation/逐日首笔成交.csv')
    check=common[common.clock=='09:30'].merge(old,on=['trade_date','policy','direction'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(check)==108
    for field in ['cost_bp','fill_ticks','fill_seconds']:
        np.testing.assert_allclose(check[field+'_new'],check[field+'_old'],rtol=1e-12,atol=1e-12)
    signalstats=[]
    for scope,g in [('全天',sf),('上午',sf[sf.part=='AM']),('下午',sf[sf.part=='PM'])]:
        eligible=g[g.hit3.notna()].copy()
        eligible['down']=eligible.y3.lt(0);eligible['flat']=eligible.y3.eq(0);eligible['up']=eligible.y3.gt(0)
        dailyacc=eligible.groupby('trade_date')[['hit3','down','flat','up']].mean()
        signalstats.append(dict(scope=scope,n=len(eligible),accuracy=float(dailyacc.hit3.mean()),
            down=float(dailyacc.down.mean()),flat=float(dailyacc.flat.mean()),up=float(dailyacc.up.mean())))
    pd.DataFrame(signalstats).to_csv(OUT/'方向预测准确率.csv',index=False,encoding='utf-8-sig')
    evidence=dict(nominal_tasks=18*240,observed_tasks=len(sf),common_tasks=int(valid.sum()),
        no_snapshot_tasks=len(excluded),invalid_common_tasks=int((~valid).sum()),
        invalid_reasons=f[f.status!='filled'].reason.value_counts().to_dict(),
        boundary_checks=causal,opening_replay_rows=len(check),execution_checks=execution_checks)
    reason_tasks=f[f.status!='filled'][['trade_date','clock','reason']].drop_duplicates()
    reason_tasks.to_csv(OUT/'异常任务与原因.csv',index=False,encoding='utf-8-sig')
    evidence['invalid_task_reasons_may_overlap']=reason_tasks.reason.value_counts().to_dict()
    save_json(OUT/'验收.json',evidence)
    plot(clock,half)
    lines=['# 每分钟首笔：开盘模型向日内迁移检验','',
        '沿用冻结的四折3秒增强模型，不训练、不调参。每分钟整点后第一条快照为实际发单起点，偏移不超过0.5秒；观察3秒、统一10秒兜底、两快照执行延迟。',
        '上午09:30–11:29、下午13:00–14:59。每个任务独立模拟买卖各1手；严格胜率=成本低于A，持平不算赢。',
        f"18个检验日共4320个名义任务，共同有效{int(valid.sum())}个；{len(excluded)}个无及时快照，{int((~valid).sum())}个撮合或信号无效。",'',
        '|范围|策略|成本bp|较A节约bp|严格胜率|持平率|日期块95%节约区间|',
        '|---|---|---:|---:|---:|---:|---|']
    for r in summary[summary.side=='买卖各半'].itertuples():
        lines.append(f'|{r.scope}|{NAMES[r.policy]}|{r.cost_bp:.4f}|{r.saving_bp:+.4f}|{r.win:.2%}|{r.tie:.2%}|[{r.low:.4f}, {r.high:.4f}]|')
    lines+=['','各日内部平均后对日期等权。每分钟曲线各点最多18日；逐点区间未校正多重比较，不能据最高的一分钟认定稳定优势。',
        '缺行情示例：2026-08-20 10:31与10:33在整点后0.5秒内无快照；2026-08-19 10:59与14:06的执行路径出现缺口。分别见无行情分钟.csv和异常任务与原因.csv；排除按同任务六条策略／方向路径共同处理。',
        '此实验检验开盘模型的直接迁移效果，不代表已训练出全天模型。已有日期参与过因子研究；L1代理撮合未计手续费、排队、冲击及部分成交。',
        '方向预测准确率单独保存在方向预测准确率.csv，不与执行胜率混用。所有原始输出保留，09:30首笔108条成交复算一致。']
    (OUT/'每分钟执行报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary[summary.side=='买卖各半'].to_string(index=False))
    print(json.dumps(evidence,ensure_ascii=False))


def plot(clock,half):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True)
    stamps=sorted(clock.clock.unique())
    for policy,label,color in [(POLICIES[1],'B 先观察3秒','#147eab'),(POLICIES[2],'C 先挂单再观察','#cf7832')]:
        g=clock[clock.policy==policy].set_index('clock').reindex(stamps)
        for start,stop in [(0,120),(120,240)]:
            x=np.arange(start,stop);sub=g.iloc[start:stop]
            axes[0].plot(x,sub.saving_bp,color=color,alpha=.25,lw=.8)
            axes[0].plot(x,sub.saving_bp.rolling(10,min_periods=1).mean(),color=color,lw=2,label=label if start==0 else None)
            axes[1].plot(x,sub.win.rolling(10,min_periods=1).mean()*100,color=color,lw=2,label=label if start==0 else None)
    for ax in axes:
        ax.axvline(119.5,color='gray',ls='--');ax.grid(alpha=.2);ax.legend(frameon=False)
        ax.spines[['top','right']].set_visible(False)
    axes[0].axhline(0,color='gray',ls='--')
    axes[0].set_ylabel('较基准节约 bp（正数更好）')
    axes[1].set_ylabel('严格胜率 %（持平不算赢）')
    ticks=list(range(0,240,30))+[239]
    axes[1].set_xticks(ticks,[stamps[i] for i in ticks]);axes[1].set_xlabel('任务所属分钟；虚线分隔午休')
    fig.suptitle('每分钟发起一次任务：冻结开盘模型的日内表现',fontsize=17)
    fig.text(.5,.92,'18个检验日 · 买卖各半 · 实线为最近10个分钟点均值，浅线为原始逐分钟节约',ha='center',fontsize=10)
    fig.text(.07,.02,'T为分钟整点后首条快照（≤0.5秒）。只在T+3判断，T+10兜底。曲线用于探索，不按局部峰值选时段。',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.9]);fig.savefig(OUT/'每分钟执行表现.png',dpi=170)
    fig.savefig(OUT/'每分钟执行表现.pdf');plt.close(fig)


if __name__=='__main__':main()
