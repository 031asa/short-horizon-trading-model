"""Frozen W3 signals, two-stage execution ablation, no model/strategy search."""
from pathlib import Path
import sys, json, hashlib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_ic import SessionData, ICConfig, limit_fill_evidence
from utils.opening_task_window import bounded_features
from utils.opening_anchored import predict
from utils.opening_two_stage import RULES, POLICIES, simulate, date_block_interval
from scripts.audit_opening_data import RAW, EXPECTED_SHA256
from scripts.run_opening_prediction import save_json

OUT = ROOT/'result/opening_execution/two_stage_ablation'
MODELS = ROOT/'result/opening_prediction/three_factor_enhancement/模型参数.json'
PREDICTIONS = MODELS.parent/'逐任务预测.parquet'
LABELS = MODELS.parent/'共同任务预测.parquet'
KEY = ['trade_date', 'task_elapsed_seconds']
NAMES = dict(zip(POLICIES, ['A 基准：0档限价10秒', 'B 先观察3秒再下单', 'C 先挂限价、3秒判断']))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_csv(frame, name):
    frame.to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [RAW, MODELS, PREDICTIONS, LABELS, MODELS.parent/'固定三因子原子表.parquet',
                 ROOT/'sample_snapshot_原子执行总表.parquet']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    assert hashes[str(RAW.relative_to(ROOT))] == EXPECTED_SHA256
    models = [m for m in json.loads(MODELS.read_text(encoding='utf-8'))
              if m['window']==3 and m['method']=='selected']
    assert len(models)==4
    by_date = {}
    for m in models:
        assert max(m['train_dates']) < min(m['test_dates'])
        assert set(['F03_OFI_latest','F04_QuoteShift','F07_QuotePosition']) <= set(m['features'])
        for date in m['test_dates']:
            assert date not in by_date
            by_date[date] = m
    assert len(by_date)==18
    frozen = dict(rules=RULES, test_dates=sorted(by_date), input_sha256=hashes,
                  selected_models=models, bootstrap=dict(method='circular date blocks', block=5,
                  repetitions=5000, interval=.95, seed=20260923))
    manifest = OUT/'实验口径.json'
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding='utf-8')) == frozen, 'Frozen configuration differs'
    else:
        save_json(manifest, frozen)
    old = pd.read_parquet(PREDICTIONS)
    old = old[(old.window==3)&(old.method=='selected')].set_index(KEY)
    assert len(old)==1080
    raw = pd.read_parquet(RAW)
    raw = raw[raw.Date.astype(str).isin(by_date)]
    records, signals, traces, opening = [], [], [], []
    replay_count = 0
    causality = 0
    action_causality = 0
    sessions = {}
    for date, g in raw.groupby(raw.Date.astype(str), sort=True):
        d = SessionData(g, pd.Timestamp(date+' 09:30', tz='Asia/Shanghai'), ICConfig())
        sessions[date] = d
        m = by_date[date]
        opening.append(dict(trade_date=date, first_snapshot_seconds=float(d.times[0])))
        for t in range(60):
            feature = bounded_features(d, t, 3)
            x = np.array([feature[f] for f in m['features']])
            signal, probs = None, np.full(3, np.nan)
            if np.isfinite(x).all():
                probs = predict(m, x[None,:])[0]
                signal = int(probs.argmax()-1)
                golden = old.loc[(date,t), ['p_down','p_flat','p_up']].to_numpy(float)
                np.testing.assert_allclose(probs, golden, rtol=1e-12, atol=1e-12)
                replay_count += 1
            signals.append(dict(trade_date=date, task_elapsed_seconds=t, fold=m['fold'], signal=signal,
                p_down=probs[0], p_flat=probs[1], p_up=probs[2], **feature))
            if t in (1,17,48):
                # Independent task-local reconstruction excludes BOTH forbidden
                # pre-observation data and every future observation.
                clipped = SessionData(d.frame[(d.times>=t)&(d.times<=t+3)], d.open_time, d.config)
                f = bounded_features(clipped,t,3)
                np.testing.assert_allclose(x,[f[k] for k in m['features']],rtol=0,atol=0,equal_nan=True)
                changed = d.frame.copy()
                outside = (d.times<t)|(d.times>t+3)
                for column in ['LastPrice','BidPrice1','AskPrice1']:
                    changed.loc[outside,column] += 100
                changed.loc[outside,['BidVolume1','AskVolume1']] *= 2
                perturbed = SessionData(changed,d.open_time,d.config)
                f = bounded_features(perturbed,t,3)
                np.testing.assert_allclose(x,[f[k] for k in m['features']],rtol=0,atol=0,equal_nan=True)
                causality += 1
                future_frame=d.frame.copy()
                for column in ['LastPrice','BidPrice1','AskPrice1']:
                    future_frame.loc[d.times>t+3,column] += 100
                future_changed=SessionData(future_frame,d.open_time,d.config)
                for side in [1,-1]:
                    for policy in POLICIES:
                        before=simulate(d,t,side,policy,signal)['trace']
                        after=simulate(future_changed,t,side,policy,signal)['trace']
                        assert [e for e in before if e['time']<=t+3]==[e for e in after if e['time']<=t+3]
                        action_causality+=1
            for direction in [1,-1]:
                for policy in POLICIES:
                    result = simulate(d,t,direction,policy,signal)
                    trace = result.pop('trace')
                    records.append(dict(trade_date=date, task_elapsed_seconds=t, fold=m['fold'],
                                        signal=signal, **result))
                    traces.append(dict(trade_date=date, task_elapsed_seconds=t, direction=direction,
                                       policy=policy, events=trace))
        print(f'{date}: 60 tasks, six execution paths each', flush=True)
    assert replay_count == 1080 and causality == 54
    frame = pd.DataFrame(records)
    signal_frame = pd.DataFrame(signals)
    assert len(frame)==6480 and not frame.duplicated(KEY+['direction','policy']).any()
    valid = frame.assign(valid=frame.status.eq('filled')).groupby(KEY).valid.all().rename('common_valid')
    frame = frame.merge(valid,on=KEY,validate='many_to_one')
    common = frame[frame.common_valid].copy()
    assert common.groupby(KEY).size().eq(6).all()
    # Attach paired benchmark cost; side-specific costs always use the SAME P0.
    base = common[common.policy==POLICIES[0]][KEY+['direction','cost_bp','cost_ticks']]
    base = base.rename(columns={'cost_bp':'benchmark_cost_bp','cost_ticks':'benchmark_cost_ticks'})
    common = common.merge(base,on=KEY+['direction'],validate='many_to_one')
    common['saving_bp'] = common.benchmark_cost_bp-common.cost_bp
    common['saving_ticks'] = common.benchmark_cost_ticks-common.cost_ticks
    common['cost_points'] = common.cost_ticks*.2
    common['improved'] = common.saving_ticks>0
    common['worse'] = common.saving_ticks<0
    common['tie'] = common.saving_ticks==0
    common['limit_fill'] = common.fill_kind.eq('limit')
    common['signal_market'] = common.fill_kind.eq('market')&common.fill_stage.eq('signal')
    common['deadline_market'] = common.fill_kind.eq('market')&common.fill_stage.eq('deadline')
    common['early_fill'] = common.elapsed_seconds<=3
    common['clock_bin'] = (common.task_elapsed_seconds//10)*10
    frame.to_parquet(OUT/'全部模拟结果.parquet',index=False)
    common.to_parquet(OUT/'共同任务成交明细.parquet',index=False)
    signal_frame.to_parquet(OUT/'冻结模型信号.parquet',index=False)
    write_csv(frame[~frame.common_valid], '排除明细')
    write_csv(pd.DataFrame(opening),'开盘行情起点')
    with (OUT/'逐任务事件.jsonl').open('w',encoding='utf-8') as f:
        for row in traces:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    fields = ['cost_bp','cost_ticks','cost_points','saving_bp','saving_ticks','elapsed_seconds',
              'submissions','replacements','improved','worse','tie','limit_fill','signal_market',
              'deadline_market','early_fill']
    daily = common.groupby(['trade_date','fold','direction','policy'])[fields].mean().reset_index()
    counts = common.groupby(['trade_date','fold','direction','policy']).size().reset_index(name='n')
    daily = daily.merge(counts)
    write_csv(daily,'逐日分方向结果')
    summaries = []
    for side, selected in [('买入',daily[daily.direction==1]), ('卖出',daily[daily.direction==-1]), ('买卖各半',daily)]:
        means = selected.groupby(['trade_date','policy'])[fields].mean().reset_index()
        for policy, group in means.groupby('policy'):
            lo,hi = date_block_interval(group.sort_values('trade_date').saving_bp)
            summaries.append(dict(side=side,policy=policy,name=NAMES[policy],days=len(group),
                n=int(selected[selected.policy==policy].n.sum()),
                **{f:float(group[f].mean()) for f in fields},
                saving_ci_low=float(lo),saving_ci_high=float(hi),
                better_days=int((group.saving_bp>1e-12).sum()),
                worse_days=int((group.saving_bp<-1e-12).sum())))
    summary = pd.DataFrame(summaries)
    write_csv(summary,'策略总表')
    fold_results = daily.groupby(['fold','policy'])[fields].mean().reset_index()
    write_csv(fold_results,'四折结果')
    pair = common.pivot(index=KEY+['fold','direction'],columns='policy',values='cost_bp').reset_index()
    pair['C_vs_B_saving_bp'] = pair[POLICIES[1]]-pair[POLICIES[2]]
    pair['B_vs_A_saving_bp'] = pair[POLICIES[0]]-pair[POLICIES[1]]
    pair['C_vs_A_saving_bp'] = pair[POLICIES[0]]-pair[POLICIES[2]]
    pair['clock_bin'] = pair.task_elapsed_seconds//10*10
    pair_daily = pair.groupby(['trade_date','fold'])[['C_vs_B_saving_bp','B_vs_A_saving_bp','C_vs_A_saving_bp']].mean().reset_index()
    write_csv(pair_daily,'逐日配对节约')
    write_csv(pair,'逐任务配对节约')
    clock_daily = pair.groupby(['trade_date','clock_bin'])[['C_vs_B_saving_bp','B_vs_A_saving_bp','C_vs_A_saving_bp']].mean().reset_index()
    clock_rows=[]
    for clock_bin, group in clock_daily.groupby('clock_bin'):
        for name in ['C_vs_B_saving_bp','B_vs_A_saving_bp','C_vs_A_saving_bp']:
            lo,hi=date_block_interval(group.sort_values('trade_date')[name])
            clock_rows.append(dict(clock_bin=clock_bin,comparison=name,mean_bp=float(group[name].mean()),
                                   low_bp=float(lo),high_bp=float(hi),days=len(group)))
    write_csv(pd.DataFrame(clock_rows),'按任务时刻分段节约')
    # Separate where the early limit changes the execution path.
    c = common[common.policy==POLICIES[2]][KEY+['direction','fill_kind','fill_stage','elapsed_seconds',
                                            'pending_cancelled','skipped_same_price']]
    c = c.assign(attribution=np.select([
        c.fill_stage.eq('initial')&c.elapsed_seconds.le(3),
        c.fill_stage.eq('initial')&c.skipped_same_price,
        c.fill_stage.eq('initial'), c.fill_kind.eq('market')&c.fill_stage.eq('signal'),
        c.fill_kind.eq('limit')&c.fill_stage.eq('signal')],
        ['initial_fill_by_3s','unchanged_limit_later_fill','initial_fill_during_replacement',
         'signal_market','updated_limit_fill'],default='deadline_market'))
    attributed = pair.merge(c,on=KEY+['direction'],validate='one_to_one')
    # Contributions use each day's FULL paired denominator, so categories sum
    # exactly to the aggregate ablation, not a misleading conditional mean.
    contrib=[]
    for date, group in attributed.groupby('trade_date'):
        for category in ['initial_fill_by_3s','unchanged_limit_later_fill','initial_fill_during_replacement',
                         'signal_market','updated_limit_fill','deadline_market']:
            a=group[group.attribution==category]
            contrib.append(dict(trade_date=date,attribution=category,n=len(a),
                share=len(a)/len(group),contribution_bp=a.C_vs_B_saving_bp.sum()/len(group)))
    contrib=pd.DataFrame(contrib)
    np.testing.assert_allclose(contrib.groupby('trade_date').contribution_bp.sum(),
                               pair_daily.set_index('trade_date').C_vs_B_saving_bp,rtol=1e-12,atol=1e-12)
    write_csv(contrib,'消融逐日归因')
    write_csv(contrib.groupby('attribution')[['share','contribution_bp']].mean().reset_index(),'消融归因汇总')
    lo,hi=date_block_interval(pair_daily.sort_values('trade_date').C_vs_B_saving_bp)
    ablation=dict(mean_saving_bp=float(pair_daily.C_vs_B_saving_bp.mean()),low=float(lo),high=float(hi),
                  better_days=int(pair_daily.C_vs_B_saving_bp.gt(0).sum()),days=len(pair_daily),
                  improved=float(pair.C_vs_B_saving_bp.gt(0).mean()),worse=float(pair.C_vs_B_saving_bp.lt(0).mean()),
                  tie=float(pair.C_vs_B_saving_bp.eq(0).mean()))
    audit = verify(frame,common,traces,sessions,hashes,protected)
    audit.update(signal_replays=replay_count,strict_feature_causality_checks=causality,
        future_action_causality_checks=action_causality,
        nominal_tasks=len(valid),common_tasks=int(valid.sum()),test_days=len(by_date),
        missing_task_count=int((~valid).sum()),ablation=ablation,
        exclusion_reasons=frame[frame.status!='filled'].reason.value_counts().to_dict(),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'utils/opening_two_stage.py']})
    save_json(OUT/'验收与结果.json',audit)
    report(summary,ablation,audit,fold_results)
    examples(common,traces)
    print(summary[['side','policy','cost_bp','saving_bp','limit_fill','signal_market','deadline_market']].to_string(index=False))
    print(json.dumps(ablation,ensure_ascii=False))


def verify(frame,common,traces,sessions,hashes,protected):
    checked=0
    for row,trace in zip(frame.sort_index().to_dict('records'),traces):
        assert row['trade_date']==trace['trade_date'] and row['policy']==trace['policy']
        d=sessions[row['trade_date']];t=row['task_elapsed_seconds']; side=row['direction']
        events=trace['events']
        assert [e['time'] for e in events]==sorted(e['time'] for e in events)
        for e in events:
            if e['event']=='submit':
                assert e['time'] in [t,t+3,t+10]
                assert e['arrival_index']==np.searchsorted(d.times,e['time'],side='right')+1
                if e['kind']=='limit':
                    assert e['limit_ticks']==d.p[d.source_index(e['time'])]
            if e['event']=='arrival':
                arrival=np.searchsorted(d.times,e['submitted'],side='right')+1
                assert e['time']==d.times[arrival]
        if row['status']!='filled':
            continue
        fills=[e for e in events if e['event']=='fill'];assert len(fills)==1
        e=fills[0];i=e['snapshot_index']
        assert row['fill_seconds']==d.times[i]
        assert row['initial_ticks']==d.p[d.source_index(t)]
        assert abs(row['cost_bp']-side*(row['fill_ticks']-row['initial_ticks'])/row['initial_ticks']*10000)<1e-12
        if row['fill_kind']=='market':
            assert row['fill_ticks']==(d.a[i] if side==1 else d.b[i])
        else:
            hit,source=limit_fill_evidence(side,row['fill_ticks'],d.p[i],d.b[i],d.a[i],d.dv[i])
            assert hit and source==row['fill_mechanism']
        # Independently find the earliest fill by scanning each submitted
        # order's active interval, rather than running the event state machine.
        commands=[e for e in events if e['event']=='submit']
        independent=None
        for j,command in enumerate(commands):
            start=command['arrival_index']
            if command['kind']=='market':
                independent=(start,float(d.a[start] if side==1 else d.b[start]),'market')
                break
            end=commands[j+1]['arrival_index'] if j+1<len(commands) else d.n-1
            limit=command['limit_ticks']
            indices=np.arange(start,min(end+1,d.n))
            quote=d.a[indices]<=limit if side==1 else d.b[indices]>=limit
            crossed=d.p[indices]<limit if side==1 else d.p[indices]>limit
            hit=quote|((d.dv[indices]>0)&crossed)
            if hit.any():
                independent=(int(indices[np.flatnonzero(hit)[0]]),limit,'limit')
                break
        assert independent==(i,row['fill_ticks'],row['fill_kind'])
        checked+=1
    np.testing.assert_allclose(common.saving_bp,common.benchmark_cost_bp-common.cost_bp,atol=0,rtol=0)
    for p in protected:
        assert sha(p)==hashes[str(p.relative_to(ROOT))]
    return dict(verified_fills=checked,independent_first_fill_replays=checked,
                old_files_unchanged=True,decision_times=[0,3,10],
                settlement_and_delay_checks=True,paired_comparison=True)


def examples(common,traces):
    """Export a small set of actual event timelines for manual inspection."""
    indexed={(r['trade_date'],r['task_elapsed_seconds'],r['direction'],r['policy']):r['events'] for r in traces}
    rows=[]
    for _,g in common.groupby(KEY+['direction'],sort=True):
        rows.append((float(g[g.policy==POLICIES[1]].cost_bp.iloc[0]-g[g.policy==POLICIES[2]].cost_bp.iloc[0]),g))
    rows.sort(key=lambda x:x[0])
    chosen=[('C较B更差的路径示例',rows[0][1]),('C较B更好的路径示例',rows[-1][1])]
    lines=['# 两阶段成交路径示例','','以下为实际任务中的极值示例，用来审计规则，不代表典型收益。']
    for title,g in chosen:
        first=g.iloc[0];side='买入' if first.direction==1 else '卖出'
        lines += ['',f'## {title}：{first.trade_date}，开盘第{int(first.task_elapsed_seconds)}秒，{side}',
                  '',f'共同初始LastPrice={first.initial_ticks*.2:.1f}。时间为相对任务起点的秒数。',
                  '', '|策略|成交价|成本 bp|耗时 s|','|---|---:|---:|---:|']
        for r in g.itertuples():
            lines.append(f'|{NAMES[r.policy]}|{r.fill_ticks*.2:.1f}|{r.cost_bp:.4f}|{r.elapsed_seconds:.1f}|')
        for r in g.itertuples():
            lines += ['',f'### {NAMES[r.policy]}','', '|时间|事件|详情|','|---:|---|---|']
            for e in indexed[(r.trade_date,r.task_elapsed_seconds,r.direction,r.policy)]:
                detail={k:v for k,v in e.items() if k not in ['time','event']}
                lines.append(f"|{e['time']-r.task_elapsed_seconds:.1f}|{e['event']}|{json.dumps(detail,ensure_ascii=False)}|")
    (OUT/'成交路径示例.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def report(summary,ablation,audit,fold_results):
    lines=['# 两阶段执行消融实验','',
        '只在第3秒判断一次；不存在第6／9秒更新。买入和卖出各模拟一手，每个任务独立，不叠加冲击。',
        'A：T时刻按LastPrice挂0档限价，T+10仍未成交则转市价。',
        'B：先观察3秒；方向不利于本次执行则市价，否则按T+3当时LastPrice挂限价。T+10兜底。',
        'C：T时刻先挂0档限价，T+3若未成交，执行与B相同的判断。其后不再更新，T+10兜底。','',
        '买入遇上涨、卖出遇下跌属于不利方向；预测不变走限价。使用四折冻结的W=3增强模型，全部1080条概率已与原文件逐项对照，不重训、不按执行收益调参。','',
        '每次下单／原子撤换／转市价均等待决策之后第2条快照生效。撤换途中原单有效；同快照先处理原单成交，再处理新单到达，最后处理定时判断。限价相同则保留原单。10秒为转市价的提交时点，因此成交可能晚于10秒。','',
        '限价买入：卖一≤限价，或者正增量成交量且LastPrice严格低于限价；卖出镜像。两种均按限价结算。新限价到达快照也按此代理规则检查。市场单按到达时对手一价结算。',
        'L1快照不能还原排队位置或快照区间内的先后成交；结果属于既定撮合代理，非实盘可保证的成交。没有手续费、市场冲击、部分成交。','',
        f"数据为IC2609期货。18个检验日共{audit['nominal_tasks']}个名义任务，三策略×双方向共同有效{audit['common_tasks']}个；无初始报价排除{audit['missing_task_count']}个。",
        '第一条开盘快照晚于09:30:00，因此每天第0秒无法取得当时P0；不借未来报价填补。任务仍可产生于首分钟末尾，10秒执行期允许跨过09:31。',
        '所有穿越路径快照均要求价格、盘口、队列、成交量有效，间隔≤1秒，连续源行；决策来源年龄≤0.5秒。缺口不跨越、缺信号不代作下跌。','',
        '成本(bp)=方向×(成交价−任务起点LastPrice)/任务起点LastPrice×10000；节约=A成本−策略成本，正数更好。各日内部求均值，再对日期等权，买卖各半。','',
        '|策略|平均成本 bp|较A节约 bp|节约95%日期块区间|限价成交率|信号市价率|到期市价率|平均耗时 s|',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for r in summary[summary.side=='买卖各半'].itertuples():
        lines.append(f'|{r.name}|{r.cost_bp:.4f}|{r.saving_bp:+.4f}|[{r.saving_ci_low:.4f}, {r.saving_ci_high:.4f}]|{r.limit_fill:.2%}|{r.signal_market:.2%}|{r.deadline_market:.2%}|{r.elapsed_seconds:.3f}|')
    lines+=['',f"C较B平均节约{ablation['mean_saving_bp']:+.4f} bp，95%日期块区间[{ablation['low']:.4f}, {ablation['high']:.4f}]；{ablation['better_days']}/18日为正。",
        '区间按连续5日循环日期块重采样5000次，未把同日重叠任务当成独立样本。此前这些日期参与过因子研究，因此仍是已有数据上的探索。',
        '', '## 四折结果（买卖各半）','', '|折|策略|成本 bp|较A节约 bp|','|---|---|---:|---:|']
    for r in fold_results.itertuples():
        lines.append(f'|{r.fold}|{NAMES[r.policy]}|{r.cost_bp:.4f}|{r.saving_bp:+.4f}|')
    lines+=['','逐日、分方向、时段、消融归因、逐任务成本及完整事件记录均保存在同目录。',
        '消融归因按C最终成交分支分组；贡献用每日全部任务为分母，可加回总体C−B节约。',
        '固定规则的B与C对照度量“预先挂单”的总执行价值，包含早成交及撤换途中的成交机会；不是模型预测贡献的独立估计。',
        '',f"验收：{audit['signal_replays']}条冻结信号重放、{audit['strict_feature_causality_checks']}组截断及双向扰动、{audit['future_action_causality_checks']}条路径的未来扰动、{audit['verified_fills']}笔成交的独立最早成交复算、价格与延迟检查通过；原数据与模型文件未改。"]
    (OUT/'两阶段执行消融报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
