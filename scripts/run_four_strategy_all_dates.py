"""All 40 source dates rerun with the latest frozen W3 model, exploratory only."""
from scripts.run_second_ranges import *
from scripts import deliver_four_strategy_comparison as charts
import zipfile
import hashlib

DEST=OUT/'four_strategy_c19_all40'
ORIGINAL=charts.DEST/'四策略逐任务配对.parquet'
FIELDS=charts.KEY+['strategy','task_seconds','initial_ticks','fill_seconds','fill_ticks','cost_ticks',
    'cost_bp','elapsed_seconds','fill_kind','fill_stage','pending_cancelled']


def full_path_safe(d,t):
    """Sufficient (not necessary) condition that every passive offset can finish."""
    i=d.source_index(t);j=int(np.searchsorted(d.times,t+10,'right'))+1
    if i<0 or j>=d.n:return False
    ids=slice(i,j+1)
    if not all(d.valid[k][ids].all() for k in ['price','book','queue','volume']):return False
    if not (d.qa[ids]>=1).all() or not (d.qb[ids]>=1).all():return False
    if not d.time_edge[i+1:j+1].all() or not d.edge['volume'][i+1:j+1].all():return False
    for clock in [t,t+3,t+10]:
        ix=d.source_index(clock)
        if ix<0 or clock-d.times[ix]>.5+1e-10:return False
    return True


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    original=pd.read_parquet(ORIGINAL)
    inputs=[RAW,MODELS,ORIGINAL]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    models=[m for m in json.loads(MODELS.read_text(encoding='utf-8')) if m['window']==3 and m['method']=='selected']
    latest=max(models,key=lambda m:m['fold'])
    raw=pd.read_parquet(RAW);dates=sorted(raw.Date.astype(str).unique());assert len(dates)==40
    records=[];signals=[];excluded=[];coverage=[];verified=0;fast_checks=0;causal_checks=0
    cfg=ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,task_range_seconds=3600,observation_seconds=(3.,)))
    for date,g in raw.groupby(raw.Date.astype(str),sort=True):
        opening=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai')
        g=g[(g.Datetime>=opening)&(g.Datetime<=opening+pd.Timedelta(seconds=3620))]
        hour=g[g.Datetime<opening+pd.Timedelta(hours=1)]
        m=latest
        phase='latest_model_forward_test' if date in m['test_dates'] else ('training_backcast' if date in m['train_dates'] else 'earlier_date_backcast')
        info=dict(trade_date=date,first_hour_snapshots=len(hour),first_snapshot=str(hour.Datetime.min()),
            model_fold=m['fold'],model_training_end=max(m['train_dates']),phase=phase,
            temporally_forward=max(m['train_dates'])<date,source_date_in_model_train=date in m['train_dates'])
        if hour.empty:
            excluded.extend(dict(trade_date=date,nominal_second=s,reason='no_first_hour_quotes') for s in range(3600))
            info['action']='no_valid_first_hour_tasks';coverage.append(info);continue
        d=SessionData(g,opening,cfg);info['action']='rerun_frozen_latest_fold4';coverage.append(info)
        for second in range(3600):
            def reject(reason):excluded.append(dict(trade_date=date,nominal_second=second,reason=reason))
            i=int(np.searchsorted(d.times,second,'left'))
            if i>=d.n or d.times[i]-second>.5+1e-10:
                reject('no_snapshot_within_half_second');continue
            t=float(d.times[i]);f=bounded_features(d,t,3)
            x=np.array([f[k] for k in m['features']])
            if not np.isfinite(x).all():reject('missing_signal_features');continue
            p=predict(m,x[None,:])[0];signal=int(p.argmax()-1)
            signals.append(dict(trade_date=date,nominal_second=second,task_seconds=t,signal=signal,
                model_fold=m['fold'],phase=phase,p_down=float(p[0]),p_flat=float(p[1]),p_up=float(p[2])))
            if second%120==0:
                bounded=SessionData(d.frame[(d.times>=t)&(d.times<=t+3)],d.open_time,cfg)
                bf=bounded_features(bounded,t,3)
                np.testing.assert_allclose(x,[bf[k] for k in m['features']],atol=0,rtol=0,equal_nan=True);causal_checks+=1
            j=int(np.searchsorted(d.times,t,'right'))+1
            if j>=d.n or not all(d.valid[k][i:j+1].all() for k in ['price','book','queue','volume']) or not d.time_edge[i+1:j+1].all() or not d.edge['volume'][i+1:j+1].all() or min(d.qa[j],d.qb[j])<1:
                reject('invalid_immediate_market_path');continue
            task_rows=[];valid=True
            for side in [1,-1]:
                price=d.a[j] if side==1 else d.b[j];ticks=side*(price-d.p[i])
                task_rows.append(dict(trade_date=date,nominal_second=second,direction=side,strategy='M',
                    task_seconds=t,initial_ticks=float(d.p[i]),fill_seconds=float(d.times[j]),fill_ticks=float(price),
                    cost_ticks=float(ticks),cost_bp=float(ticks/d.p[i]*10000),elapsed_seconds=float(d.times[j]-t),
                    fill_kind='market',fill_stage='immediate',pending_cancelled=False))
                for strategy,policy,offset in [('A','A_benchmark',0),('B','B_observe_first',0),('C','C_limit_first',19),('C0','C_limit_first',0)]:
                    r=simulate(d,t,side,policy,signal,c_limit_offset_ticks=offset);trace=r.pop('trace')
                    if r['status']!='filled':valid=False;reject(strategy+':'+r['reason']);break
                    ix,fill=independent_fill(d,trace,side)
                    assert r['fill_ticks']==fill and r['fill_seconds']==d.times[ix];verified+=1
                    if strategy!='C0':
                        values=dict(trade_date=date,nominal_second=second,strategy=strategy,**r)
                        task_rows.append({k:values[k] for k in FIELDS})
                if not valid:break
            if not valid:continue
            # Match the established search-cohort quality rule, not a looser C19-only cohort.
            offsets=(list(range(101)) if second<60 else
                     list(range(29))+[32,40,60,100] if second<1140 else list(range(21))+[24,28,32,40,60,100])
            safe=full_path_safe(d,t)
            if not safe or second%300==0:
                for offset in offsets:
                    for side in [1,-1]:
                        r=simulate(d,t,side,'C_limit_first',signal,c_limit_offset_ticks=offset)
                        if safe:assert r['status']=='filled'
                        if r['status']!='filled':valid=False;reject(f'cohort_offset_{offset}:'+r['reason']);break
                    if not valid:break
                if safe:fast_checks+=1
            if valid:records.extend(task_rows)
        print(date,'latest-model tasks processed',flush=True)
    new=pd.DataFrame(records);assert new.groupby(charts.TASK).size().eq(8).all()
    orders=new[FIELDS].copy()
    date_info=pd.DataFrame(coverage)
    orders=orders.merge(date_info[['trade_date','phase','model_fold','temporally_forward']],on='trade_date',validate='many_to_one')
    assert not orders.duplicated(charts.KEY+['strategy']).any()
    assert orders.groupby(charts.TASK).size().eq(8).all()
    for col in ['task_seconds','initial_ticks']:assert orders.groupby(charts.KEY)[col].nunique().eq(1).all()
    replay=orders[orders.trade_date.isin(latest['test_dates'])].sort_values(charts.KEY+['strategy']).reset_index(drop=True)
    expected=original[original.trade_date.isin(latest['test_dates'])].sort_values(charts.KEY+['strategy']).reset_index(drop=True)
    pd.testing.assert_frame_equal(replay[FIELDS],expected[FIELDS],check_dtype=False,check_exact=True)
    control=orders[orders.strategy=='M'][charts.KEY+['cost_ticks','cost_bp']].rename(columns={'cost_ticks':'market_cost_ticks','cost_bp':'market_cost_bp'})
    orders=orders.merge(control,on=charts.KEY,validate='many_to_one')
    orders['saving_vs_market_bp']=orders.market_cost_bp-orders.cost_bp
    delta=orders.market_cost_ticks-orders.cost_ticks
    orders['win'],orders['tie'],orders['loss']=delta>0,delta==0,delta<0
    mapping_method={('market','immediate'):'immediate_market',('limit','initial'):'initial_limit',
        ('limit','signal'):'signal_limit',('market','signal'):'signal_market',('market','deadline'):'deadline_market'}
    orders['method']=[mapping_method[(k,s)] for k,s in zip(orders.fill_kind,orders.fill_stage)]
    for method in charts.METHODS:orders[method]=orders.method.eq(method)
    assert orders[charts.METHODS].sum(axis=1).eq(1).all() and orders[charts.OUTCOMES].sum(axis=1).eq(1).all()
    assert orders[orders.strategy=='M'].tie.all()
    orders.to_parquet(DEST/'四策略逐任务配对.parquet',index=False)
    pd.DataFrame(signals).to_parquet(DEST/'全部日期信号.parquet',index=False)
    pd.DataFrame(excluded).to_csv(DEST/'全部日期排除任务.csv',index=False,encoding='utf-8-sig')
    date_info.to_csv(DEST/'40日模型与行情覆盖.csv',index=False,encoding='utf-8-sig')
    summary,counts=aggregate(orders,dates)
    charts.DEST=DEST
    charts.draw(summary,scope_label='40日覆盖 · 有效日期等权 · 全样本探索',expected=counts,title_prefix='40日覆盖｜',show_days=True)
    for path in inputs:assert sha(path)==hashes[str(path.relative_to(ROOT))]
    audit=dict(raw_dates=40,valid_dates=int(orders.trade_date.nunique()),original_dates=18,
        same_model_last3_orders_replayed=len(replay),last3_values_exact=True,
        rerun_orders=len(new),total_tasks=len(orders[charts.TASK].drop_duplicates()),
        fills_independently_checked=verified,full_path_sufficiency_checks=fast_checks,feature_boundary_checks=causal_checks,
        model_policy='all dates rerun with latest frozen fold4 W3 model; exploratory, not 40-day forward validation',
        frozen_model=latest,
        counts=counts,input_sha256=hashes,inputs_unchanged=True)
    save_json(DEST/'验收与口径.json',audit)
    report(summary,date_info)
    package()
    print(summary[(summary.side=='买卖各半')&(summary.strategy=='C')][['minutes','grid','days','tasks','cost_bp','saving_vs_market_bp']].to_string(index=False),flush=True)


def aggregate(orders,dates):
    metrics=['cost_bp','saving_vs_market_bp','elapsed_seconds']+charts.METHODS+charts.OUTCOMES
    rows=[];daily_rows=[];coverage=[];wide=[];counts={'minute':{},'second':{}}
    for minutes in [1,19,30,60]:
        for grid in ['minute','second']:
            group=orders[orders.nominal_second<minutes*60]
            if grid=='minute':group=group[group.nominal_second%60==0]
            ntasks=group[charts.TASK].drop_duplicates().shape[0];counts[grid][minutes]=ntasks
            byday=group[charts.TASK].drop_duplicates().groupby('trade_date').size()
            for date in dates:coverage.append(dict(minutes=minutes,grid=grid,trade_date=date,
                planned_tasks=minutes*(1 if grid=='minute' else 60),valid_tasks=int(byday.get(date,0))))
            for side,sub in [('买卖各半',group),('买入',group[group.direction==1]),('卖出',group[group.direction==-1])]:
                for strategy,z in sub.groupby('strategy'):
                    daily=z.groupby('trade_date')[metrics].mean().sort_index()
                    lo,hi=date_block_interval(daily.saving_vs_market_bp)
                    row=dict(minutes=minutes,grid=grid,side=side,strategy=strategy,source_days=40,days=len(daily),tasks=ntasks,orders=len(z),
                        **daily.mean().to_dict(),saving_ci_low=float(lo),saving_ci_high=float(hi))
                    rows.append(row)
                    for date,r in daily.iterrows():daily_rows.append(dict(minutes=minutes,grid=grid,side=side,strategy=strategy,trade_date=date,**r.to_dict()))
                    np.testing.assert_allclose(sum(row[k] for k in charts.METHODS),1,atol=1e-12)
                    np.testing.assert_allclose(sum(row[k] for k in charts.OUTCOMES),1,atol=1e-12)
            r=dict(时间段=f'前{minutes}分钟',任务频率='每分钟' if grid=='minute' else '每秒',覆盖日期数=40,有效日期数=len(byday),有效任务数=ntasks)
            for v in [v for v in rows if v['minutes']==minutes and v['grid']==grid and v['side']=='买卖各半']:
                for label,k in [('成本bp','cost_bp'),('完成秒数','elapsed_seconds'),('较市价节约bp','saving_vs_market_bp'),('胜率','win'),('平局率','tie'),('负率','loss')]:r[v['strategy']+'_'+label]=v[k]
            wide.append(r)
    summary=pd.DataFrame(rows)
    summary.to_csv(DEST/'四策略_分方向汇总.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(daily_rows).to_csv(DEST/'四策略_逐日结果.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(coverage).to_csv(DEST/'逐日逐时段样本覆盖.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(wide).to_csv(DEST/'四策略_8组总表.csv',index=False,encoding='utf-8-sig')
    return summary,counts


def report(summary,date_info):
    lines=['# 四策略40日覆盖版：固定C19，全样本探索','',
        '四策略执行规则、配色和图表布局沿用18日版。根据用户确认，全部日期统一使用最新保存的W3第四折模型重新计算信号和执行，不重新训练或重新选档。旧18日实验保持原文件不变。',
        '该模型训练覆盖2026-07-21至2026-09-08的35个有效开盘日；本版包含训练内回算。07-27虽不在训练日期内，但所用模型来自该日期之后，同样不属于历史可执行预测。整个40日版不是40天向前检验。',
        '仅09-09、09-10、09-11三日位于本模型训练结束之后，并与旧版使用同一模型；三日逐任务执行结果已精确复算核对。其余原18日期改用了统一的最新模型，因此数值可以改变。所有日期和模型关系逐日列明。',
        '07-20没有前一小时行情，各组缺失；07-27从09:43:16.5才有行情，首分钟缺失，其后符合规则的任务纳入19/30/60分钟区间。没有行情的日期不按零值计入均值。',
        '07-28首条行情为09:30:01，超过整点任务允许的0.5秒起点偏移，因此其09:30:00任务缺失；首分钟按分钟采样有37个有效日期，逐秒采样有38个，较长区间有39个。',
        '每个时间段对有有效任务的日期等权，买卖各半。同日的行情覆盖不一定完整，尤其07-27；逐日覆盖表保留全部40日期及计划/有效任务数。',
        '全部日期沿用旧档位研究的共同支持口径：A/B/C0、市价及扫描档位都必须能够结算；首分钟0–100档、前19分钟0–28档及32/40/60/100档，其余0–20档及24/28/32/40/60/100档。完整有效路径用充分条件加速判断，并抽查全档位；其余逐档核实。',
        'C初始及第3秒限价：买价LastPrice−3.8、卖价LastPrice+3.8；其余规则不变，含两条快照延迟、10秒提交兜底市价、旧单先成交优先。',
        '所有图都是已有日期上的探索性描述。95%区间采用5日循环日期块、5,000次重采样，未经模型/档位选择校正。未计真实排队、冲击、手续费或部分成交。','',
        '|时段|频率|有效日期/40|任务数|M成本|A成本|B成本|C19成本|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    s=summary[summary.side=='买卖各半']
    for minutes in [1,19,30,60]:
        for grid in ['minute','second']:
            g=s[(s.minutes==minutes)&(s.grid==grid)].set_index('strategy')
            lines.append(f'|前{minutes}分钟|'+('每分钟' if grid=='minute' else '每秒')+
                f'|{int(g.loc["M","days"])}/40|{int(g.loc["M","tasks"]):,}|'+ '|'.join(f'{g.loc[p,"cost_bp"]:.4f}' for p in charts.ORDER)+'|')
    (DEST/'四策略40日对比说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def package():
    target=ROOT/'result/四策略完整对比_C19_40日覆盖.zip'
    files=sorted(p for p in DEST.iterdir() if p.is_file())
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,arcname='四策略40日覆盖/'+p.name)
    with zipfile.ZipFile(target) as z:
        for p in files:assert hashlib.sha256(z.read('四策略40日覆盖/'+p.name)).hexdigest()==sha(p)
    print('All-date archive verified',len(files),'files',flush=True)


if __name__=='__main__':main()
