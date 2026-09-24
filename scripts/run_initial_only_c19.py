"""Preserve dual-offset C19; rerun initial19 / T+3 LastPrice on both legacy cohorts."""
import sys
sys.path=[p for p in sys.path if not p.replace('\\','/').endswith('.cache/model-packages')]
from scripts.run_second_ranges import *
from scripts import deliver_four_strategy_comparison as charts
from scripts import run_four_strategy_all_dates as all_dates
from concurrent.futures import ProcessPoolExecutor
import hashlib,zipfile

DEST=OUT/'initial_only_c19'
KEY=charts.KEY;TASK=charts.TASK
FIELDS=all_dates.FIELDS
SOURCES={'18d':OUT/'four_strategy_c19','40d':OUT/'four_strategy_c19_all40'}
SIGNALS={'18d':OUT/'首小时逐秒信号.parquet','40d':SOURCES['40d']/'全部日期信号.parquet'}
COUNTS={'18d':charts.EXPECTED,'40d':{'minute':{1:37,19:720,30:1146,60:2279},'second':{1:2258,19:43233,30:68473,60:135318}}}


def worker(job):
    scope,date,frame,old,signals=job
    cfg=ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,task_range_seconds=3600,observation_seconds=(3.,)))
    d=SessionData(frame,pd.Timestamp(date+' 09:30',tz='Asia/Shanghai'),cfg)
    sig=signals.set_index('nominal_second');reference=old.set_index(KEY+['strategy'])
    tasks=old[old.strategy.eq('C')];records=[];errors=[];checked=0;early=0;market_checks=0
    for nominal,g in tasks.groupby('nominal_second',sort=True):
        signal=int(sig.loc[nominal,'signal']);t=float(g.task_seconds.iloc[0]);taskrows=[]
        i=d.source_index(t);j=int(np.searchsorted(d.times,t,'right'))+1
        assert float(sig.loc[nominal,'task_seconds'])==t
        assert j<d.n and all(d.valid[k][i:j+1].all() for k in ['price','book','queue','volume'])
        assert d.time_edge[i+1:j+1].all() and d.edge['volume'][i+1:j+1].all()
        for side in [-1,1]:
            price=float(d.a[j] if side==1 else d.b[j]);assert (d.qa[j] if side==1 else d.qb[j])>=1
            ticks=side*(price-d.p[i])
            market=dict(trade_date=date,nominal_second=nominal,direction=side,strategy='M',task_seconds=t,
                initial_ticks=float(d.p[i]),fill_seconds=float(d.times[j]),fill_ticks=price,cost_ticks=float(ticks),
                cost_bp=float(ticks/d.p[i]*10000),elapsed_seconds=float(d.times[j]-t),fill_kind='market',fill_stage='immediate',pending_cancelled=False)
            previous_market=reference.loc[(date,nominal,side,'M')]
            for k in FIELDS:
                if k not in KEY+['strategy']:assert market[k]==previous_market[k]
            taskrows.append(market);market_checks+=1
            for strategy,policy in [('A','A_benchmark'),('B','B_observe_first'),('C','C_limit_first')]:
                action=('market' if side*signal>0 else 'lastprice') if strategy=='C' else None
                r=simulate(d,t,side,policy,signal,c_limit_offset_ticks=19 if strategy=='C' else 0,c_signal_action=action)
                trace=r.pop('trace')
                if r['status']!='filled':
                    errors.append(dict(trade_date=date,nominal_second=nominal,direction=side,strategy=strategy,reason=r['reason']))
                    continue
                ix,price=independent_fill(d,trace,side)
                assert r['fill_seconds']==d.times[ix] and r['fill_ticks']==price;checked+=1
                previous=reference.loc[(date,nominal,side,strategy)]
                if strategy!='C' or previous.elapsed_seconds<=3:
                    for k in FIELDS:
                        if k not in KEY+['strategy']:assert r[k]==previous[k]
                    if strategy=='C':early+=1
                if strategy=='C':
                    submissions=[e for e in trace if e['event']=='submit']
                    assert submissions[0]['limit_ticks']==r['initial_ticks']-side*19
                    for e in submissions:
                        if e['stage']=='signal' and e['kind']=='limit':
                            assert e['limit_ticks']==d.p[d.source_index(t+3)]
                    for e in trace:
                        if e['event']=='keep_same_limit':assert e['limit_ticks']==d.p[d.source_index(t+3)]
                row=dict(trade_date=date,nominal_second=nominal,strategy=strategy,**r)
                taskrows.append({k:row[k] for k in FIELDS})
        if len(taskrows)==8:records.extend(taskrows)
    pd.DataFrame(records).to_parquet(DEST/'.replay'/f'{scope}_{date}.parquet',index=False)
    return dict(scope=scope,trade_date=date,independent_fills=checked,market_replays=market_checks,
                preserved_early_C=early,errors=errors,valid_tasks=len(records)//8)


def main():
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'.replay').mkdir(exist_ok=True)
    old_files=[p for folder in SOURCES.values() for p in folder.iterdir() if p.is_file()]
    inputs=list(dict.fromkeys([RAW,MODELS,*SIGNALS.values(),*old_files]))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    if '--finish-only' in sys.argv:
        # A completed worker writes its parquet only after all per-fill checks.
        # Recover presentation after interruption without rerunning those dates.
        previous=json.loads((DEST/'实验口径.json').read_text(encoding='utf-8'))
        assert hashes==previous['input_sha256']
    save_json(DEST/'实验口径.json',dict(initial_offset_ticks=19,signal_limit_offset_ticks=0,
        rule='T initial passive19; T+3 adverse => market, favorable/flat => current LastPrice; T+10 market fallback',
        source_18='original four frozen forward-fold W3 models; existing18 dates',
        source_40='latest frozen fold4 W3 for all40 source dates, includes training backcast',
        models_not_refit=True,offset_not_reselected=True,input_sha256=hashes,
        cohort='same established8 cohorts, require all4 strategies/both sides valid',
        aggregation='date equal, buy/sell half',delay_snapshots=2))
    raw=pd.read_parquet(RAW);original={s:pd.read_parquet(p/'四策略逐任务配对.parquet') for s,p in SOURCES.items()}
    signals={s:pd.read_parquet(p) for s,p in SIGNALS.items()};jobs=[]
    for date,frame in raw.groupby(raw.Date.astype(str),sort=True):
        start=pd.Timestamp(date+' 09:30',tz='Asia/Shanghai')
        frame=frame[(frame.Datetime>=start)&(frame.Datetime<=start+pd.Timedelta(seconds=3620))]
        frame=frame[[k for k in ['Datetime','LastPrice','BidPrice1','AskPrice1','BidVolume1','AskVolume1','Volume','source_row'] if k in frame]].copy()
        for scope,old in original.items():
            group=old[old.trade_date.eq(date)]
            if not group.empty:jobs.append((scope,date,frame,group[FIELDS].copy(),signals[scope][signals[scope].trade_date.eq(date)][['nominal_second','task_seconds','signal']].copy()))
    checks=[]
    if '--finish-only' in sys.argv:
        for scope,date,_,old,_ in jobs:
            complete=pd.read_parquet(DEST/'.replay'/f'{scope}_{date}.parquet')
            assert len(complete)==len(old) and complete.groupby(TASK).size().eq(8).all()
            checks.append(dict(scope=scope,trade_date=date,independent_fills=int(complete.strategy.ne('M').sum()),
                market_replays=int(complete.strategy.eq('M').sum()),
                preserved_early_C=int((old.strategy.eq('C')&old.elapsed_seconds.le(3)).sum()),
                errors=[],valid_tasks=len(complete)//8,recovered_completed_worker=True))
    else:
        with ProcessPoolExecutor(max_workers=2) as pool:
            # Bound in-flight Windows IPC data instead of queuing every date at once.
            for start in range(0,len(jobs),4):
                for r in pool.map(worker,jobs[start:start+4]):
                    checks.append(r);print(r['scope'],r['trade_date'],'verified',r['valid_tasks'],'tasks',flush=True)
    comparisons=[]
    for scope,old in original.items():
        folder=DEST/scope;folder.mkdir(exist_ok=True)
        orders=pd.concat([pd.read_parquet(DEST/'.replay'/f'{scope}_{r["trade_date"]}.parquet') for r in checks if r['scope']==scope],ignore_index=True)
        assert orders.groupby(TASK).size().eq(8).all() and not orders.duplicated(KEY+['strategy']).any()
        m=orders[orders.strategy.eq('M')][KEY+['cost_bp','cost_ticks']].rename(columns={'cost_bp':'market_cost_bp','cost_ticks':'market_cost_ticks'})
        orders=orders.merge(m,on=KEY,validate='many_to_one')
        orders['saving_vs_market_bp']=orders.market_cost_bp-orders.cost_bp
        diff=orders.market_cost_ticks-orders.cost_ticks
        orders['win']=diff>1e-10;orders['tie']=abs(diff)<=1e-10;orders['loss']=diff< -1e-10
        methods={('market','immediate'):'immediate_market',('limit','initial'):'initial_limit',('limit','signal'):'signal_limit',('market','signal'):'signal_market',('market','deadline'):'deadline_market'}
        orders['method']=[methods[x] for x in zip(orders.fill_kind,orders.fill_stage)]
        for method in charts.METHODS:orders[method]=orders.method.eq(method)
        orders.to_parquet(folder/'四策略逐任务配对.parquet',index=False)
        all_dates.DEST=folder
        dates=sorted(raw.Date.astype(str).unique()) if scope=='40d' else sorted(old.trade_date.unique())
        summary,counts=all_dates.aggregate(orders,dates,source_days=40 if scope=='40d' else 18)
        assert counts==COUNTS[scope]
        if scope=='40d':
            l0=pd.read_parquet(OUT/'three_action_threshold/阈值策略逐单结果.parquet',filters=[('policy','==','L0')])
            c=orders[orders.strategy.eq('C')].set_index(KEY).sort_index();l0=l0.set_index(KEY).loc[c.index]
            np.testing.assert_array_equal(c[['cost_bp','fill_ticks','fill_seconds','elapsed_seconds']],l0[['cost_bp','fill_ticks','fill_seconds','elapsed_seconds']])
        charts.DEST=folder;charts.NAMES['C']='C 初始19档→3秒LastPrice'
        charts.draw(summary,scope_label='18日等权' if scope=='18d' else '40日覆盖 · 有效日期等权 · 探索',
            expected=counts,title_prefix='18日｜' if scope=='18d' else '40日覆盖｜',show_days=True,
            c_description='初始19档→3秒LastPrice',c_footer='C仅初始±3.8；3秒限价=当时LastPrice')
        report(scope,folder,summary)
        oldc=old[old.strategy.eq('C')].set_index(KEY)
        newc=orders[orders.strategy.eq('C')].set_index(KEY)
        pairs=newc[['cost_bp','market_cost_bp','elapsed_seconds']].join(oldc[['cost_bp','elapsed_seconds']],rsuffix='_old',validate='one_to_one').reset_index()
        pairs['saving_vs_old_C']=pairs.cost_bp_old-pairs.cost_bp
        pairs.to_parquet(folder/'新旧C逐单配对.parquet',index=False)
        for minutes in [1,19,30,60]:
            for grid in ['minute','second']:
                g=pairs[pairs.nominal_second.lt(minutes*60)&((pairs.nominal_second%60==0) if grid=='minute' else True)]
                daily=g.groupby('trade_date')[['cost_bp','cost_bp_old','market_cost_bp','saving_vs_old_C']].mean()
                lo,hi=date_block_interval(daily.saving_vs_old_C)
                comparisons.append(dict(scope=scope,minutes=minutes,grid=grid,tasks=len(g)//2,days=len(daily),
                    **daily.mean().to_dict(),saving_vs_old_low=lo,saving_vs_old_high=hi))
        errors=[e for r in checks if r['scope']==scope for e in r['errors']]
        pd.DataFrame(errors,columns=KEY+['strategy','reason']).to_csv(folder/'新增排除任务.csv',index=False,encoding='utf-8-sig')
    compare=pd.DataFrame(comparisons);compare.to_csv(DEST/'新旧C与市价16组比较.csv',index=False,encoding='utf-8-sig')
    for p in inputs:assert sha(p)==hashes[str(p.relative_to(ROOT))]
    save_json(DEST/'验收结果.json',dict(checks=checks,old_files_and_inputs_unchanged=True,
        independent_fills=sum(r['independent_fills'] for r in checks),
        immediate_market_exact_replays=sum(r['market_replays'] for r in checks),
        all40_matches_previous_LastPrice_control=True,all8_counts_match_both_scopes=True,
        source18_tasks=len(original['18d'])//8,source40_tasks=len(original['40d'])//8,
        fixed_initial19_no_offset_search=True))
    package()
    print(compare.to_string(index=False),flush=True)


def report(scope,folder,summary):
    text=['# 四策略比较：仅初始偏移19档，第3秒限价恢复LastPrice','',
        '原初始和第3秒均偏移19档的C实验完整保留在原目录；本实验单独保存，不覆盖也不否定其独立策略结果。',
        'M：任务起点提交市价；A：起点LastPrice限价、10秒兜底；B：先观察3秒，不利方向市价、其余当时LastPrice限价、10秒兜底。',
        'C：起点买入LastPrice−3.8、卖出LastPrice+3.8；3秒若仍未成交，预测对执行不利则市价，否则限价=第3秒当时LastPrice，不再加减3.8。',
        '保留双快照延迟；撤换到达前旧单仍能成交；同刻旧单成交优先；10秒提交兜底市价。限价触发后按限价结算，市价按到达对手价结算。',
        '本轮固定初始19档，不重新搜索最优初始档位；旧19档优选同时改变两个阶段，不能当作本规则下初始19档最优的证据。',
        ('18日版沿用原四折按日期向前的冻结W3模型，各日使用其原折模型；日期此前均参与研究，仍非全新样本。' if scope=='18d' else
         '40日版所有日期沿用最新冻结W3第四折模型，含35个训练日期。缺行情日期不补零，首分钟分钟/秒任务分别37/38有效日，其余39日；不是40日样本外验证。'),
        '8组均沿用各自旧实验共同任务，不扩充样本；任务独立模拟买卖各1手，先逐日等权，买卖各半。18日与40日模型和样本不同，不能用两个平均值之差解释模型提升。',
        '成本=方向×(成交价−任务起点LastPrice)/起点LastPrice×10000。节约=立即市价成本−本策略成本；正数表示更省。',
        '95%区间为5日循环日期块5000次重采样，未经档位/模型选择校正。未计真实排队、部分成交、冲击及手续费。','',
        '|范围|频率|有效日|任务数|M成本bp|A成本bp|B成本bp|C成本bp|C较M节约bp|95%区间bp|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for minutes in [1,19,30,60]:
        for grid in ['minute','second']:
            g=summary[summary.side.eq('买卖各半')&summary.minutes.eq(minutes)&summary.grid.eq(grid)].set_index('strategy');r=g.loc['C']
            text.append(f'|前{minutes}分钟|'+('每分钟' if grid=='minute' else '每秒')+f'|{int(r.days)}|{int(r.tasks)}|'+
                '|'.join(f'{g.loc[p,"cost_bp"]:.4f}' for p in charts.ORDER)+f'|{r.saving_vs_market_bp:+.4f}|[{r.saving_ci_low:+.4f},{r.saving_ci_high:+.4f}]|')
    text+=['','四张图：执行成本与相对市价节约、从T到成交的完成时间、实际成交方式、相对市价的成本胜平负。胜率不是方向预测准确率。',
           'CSV含全部方向；Parquet为逐单数据；新旧C配对另存。同任务同时满足双方限价和市价结果有效，不能混合不同覆盖率。']
    (folder/'四策略对比说明.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def package():
    target=ROOT/'result/四策略对比_仅初始19档_18日与40日.zip'
    files=sorted(p for p in DEST.rglob('*') if p.is_file() and '.replay' not in p.parts)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,'仅初始19档/'+p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for p in files:assert hashlib.sha256(z.read('仅初始19档/'+p.relative_to(DEST).as_posix())).hexdigest()==sha(p)
    print('Verified package',len(files),'files',flush=True)


if __name__=='__main__':main()
