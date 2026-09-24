"""Check saved matched results against their old cohorts and market benchmarks."""
from scripts.run_initial_only_c19 import *


def main():
    evidence=[];leader=[]
    for scope in ['18d','40d']:
        folder=DEST/scope;new=pd.read_parquet(folder/'四策略逐任务配对.parquet')
        old=pd.read_parquet(SOURCES[scope]/'四策略逐任务配对.parquet')
        pd.testing.assert_frame_equal(new[TASK].drop_duplicates().sort_values(TASK).reset_index(drop=True),
            old[TASK].drop_duplicates().sort_values(TASK).reset_index(drop=True),check_dtype=False)
        a=new.set_index(KEY+['strategy']).sort_index();b=old.set_index(KEY+['strategy']).reindex(a.index)
        for strategy in ['M','A','B']:
            np.testing.assert_array_equal(a.xs(strategy,level='strategy')[FIELDS[len(KEY)+1:]],b.xs(strategy,level='strategy')[FIELDS[len(KEY)+1:]])
        np.testing.assert_allclose(new.cost_bp,new.direction*(new.fill_ticks-new.initial_ticks)/new.initial_ticks*10000,atol=1e-12,rtol=0)
        np.testing.assert_allclose(new.elapsed_seconds,new.fill_seconds-new.task_seconds,atol=0,rtol=0)
        assert new[charts.METHODS].sum(axis=1).eq(1).all() and new[charts.OUTCOMES].sum(axis=1).eq(1).all()
        summary=pd.read_csv(folder/'四策略_分方向汇总.csv');checked=0
        for row in summary.itertuples():
            g=new[new.strategy.eq(row.strategy)&new.nominal_second.lt(row.minutes*60)]
            if row.grid=='minute':g=g[g.nominal_second.mod(60).eq(0)]
            if row.side=='买入':g=g[g.direction.eq(1)]
            if row.side=='卖出':g=g[g.direction.eq(-1)]
            assert len(g)==row.orders and g[TASK].drop_duplicates().shape[0]==row.tasks
            daily=g.groupby('trade_date')[['cost_bp','saving_vs_market_bp','elapsed_seconds','win','tie','loss']].mean()
            assert len(daily)==row.days
            for col in daily:assert abs(daily[col].mean()-getattr(row,col))<1e-10
            lo,hi=date_block_interval(daily.saving_vs_market_bp)
            np.testing.assert_allclose([lo,hi],[row.saving_ci_low,row.saving_ci_high],atol=1e-10,rtol=0)
            checked+=1
        both=summary[summary.side.eq('买卖各半')]
        for minutes in [1,19,30,60]:
            for grid in ['minute','second']:
                z=both[both.minutes.eq(minutes)&both.grid.eq(grid)].set_index('strategy');c=z.loc['C'];m=z.loc['M']
                leader.append({'版本':'18日原分折模型' if scope=='18d' else '40日最新模型回放',
                    '时段':f'前{minutes}分钟','任务频率':'每分钟' if grid=='minute' else '每秒',
                    '有效日期':int(c.days),'任务数':int(c.tasks),'立即市价成本bp':m.cost_bp,'C初始19转LastPrice成本bp':c.cost_bp,
                    'C较市价节约bp':c.saving_vs_market_bp,'节约95%区间下界bp':c.saving_ci_low,'节约95%区间上界bp':c.saving_ci_high,
                    'C平均完成秒数':c.elapsed_seconds,'成本胜率%':c.win*100,'成本平局率%':c.tie*100,'成本负率%':c.loss*100})
        evidence.append(dict(scope=scope,rows=len(new),summary_rows=checked,all_old_MAB_exact=True,
            exact_cohort=True,cost_elapsed_identities=True,shares_sum100=True,market_intervals_recomputed=True))
    pd.DataFrame(leader).to_csv(DEST/'给leader的16组市价对照.csv',index=False,encoding='utf-8-sig')
    comparison=pd.read_csv(DEST/'新旧C与市价16组比较.csv')
    lines=['# 仅初始19档：18日与40日四策略对比','',
        '新C：T时刻买入LastPrice−3.8、卖出LastPrice+3.8；T+3若未成交，不利预测转市价，否则按当时LastPrice更新限价，T+10未成交转市价。保留双快照延迟。',
        '旧C：初始和第3秒的限价均偏移19档，旧代码与旧实验结果均完整保留。新旧C是两种不同策略，本轮未重新选择最优初始档位。',
        'M立即市价同样保留双快照延迟；四策略沿用各自旧实验的共同任务，买卖各半、日期等权；没有新增剔除。',
        '18日继续使用各日原分折冻结模型；40日版使用最新冻结模型，包含训练内回算，实际有效日期37–39日。两版之间不是同模型、同样本对照。','',
        '|首小时每秒任务|市价成本bp|新C成本bp|旧双阶段C成本bp|新C较市价节约bp|',
        '|---|---:|---:|---:|---:|']
    for r in comparison[comparison.minutes.eq(60)&comparison.grid.eq('second')].itertuples():
        lines.append(f'|{r.scope}|{r.market_cost_bp:.6f}|{r.cost_bp:.6f}|{r.cost_bp_old:.6f}|{r.market_cost_bp-r.cost_bp:+.6f}|')
    lines+=['','两版各8组平均成本均高于立即市价。首小时18日新C比市价多花0.128389bp，40日版多花0.150024bp；成本胜率与平均成本分别判断。',
        '四类静态图在18d/与40d/子目录，均有PNG及SVG：执行成本与节约、平均完成时间、实际成交方式、相对市价胜平负。',
        '给leader的16组市价对照.csv含新C与市价的成本、节约区间、完成时间和胜平负。新旧C与市价16组比较.csv额外对比旧双阶段19档。完整买卖分方向、逐日与逐单结果在各子目录。',
        '122项测试通过；两版192行分方向汇总独立复算，市价/A/B逐笔与旧版完全相同，40日新C逐笔等同上轮三动作实验的LastPrice控制。',
        '区间使用5日循环日期块、5000次重采样，属于已有数据上的描述性分析，未经选择校正；当前L1撮合未模拟队列、冲击、费用或部分成交。']
    (DEST/'结果摘要.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    models=[m for m in json.loads(MODELS.read_text(encoding='utf-8')) if m['window']==3 and m['method']=='selected']
    save_json(DEST/'冻结模型来源.json',dict(models18=models,model40=max(models,key=lambda m:m['fold']),refit=False))
    save_json(DEST/'独立汇总复核.json',evidence)
    config=json.loads((DEST/'实验口径.json').read_text(encoding='utf-8'))
    for path,digest in config['input_sha256'].items():assert sha(ROOT/path)==digest
    package()
    print(pd.DataFrame(leader).to_string(index=False),flush=True)


if __name__=='__main__':main()
