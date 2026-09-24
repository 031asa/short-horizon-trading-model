"""Signal-strength / three-action curves with explicit reference policies."""
from scripts.run_rebased_thresholds import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zipfile,hashlib

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
    'axes.unicode_minus':False,'svg.fonttype':'path','font.size':11,
    'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
EDGES=[-1,-.3,-.2,-.15,-.1,-.075,-.05,-.025,0,.025,.05,.075,.1,.15,.2,.3,1]
COLORS={'market':'#247E87','lastprice':'#D68B29','passive':'#865BB4'}
NAMES={'market':'第3秒转市价','lastprice':'第3秒改挂LastPrice','passive':'第3秒挂被动19档'}


def calculate_new_baseline_bins(folder):
    info=pd.read_parquet(folder/'任务与信号.parquet').set_index(KEY).sort_index()
    orders=pd.read_parquet(folder/'阈值与基准_逐单结果.parquet',filters=[('policy','in',['C_new','R_market','R_lastprice','R_passive'])])
    baseline=orders[orders.policy.eq('C_new')].set_index(KEY).loc[info.index]
    rows=[];dayrows=[]
    for action in legacy.ACTIONS:
        branch=orders[orders.policy.eq('R_'+action)].set_index(KEY).loc[info.index]
        f=info[['survivor3','favorable_score']].copy()
        f['saving_vs_C_new']=baseline.cost_bp-branch.cost_bp
        f['cost_bp']=branch.cost_bp;f['baseline_cost_bp']=baseline.cost_bp
        f['bin']=pd.cut(f.favorable_score,EDGES,include_lowest=True,right=False)
        f=f.reset_index();f=f[f.survivor3]
        for minutes in [1,19,30,60]:
            for grid in ['minute','second']:
                g=f[f.nominal_second.lt(minutes*60)&((f.nominal_second%60==0) if grid=='minute' else True)]
                for label,h in g.groupby('bin',observed=True):
                    daily=h.groupby('trade_date')[['saving_vs_C_new','cost_bp','baseline_cost_bp']].mean()
                    lo,hi=date_block_interval(daily.saving_vs_C_new)
                    meta=dict(action=action,minutes=minutes,grid=grid,bin=str(label),left=label.left,right=label.right)
                    rows.append(dict(**meta,orders=len(h),days=len(daily),buy_orders=int(h.direction.eq(1).sum()),
                        **daily.mean().to_dict(),saving_low=lo,saving_high=hi))
                    q=daily.reset_index()
                    for k,v in meta.items():q[k]=v
                    dayrows.append(q)
    result=pd.DataFrame(rows)
    result.to_csv(folder/'信号强弱分箱_相对新C.csv',index=False,encoding='utf-8-sig')
    pd.concat(dayrows,ignore_index=True).to_csv(folder/'信号强弱分箱_相对新C逐日.csv',index=False,encoding='utf-8-sig')
    # Same cohorts, absolute costs and support as the independent market-referenced bins.
    previous=pd.read_csv(folder/'信号强弱分箱_三动作成本.csv')
    merged=result.merge(previous,on=['action','minutes','grid','bin'],validate='one_to_one',suffixes=('_new','_old'))
    assert len(merged)==len(result)==len(previous)
    assert merged.orders_new.eq(merged.orders_old).all() and merged.days_new.eq(merged.days_old).all()
    np.testing.assert_allclose(merged.cost_bp_new,merged.cost_bp_old,atol=1e-8,rtol=0)
    return result


def draw_bins(folder,scope,bins,reference):
    b=bins[bins.minutes.eq(60)&bins.grid.eq('second')]
    support=b[b.action.eq('market')].sort_values('left');labels=support['bin'].tolist();x=np.arange(len(labels))
    fig,(ax,count)=plt.subplots(2,1,figsize=(15.5,8.7),sharex=True,gridspec_kw={'height_ratios':[3,1]})
    relative_market=reference=='market'
    metric='saving_vs_3s_market' if relative_market else 'saving_vs_C_new'
    zero='第3秒转市价' if relative_market else '新C：不利转市价，其余挂LastPrice'
    for action in legacy.ACTIONS:
        g=b[b.action.eq(action)].set_index('bin').loc[labels]
        if relative_market and action=='market':
            ax.plot(x,g[metric],ls='--',color=COLORS[action],label=NAMES[action]+'（零线）')
        else:
            ax.plot(x,g[metric],'o-',color=COLORS[action],label=NAMES[action])
            ax.fill_between(x,g.saving_low,g.saving_high,color=COLORS[action],alpha=.13)
    if not relative_market:ax.axhline(0,ls='--',color='#555',label='新C基准（零线）')
    boundary=labels.index('[0.0, 0.025)')-.5
    ax.axvline(boundary,color='#AAA',ls=':',lw=1)
    ax.set_ylabel('相对'+('第3秒市价' if relative_market else '新C')+'的节约（bp；正值更省）')
    ax.grid(alpha=.16);ax.legend(loc='best',fontsize=10)
    scope_name='18日原分折模型' if scope=='18d' else '40日覆盖 · 最新冻结模型'
    ax.set_title(scope_name+'｜信号强弱分箱与三种执行动作\n首小时每秒任务；仅第3秒尚未成交的同一批订单',fontsize=15,pad=14)
    count.bar(x,support.orders,color='#9EAFBD');count.set_ylabel('方向订单数')
    count.set_xticks(x,labels,rotation=40,ha='right')
    count.set_xlabel('有利方向概率差 F：买入 p跌−p涨；卖出 p涨−p跌。右侧为更有利的方向倾向；不是预测跌幅。')
    n=int(support.orders.sum());nd=int(support.days.max())
    fig.text(.025,.038,f'初始均挂19档；3秒后三动作分别回放；10秒兜底，保留执行延迟。零线＝{zero}。',fontsize=9,color='#444')
    fig.text(.025,.017,f'{nd}个有效日 / {n:,}笔未成交方向订单；分箱内日期等权、实际买卖构成。阴影为95%日期块区间，未经选择校正。',fontsize=9,color='#444')
    fig.tight_layout(rect=[0,.075,1,1])
    name='01_三动作成本_相对3秒市价' if relative_market else '02_三动作成本_相对新C'
    for suffix in ['png','svg']:fig.savefig(folder/(name+'.'+suffix),dpi=210,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    allrows=[];notes=[]
    for scope in ['18d','40d']:
        folder=DEST/scope
        newbins=calculate_new_baseline_bins(folder)
        draw_bins(folder,scope,pd.read_csv(folder/'信号强弱分箱_三动作成本.csv'),'market')
        draw_bins(folder,scope,newbins,'C_new')
        s=pd.read_csv(folder/'首小时逐秒阈值总表.csv').set_index('policy')
        audit=json.loads((folder/'验收与选择.json').read_text(encoding='utf-8'))
        best=audit['selected_policy'];r=s.loc[best]
        for label in ['C_new','M']+[f'T{t:g}' for t in legacy.THRESHOLDS]:
            q=s.loc[label]
            allrows.append({'版本':scope,'规则':'新C（无强弱阈值）' if label=='C_new' else '立即市价M' if label=='M' else 'τ='+label[1:],
                '平均成本bp':q.cost_bp,'较新C节约bp':q.saving_vs_C_new,'较市价M节约bp':q.saving_vs_M,
                '平均完成秒数':q.elapsed_seconds,'10秒兜底比例%':q.deadline_market*100,
                '较新C节约区间下界bp':q.saving_vs_C_new_low,'较新C节约区间上界bp':q.saving_vs_C_new_high})
        notes.append(f'{scope}：首小时每秒任务最低阈值为{best[1:]}，成本{r.cost_bp:.6f}bp；较新C节约{r.saving_vs_C_new:+.6f}bp，95%区间[{r.saving_vs_C_new_low:+.6f},{r.saving_vs_C_new_high:+.6f}]；较立即市价节约{r.saving_vs_M:+.6f}bp，区间[{r.saving_vs_M_low:+.6f},{r.saving_vs_M_high:+.6f}]。')
    pd.DataFrame(allrows).to_csv(DEST/'给leader的阈值扫描.csv',index=False,encoding='utf-8-sig')
    lines=['# 新C基准下的三动作阈值实验','',
        '无阈值基准是新C：初始19档，3秒不利方向转市价，有利/不变方向限价=当时LastPrice。无阈值不是τ=0；若按本实验参数表示，新C对应不启用19档分支，即阈值无穷大。',
        '阈值策略：初始19档不变；3秒不利方向转市价，不变信号挂LastPrice；有利方向若F≥τ，按当时LastPrice被动19档，否则LastPrice。F=方向×(p跌−p涨)，不代表跌幅。',
        '固定τ=0/.025/.05/.075/.10/.15/.20/.30。单一阈值按首小时逐秒全部订单日期等权成本选择，各时段不单独择优。原模型、初始19档及延迟/兜底不变。',
        '18日按原四折模型新增重放三种动作；40日复用已逐笔核验且输入指纹相同的三动作回放，重新计算阈值与新C比较。改变基准不会改变同一阈值策略的实际成交成本。旧实验全部保留。','',*notes,'',
        '本批τ=0等同旧双阶段19档。虽然相对新C改善，但最低点退回两动作规则，不能称为已经找到保留LastPrice中间档的更优三档阈值。',
        '每版01图采用用户指定的分箱曲线格式：零线=第3秒市价。每版02图改以新C为零线，便于直接检查弱/强信号下的动作增益。图中仅含T+3尚未成交订单；阈值汇总表包含提前成交订单，两个样本口径分开。',
        '图的分箱内三动作使用完全相同的订单，各日内保留该箱实际买卖构成，再日期等权。主策略表则对完整配对任务按日期等权、买卖各半。支持数与绝对成本已与原市价参考分箱逐项核对。',
        '40日源数据中首小时39日有效；包含模型训练内回算。18日也已用于前期研究。区间为循环5日块5000次重采样，未经模型、档位、阈值选择校正；不是独立实盘验证。',
        '限价成交按既定口径在限价结算，市价按到达对手价结算；无队列、费用、冲击及部分成交模拟。']
    (DEST/'结果说明.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    save_json(DEST/'曲线复核.json',dict(both_reference_bin_counts_and_absolute_costs_match=True,
        chart_market_reference_is_Tplus3_not_T=True,all_original_inputs_unchanged=True))
    package()
    print('\n'.join(notes),flush=True)


def package():
    target=ROOT/'result/新C基准_三动作阈值实验_18日与40日.zip'
    files=sorted(p for p in DEST.rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.relative_to(DEST).parts))
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,'新C阈值实验/'+p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for p in files:assert hashlib.sha256(z.read('新C阈值实验/'+p.relative_to(DEST).as_posix())).hexdigest()==sha(p)
    print('Package verified',len(files),'files',flush=True)


if __name__=='__main__':main()
