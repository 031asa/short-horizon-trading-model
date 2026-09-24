"""Date-balanced conditional cost optima, linear fit, and forward diagnostics."""
from scripts.run_probability_offset_relation import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zipfile,hashlib
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'svg.fonttype':'path','font.size':11})
BINS=np.arange(20);THRESHOLDS=np.arange(10,41,5)/100


def wls(x,y,w):
    valid=np.isfinite(x)&np.isfinite(y)&(w>0)
    x=x[valid];y=y[valid];w=w[valid]
    if len(x)<3 or np.ptp(x)<1e-9:return np.full(3,np.nan)
    mx=np.average(x,weights=w);my=np.average(y,weights=w)
    b=np.sum(w*(x-mx)*(y-my))/np.sum(w*(x-mx)**2);a=my-b*mx
    total=np.sum(w*(y-my)**2);r2=1-np.sum(w*(y-a-b*x)**2)/total if total>0 else np.nan
    return np.array([a,b,r2])


def fit_bins(daily):
    out=daily.groupby('bin').agg(probability=('probability','mean'),orders=('orders','sum'),days=('trade_date','nunique'))
    costs=daily.groupby('bin')[list(GRID)].mean()
    out['best_k']=costs.idxmin(axis=1).astype(int);out['best_cost']=costs.min(axis=1)
    out['supported']=(out.orders>=100)&(out.days>=5)
    f=out[out.supported]
    coef=wls(f.probability.to_numpy(),f.best_k.to_numpy(),f.days.to_numpy())
    return out,coef


def draw_thresholds(thresholds):
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for row,scope in enumerate(['18d','40d']):
        for col,side in enumerate([1,-1]):
            ax=axes[row,col];part=thresholds[thresholds.scope.eq(scope)&thresholds.direction.eq(side)]
            for group,color,label in [('le','#237D94','p_up ≤ 阈值'),('gt','#D28832','p_up > 阈值')]:
                q=part[part.group.eq(group)].sort_values('threshold')
                # Unsupported optima shown as isolated crosses, never joined as evidence.
                ax.plot(q.threshold,q.best_k.where(q.supported),'-o',color=color,label=label)
                low=q[~q.supported&q.orders.gt(0)];ax.scatter(low.threshold,low.best_k,marker='x',color=color,alpha=.4)
            support=part[part.group.eq('le')].sort_values('threshold')
            ax.set_xticks(THRESHOLDS,[f'{t:.2f}\n≤组{int(n):,}笔' for t,n in zip(support.threshold,support.orders)],fontsize=9)
            ax.set(xlim=(.08,.42),ylim=(-3,63),ylabel='组内最低成本档位（0–60档）',xlabel='上涨概率分界 τ（两组为嵌套样本）')
            ax.set_title(f'{"18日" if scope=="18d" else "40日覆盖"}｜{"买入" if side==1 else "卖出"}');ax.grid(alpha=.15);ax.legend(fontsize=10)
    fig.suptitle('上涨概率阈值扫描：0.10–0.40，步长0.05\n相同任务、相同p_up分组，分别查看买卖两侧的条件最低成本档位',fontsize=15)
    fig.supxlabel('下方笔数为p_up≤阈值组，买卖相同；总数18日每侧57,995笔、40日每侧117,850笔。×表示不足100笔或5日，不据此判断规律。\n初始19档；只分析两侧到3秒均未成交时的强制限价动作，保留延迟和10秒兜底。最低点为全样本探索，不是验证后的阈值策略。',fontsize=10)
    for ext in ['png','svg']:fig.savefig(DEST/f'上涨概率0.1到0.4阈值扫描.{ext}',dpi=180)
    plt.close(fig)


def main():
    fitrows=[];binrows=[];policyrows=[];thresholdrows=[];forwardrows=[];checks=[]
    allfig={}
    for scope in SOURCES:
        folder=DEST/scope;info=pd.read_parquet(folder/'共同任务与概率.parquet').set_index(KEY)
        parts=[]
        for path in sorted((folder/'逐日逐单').glob('*.parquet')):
            q=pd.read_parquet(path);wide=q.pivot(index=KEY,columns='k',values='cost_bp').sort_index()
            assert list(wide.columns)==list(GRID) and wide.notna().all().all()
            meta=info.loc[wide.index,['p_up','p_down','signal','market3_cost_bp']]
            parts.append(wide.join(meta))
        data=pd.concat(parts).reset_index();data['bin']=np.minimum(np.floor(data.p_up*20).astype(int),19)
        assert data.groupby(TASK).size().eq(2).all()
        both=data.groupby(['direction','bin']).size().unstack('direction',fill_value=0)
        np.testing.assert_array_equal(both[1],both[-1])
        daily=[]
        for side in [1,-1]:
            z=data[data.direction.eq(side)];g=z.groupby(['trade_date','bin'])
            dd=g[list(GRID)].mean().join(g.agg(probability=('p_up','mean'),orders=('p_up','size'))).reset_index();dd['direction']=side;daily.append(dd)
        daily=pd.concat(daily,ignore_index=True);daily.to_csv(folder/'概率分箱逐日成本.csv',index=False,encoding='utf-8-sig')
        dates=sorted(data.trade_date.unique());n=len(dates);rng=np.random.default_rng(20260923)
        starts=rng.integers(0,n,(5000,int(np.ceil(n/5))));draw=((starts[...,None]+np.arange(5))%n).reshape(5000,-1)[:,:n]
        for side in [1,-1]:
            z=data[data.direction.eq(side)].copy();dd=daily[daily.direction.eq(side)].copy()
            bins,coef=fit_bins(dd);a,b,r2=coef
            support=bins.index[bins.supported].to_numpy();cube=np.full((n,len(support),len(GRID)),np.nan);px=np.full((n,len(support)),np.nan)
            for di,date in enumerate(dates):
                q=dd[dd.trade_date.eq(date)].set_index('bin').reindex(support)
                cube[di]=q[list(GRID)].to_numpy();px[di]=q.probability.to_numpy()
            samples=[]
            for indices in draw:
                mask=np.isfinite(px[indices]);weights=mask.sum(axis=0);means=np.nansum(cube[indices],axis=0)/np.maximum(weights[:,None],1)
                x=np.nansum(px[indices],axis=0)/np.maximum(weights,1);y=means.argmin(axis=1)
                samples.append(wls(x,y,weights))
            samples=np.asarray(samples);lo,hi=np.nanquantile(samples[:,1],[.025,.975])
            row=dict(scope=scope,direction=side,intercept=a,slope=b,r_squared=r2,slope_low=lo,slope_high=hi,supported_bins=len(support),orders=len(z),days=n,
                     expected_sign='negative' if side==1 else 'positive',boundary_bins=int(bins.loc[support,'best_k'].isin([0,60]).sum()))
            fitrows.append(row)
            bx=bins.reset_index();bx['scope']=scope;bx['direction']=side;binrows.append(bx)
            constant=int(z.groupby('trade_date')[list(GRID)].mean().mean().idxmin())
            predicted=np.clip(np.floor(a+b*z.p_up.to_numpy()+.5),0,60).astype(int)
            costs=z[list(GRID)].to_numpy();linear=costs[np.arange(len(z)),predicted]
            for name,value in [('linear_all_dates',linear),('best_constant_all_dates',costs[:,constant]),('constant19',costs[:,19]),('market3',z.market3_cost_bp.to_numpy())]:
                frame=pd.DataFrame({'trade_date':z.trade_date.to_numpy(),'cost_bp':value});frame['policy']=name;frame['scope']=scope;frame['direction']=side
                d=frame.groupby(['scope','direction','trade_date','policy'],as_index=False).cost_bp.mean();policyrows.append(d)
            # Explicit exploratory threshold bins: <=tau and >tau, common p_up for sides.
            for tau in THRESHOLDS:
                for lower in [True,False]:
                    h=z[z.p_up.le(tau) if lower else z.p_up.gt(tau)]
                    if h.empty:
                        thresholdrows.append(dict(scope=scope,direction=side,threshold=tau,group='le' if lower else 'gt',orders=0,days=0,best_k=np.nan,cost_bp=np.nan,supported=False));continue
                    means=h.groupby('trade_date')[list(GRID)].mean().mean();k=int(means.idxmin())
                    thresholdrows.append(dict(scope=scope,direction=side,threshold=tau,group='le' if lower else 'gt',orders=len(h),days=h.trade_date.nunique(),best_k=k,cost_bp=float(means[k]),supported=len(h)>=100 and h.trade_date.nunique()>=5))
            # Earlier dates choose the bin-fit line and constant, subsequent dates evaluate.
            folds=[(8,13),(13,18)] if scope=='18d' else [(20,25),(25,30),(30,35),(35,n)]
            for fi,(train_end,test_end) in enumerate(folds):
                tr=dates[:train_end];te=dates[train_end:test_end];bb,cc=fit_bins(dd[dd.trade_date.isin(tr)])
                assert max(tr)<min(te) and np.isfinite(cc[:2]).all()
                train=z[z.trade_date.isin(tr)];test=z[z.trade_date.isin(te)]
                k=int(train.groupby('trade_date')[list(GRID)].mean().mean().idxmin());ks=np.clip(np.floor(cc[0]+cc[1]*test.p_up.to_numpy()+.5),0,60).astype(int)
                matrix=test[list(GRID)].to_numpy();frame=pd.DataFrame(dict(trade_date=test.trade_date.to_numpy(),linear=matrix[np.arange(len(test)),ks],constant=matrix[:,k],constant19=matrix[:,19],market3=test.market3_cost_bp.to_numpy()))
                g=frame.groupby('trade_date').mean().reset_index();g['scope']=scope;g['direction']=side;g['fold']=fi;g['intercept']=cc[0];g['slope']=cc[1];g['constant_k']=k;g['train_last']=tr[-1];forwardrows.append(g)
            allfig[(scope,side)]=(bins,coef,samples,constant,row)
        checks.append(dict(scope=scope,paired_tasks=len(data)//2,same_probability_bin_counts=True,all61_offsets_common=True))
    fits=pd.DataFrame(fitrows);bins=pd.concat(binrows);policies=pd.concat(policyrows);thresholds=pd.DataFrame(thresholdrows);forward=pd.concat(forwardrows)
    for frame,name in [(fits,'线性拟合'),(bins,'概率分箱最小成本档位'),(policies,'全样本条件策略逐日'),(thresholds,'上涨概率阈值扫描'),(forward,'按日期前推条件策略')]:frame.to_csv(DEST/f'{name}.csv',index=False,encoding='utf-8-sig')
    comparisons=[]
    for (scope,side),g in forward.groupby(['scope','direction']):
        diff=g.constant-g.linear;lo,hi=date_block_interval(diff)
        comparisons.append(dict(scope=scope,direction=side,days=len(g),linear=g.linear.mean(),constant=g.constant.mean(),saving_vs_constant=diff.mean(),low=lo,high=hi,constant19=g.constant19.mean(),market3=g.market3.mean()))
    compare=pd.DataFrame(comparisons);compare.to_csv(DEST/'前推比较汇总.csv',index=False,encoding='utf-8-sig')
    for scope in SOURCES:
        fig,axes=plt.subplots(2,2,figsize=(15,11),layout='constrained')
        for col,side in enumerate([1,-1]):
            table,coef,samples,k,row=allfig[(scope,side)];a,b,r2=coef;ok=table.supported
            ax=axes[0,col];ax.scatter(table.loc[~ok,'probability'],table.loc[~ok,'best_k'],marker='x',color='#aaaaaa',label='稀疏箱，不用于拟合')
            ax.scatter(table.loc[ok,'probability'],table.loc[ok,'best_k'],s=35+4*np.sqrt(table.loc[ok,'orders']),alpha=.65,color='#277a91',label='支持充足的箱内最低成本档位')
            xx=np.linspace(table.loc[ok,'probability'].min(),table.loc[ok,'probability'].max(),100)
            yy=samples[:,0,None]+samples[:,1,None]*xx;low,high=np.nanquantile(yy,[.025,.975],axis=0)
            ax.fill_between(xx,low,high,color='#cf7d28',alpha=.15,label='日期块重采样拟合线区间')
            ax.plot(xx,a+b*xx,color='#cf7d28',lw=2,label=f'k={a:.1f}{b:+.1f}×p_up；R²={r2:.2f}')
            ax.axhline(k,color='#777777',ls='--',label=f'全样本最优固定档位 {k}')
            ax.set(xlim=(0,1),ylim=(-5,65),xlabel='上涨概率 p_up（两侧统一）',ylabel='条件最低成本档位 k')
            label='买入：预期负斜率' if side==1 else '卖出：预期正斜率'
            ax.set_title(f'{label}\n斜率95%区间 [{row["slope_low"]:.1f}, {row["slope_high"]:.1f}]');ax.legend(fontsize=8,loc='best');ax.grid(alpha=.18)
            ax=axes[1,col];g=forward[forward.scope.eq(scope)&forward.direction.eq(side)].sort_values('trade_date')
            diff=g.constant-g.linear;ax.bar(np.arange(len(g)),diff,color=np.where(diff>=0,'#198573','#bc5260'))
            r=compare[compare.scope.eq(scope)&compare.direction.eq(side)].iloc[0]
            ax.axhline(0,color='#555555',lw=.8);ax.axhline(r.saving_vs_constant,color='#333333',ls='--',label=f'日均节约 {r.saving_vs_constant:+.4f} bp')
            ax.set_xticks(np.arange(len(g)),[d[5:] for d in g.trade_date],rotation=45);ax.set_ylabel('线性调档较同期选出固定档位的节约（bp）');ax.set_xlabel('随后检验日期');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.18)
        title='18日' if scope=='18d' else '40日覆盖（39日有效；模型含训练内回算）'
        fig.suptitle(f'{title}｜上涨概率与被动档位的线性关系\n初始19档 · 首小时逐秒 · 买卖到3秒均未成交的相同任务 · 3秒强制限价诊断',fontsize=15)
        fig.supxlabel('上图：0.05概率分箱，至少100笔且5日参与拟合，按有效日期数加权；每次日期块抽样重新选箱内最低档位。\n下图：只用较早日期拟合线/选择固定档位，后续日期比较；没有重训原模型。不是独立样本外策略验证，也不是完整C策略收益。\n0/60档可能为搜索边界，稀疏箱不代表稳定最优。真正使用时档位四舍五入并限制在0–60；不按事后涨跌筛样本。',fontsize=9)
        for ext in ['png','svg']:fig.savefig(DEST/f'{scope}_概率档位线性关系.{ext}',dpi=180)
        plt.close(fig)
    save_json(DEST/'分析验收.json',dict(checks=checks,bootstrap_repeats=5000,block_dates=5,bin_width=.05,supported_min_orders=100,supported_min_days=5,regression_weight='number of effective dates in bin',thresholds=THRESHOLDS.tolist(),threshold_interpretation='raw p_up cut, <= vs >, descriptive',no_dynamic_strategy_deployed=True))
    draw_thresholds(thresholds)
    report=['# 上涨概率与档位线性关系','', '买卖使用同一p_up和相同任务，只保留两侧到3秒均未成交；均强制回放3秒限价以比较挂档，不沿用原信号市价筛选。初始19档、两快照延迟、10秒兜底不变。成本按同一起点LastPrice计算。',
      '扫描0–60整数档。每个0.05概率箱内先逐日平均各档成本，再日期等权，取最低档（并列取小档）。箱内至少100笔且5日才参与拟合，以有效日期数加权；不回归每笔事后最优档。重采样循环5日块5000次，每次重新选箱内最低档再拟合。R²描述箱内最优档位的线性近似，不是价格预测能力。',
      '阈值0.10–0.40步长0.05按原始上涨概率切分<=与>两组，输出各侧成本和支持数；不将其解释为概率差或交易阈值。买卖同组订单数一致。',
      '日期前推：18日前8日选择、后5日检验，再前13日选择、后5日检验；39日前20/25/30/35日选择，随后5/5/5/4日检验。线性和固定档位均只用此前日期确定，同一任务比较。40日原冻结预测本身含训练内信息，故这一前推不能宣称独立样本外验证。', '',fits.to_csv(index=False),'',compare.to_csv(index=False),
      '所有日期已参与先前研究；这是一项条件限价诊断，不能把结果直接当作完整C策略盈利或部署建议。共同存活是为了买卖公平对照，未覆盖所有真实未成交单。']
    (DEST/'结果说明.md').write_text('\n'.join(report),encoding='utf-8')
    archive=ROOT/'result/上涨概率与挂档线性关系.zip';files=[p for p in DEST.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(DEST).parts)]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(archive) as z:
        for p in files:assert hashlib.sha256(z.read(p.relative_to(DEST).as_posix())).hexdigest()==sha(p)
    print(fits.to_string(index=False),flush=True);print(compare.to_string(index=False),flush=True)


if __name__=='__main__':main()
