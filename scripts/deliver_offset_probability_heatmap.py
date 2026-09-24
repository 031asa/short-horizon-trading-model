"""Conditional probability x offset diagnostic; no new action selection or replay."""
from scripts.run_signal_offset_sweep import *
from scripts.deliver_signal_offset_sweep import package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'svg.fonttype':'path','font.size':11})


def main():
    output=DEST/'probability_heatmap';output.mkdir(exist_ok=True)
    sources=[];summaries=[];daily_all=[];audit=[]
    for scope in SOURCES:
        folder=DEST/scope;previous=OUT/'rebased_three_action_threshold'/scope
        paths=[previous/'任务与信号.parquet',previous/'阈值与基准_逐单结果.parquet',*sorted((folder/'逐日逐单').glob('*.parquet'))]
        sources.extend(paths)
        info=pd.read_parquet(paths[0])
        assert info[['p_down','p_flat','p_up']].notna().all().all()
        np.testing.assert_allclose(info[['p_down','p_flat','p_up']].sum(axis=1),1,atol=1e-8)
        eligible=info[info.survivor3&(info.direction*info.signal<=0)].copy()
        eligible['probability']=np.where(eligible.direction.eq(1),eligible.p_down,eligible.p_up)
        assert eligible.probability.between(0,1).all()
        eligible['bin']=np.minimum(np.floor(eligible.probability*10).astype(int),9)
        branches=pd.read_parquet(paths[1],filters=[('policy','in',['R_market','R_lastprice','R_passive'])])
        market=branches[branches.policy.eq('R_market')][KEY+['cost_bp']].rename(columns={'cost_bp':'market3_cost_bp'})
        eligible=eligible.merge(market,on=KEY,validate='one_to_one')
        dayparts=[];joined_orders=0
        for path in paths[2:]:
            frame=pd.read_parquet(path)
            g=frame.merge(eligible[KEY+['bin','probability','market3_cost_bp']],on=KEY,how='inner',validate='many_to_one')
            assert g.groupby(KEY).size().eq(len(OFFSETS)).all()
            for k,policy in [(0,'R_lastprice'),(19,'R_passive')]:
                a=g[g.k.eq(k)].set_index(KEY);b=branches[branches.policy.eq(policy)].set_index(KEY).loc[a.index]
                np.testing.assert_allclose(a[['cost_bp','fill_seconds','fill_ticks']],b[['cost_bp','fill_seconds','fill_ticks']],rtol=0,atol=1e-10)
            g['saving_bp']=g.market3_cost_bp-g.cost_bp
            g['deadline_rate']=g.method.eq(2).astype(float)
            day=g.groupby(['trade_date','direction','bin','k']).agg(cost_bp=('cost_bp','mean'),market3_cost_bp=('market3_cost_bp','mean'),saving_bp=('saving_bp','mean'),elapsed_seconds=('elapsed_seconds','mean'),deadline_rate=('deadline_rate','mean'),orders=('cost_bp','size')).reset_index()
            dayparts.append(day);joined_orders+=len(g)//len(OFFSETS)
        assert joined_orders==len(eligible)
        daily=pd.concat(dayparts,ignore_index=True);daily['scope']=scope;daily_all.append(daily)
        summary=daily.groupby(['scope','direction','bin','k']).agg(cost_bp=('cost_bp','mean'),market3_cost_bp=('market3_cost_bp','mean'),saving_bp=('saving_bp','mean'),elapsed_seconds=('elapsed_seconds','mean'),deadline_rate=('deadline_rate','mean'),orders=('orders','sum'),days=('trade_date','nunique')).reset_index()
        summary['probability_left']=summary.bin/10;summary['probability_right']=(summary.bin+1)/10
        for _,g in summary.groupby(['direction','bin']):
            assert len(g)==len(OFFSETS) and g.orders.nunique()==1 and g.days.nunique()==1
            np.testing.assert_allclose(g.market3_cost_bp-g.cost_bp,g.saving_bp,atol=1e-12)
        summaries.append(summary)
        audit.append(dict(scope=scope,all_orders=len(info),surviving_at3=int(info.survivor3.sum()),limit_branch_orders=len(eligible),buy_orders=int(eligible.direction.eq(1).sum()),sell_orders=int(eligible.direction.eq(-1).sum()),all_offsets_common=True,k0_k19_exact_action_reproduction=True))
    summary=pd.concat(summaries,ignore_index=True);daily=pd.concat(daily_all,ignore_index=True)
    summary.to_csv(output/'概率档位热力图汇总.csv',index=False,encoding='utf-8-sig');daily.to_csv(output/'概率档位逐日.csv',index=False,encoding='utf-8-sig')
    vmax=float(summary.saving_bp.abs().max());norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
    for scope in SOURCES:
        fig,axes=plt.subplots(1,2,figsize=(17,12),layout='constrained')
        for ax,side,label,prob in zip(axes,[1,-1],['买入','卖出'],['p_down（下跌概率）','p_up（上涨概率）']):
            g=summary[summary.scope.eq(scope)&summary.direction.eq(side)]
            values=g.pivot(index='k',columns='bin',values='saving_bp').reindex(index=OFFSETS,columns=range(10))
            cmap=plt.get_cmap('RdYlGn').copy();cmap.set_bad('#eeeeee')
            im=ax.imshow(np.ma.masked_invalid(values.to_numpy()),aspect='auto',cmap=cmap,norm=norm,interpolation='nearest')
            support=g[g.k.eq(0)].set_index('bin')
            labels=[]
            for b in range(10):
                n=int(support.loc[b,'orders']) if b in support.index else 0;days=int(support.loc[b,'days']) if b in support.index else 0
                flag='†' if n>0 and (n<100 or days<5) else ''
                labels.append(f'{b/10:.1f}–{(b+1)/10:.1f}\n{n:,}笔{flag}\n{days}日')
            ax.set_xticks(range(10),labels,fontsize=9);ax.set_yticks(range(len(OFFSETS)),OFFSETS)
            ax.set_xlabel(f'{prob} · 区间内订单数 / 有效日期数')
            ax.set_ylabel('第3秒限价被动偏移（档；离散档位等距展示）')
            ax.set_title(f'{label}｜{int(support.orders.sum()):,} 笔限价分支订单',fontsize=14)
            row=list(OFFSETS).index(19);ax.axhline(row-.5,color='#333333',lw=.8,ls='--');ax.axhline(row+.5,color='#333333',lw=.8,ls='--')
        cbar=fig.colorbar(im,ax=axes,shrink=.8,pad=.02);cbar.set_label('相对第3秒转市价的节约（bp）：绿=更省，红=更贵')
        label='18日' if scope=='18d' else '40日覆盖（首小时39个有效日，含训练内回算）'
        fig.suptitle(f'{label}｜概率 × 第3秒挂档\n初始固定19档 · 首小时逐秒任务 · 第3秒未成交且原信号走限价 · 逐日等权',fontsize=16)
        fig.supxlabel('每格均为同订单配对节约，包含改单延迟内旧单成交及10秒兜底；零线不是任务开始立即市价。\n灰色=无样本；†=不足100笔或5日，仅提示稀疏；最后区间含1.0，其他左闭右开。四个面板统一色标，虚线标19档。\n模型概率未经本轮校准，不等同于跌幅/涨幅；本图为条件诊断，不按热力图直接选动态规则。',fontsize=10)
        for ext in ['png','svg']:fig.savefig(output/f'{scope}_买卖概率档位热力图.{ext}',dpi=180)
        plt.close(fig)
    save_json(output/'验收.json',dict(cohorts=audit,bin_width=.1,bin_closure='left closed, right open; final includes1',same_color_scale=[-vmax,vmax],aggregation='equal nonempty dates within each side/bin',zero_reference='T+3 market with initial19 retained until replacement arrives',model_and_execution_unchanged=True))
    (output/'说明.md').write_text('买入使用p_down，卖出使用p_up。仅第3秒仍未成交且原信号决定限价的订单，排除原本市价分支，避免把与挂档无关的零差异混入。\n每格先计算同单第3秒市价成本减限价k档成本，再逐日等权；同侧同概率箱全部档位严格同订单，空日期不补零。初始19档、延迟和兜底沿用原口径。\n热力图不按事后方向正确与否分组，未选择动态档位。0/19档已逐单对照既有三动作结果；CSV保留全部格子的成本、样本数、有效日期、时间和兜底比例。†仅提示样本稀疏，不删除数据。\n',encoding='utf-8')
    package();print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
