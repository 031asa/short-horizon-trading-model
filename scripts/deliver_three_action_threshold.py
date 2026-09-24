"""Static figures, readable tables and a standalone package for the T+3 experiment."""
from scripts.run_three_action_threshold import DEST, SOURCE, ROOT, THRESHOLDS, KEY, TASK, sha
import json, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
    'axes.unicode_minus':False,'svg.fonttype':'path','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
COLORS={'C19':'#865BB4','L0':'#DB902C','adaptive':'#147D92','M':'#55616D'}


def export(fig, name, footer):
    fig.text(.02,.018,footer,fontsize=9,color='#555')
    fig.savefig(DEST/(name+'.png'),dpi=210,bbox_inches='tight')
    fig.savefig(DEST/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def table(frame, columns, names=None):
    names=columns if names is None else names
    lines=['| '+' | '.join(names)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for row in frame[columns].itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    s=pd.read_csv(DEST/'全策略8组分方向汇总.csv')
    d=pd.read_csv(DEST/'全策略逐日结果.csv')
    bins=pd.read_csv(DEST/'信号强弱分箱_三动作成本.csv')
    audit=json.loads((DEST/'筛选与汇总验收.json').read_text(encoding='utf-8'))
    replay=json.loads((DEST/'回放验收.json').read_text(encoding='utf-8'))
    best=audit['selected_policy']; tau=audit['selected_threshold']
    diagnostic = 'T0.025' if best=='T0' else best
    diagnostic_tau=float(diagnostic[1:])
    names={'M':'立即市价','A':'A 初始LastPrice','B':'B 先观察3秒','C19':'原 C19','L0':'有利信号全挂LastPrice',best:f'阈值 {tau:g}'}
    names[diagnostic]=f'阈值 {diagnostic_tau:g}'
    foot='最新冻结模型回放全部40个源日期；实际有效37–39日。日期等权；已有日期探索，含训练内回算；区间未作阈值选择校正。'
    base=s[s.side.eq('both')&s.subset.eq('all')]
    primary=base[base.minutes.eq(60)&base.grid.eq('second')].set_index('policy')
    curves=[f'T{t:g}' for t in THRESHOLDS]
    # One full curve per grid and time range: no different optimal threshold for each group.
    fig,axes=plt.subplots(2,4,figsize=(17,8),squeeze=False)
    for r,grid in enumerate(['minute','second']):
        for c,minutes in enumerate([1,19,30,60]):
            ax=axes[r,c]; g=base[base.minutes.eq(minutes)&base.grid.eq(grid)].set_index('policy')
            y=g.loc[curves,'cost_bp'].values
            ax.plot(THRESHOLDS,y,'o-',color=COLORS['adaptive'],label='三动作阈值')
            for control in ['C19','L0','M']:
                ax.axhline(g.loc[control,'cost_bp'],ls='--',color=COLORS[control],lw=1.2,label=names[control])
            ax.scatter([tau],[g.loc[best,'cost_bp']],marker='*',s=190,color=COLORS['adaptive'],zorder=5)
            ax.set_title(f'前{minutes}分钟 · '+('每分钟任务' if grid=='minute' else '每秒任务')+f'\n{int(g.loc[best,"days"])}日 / {int(g.loc[best,"tasks"]):,}任务')
            ax.set_xlabel('信号阈值 τ'); ax.set_ylabel('平均执行成本（bp，越低越好）');ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8,loc='best')
    fig.suptitle(f'阈值扫描：同一阈值用于全部8组；星号为首小时逐秒成本最低的 τ={tau:g}',fontsize=15)
    fig.tight_layout(rect=[0,.055,1,.94]); export(fig,'01_阈值与执行成本',foot)
    # Conditional action comparisons use the exact same orders within each fixed score bin.
    b=bins[bins.minutes.eq(60)&bins.grid.eq('second')].copy()
    order=b[b.action.eq('market')].sort_values('left'); labels=order['bin'].tolist(); x=np.arange(len(labels))
    fig,(ax,counts)=plt.subplots(2,1,figsize=(15,8.2),sharex=True,gridspec_kw={'height_ratios':[3,1]})
    for action,label,color in [('lastprice','第3秒改挂LastPrice','#DB902C'),('passive','第3秒继续被动19档','#865BB4')]:
        g=b[b.action.eq(action)].set_index('bin').loc[labels]
        y=g.saving_vs_3s_market.to_numpy(); low=g.saving_low.to_numpy(); high=g.saving_high.to_numpy()
        ax.plot(x,y,'o-',color=color,label=label)
        ax.fill_between(x,low,high,color=color,alpha=.13)
    ax.axhline(0,color='#55616D',ls='--',label='第3秒转市价（对照）')
    ax.set_ylabel('相对第3秒转市价的节约（bp）');ax.grid(alpha=.2);ax.legend()
    ax.set_title('三种动作的配对回放：方向信号越强，挂得更深是否更划算？\n仅第3秒尚未成交；首小时每秒任务',fontsize=14)
    counts.bar(x,order.orders,color='#9CACB8');counts.set_ylabel('订单数');counts.set_xticks(x,labels,rotation=40,ha='right')
    counts.set_xlabel('有利方向概率差 F：买入 p跌−p涨；卖出 p涨−p跌。F 越大，预测方向越有利。')
    fig.tight_layout(rect=[0,.07,1,1]);export(fig,'02_信号强弱与三动作成本','区间：5日日期块重采样95%。分箱内按日期等权、保留实际买卖构成；不是预测跌幅，也不是逐单事后挑最便宜动作。')
    fig,axes=plt.subplots(1,3,figsize=(16,5.5))
    for ax,metric,ylabel in zip(axes,['elapsed_seconds','deadline_market','action_passive'],['平均完成时间（秒，含执行延迟）','10秒兜底市价成交比例','选择19档的订单比例（含早成订单为分母）']):
        ax.plot(THRESHOLDS,primary.loc[curves,metric],'o-',color=COLORS['adaptive'])
        for control in ['C19','L0']:
            if metric!='action_passive':ax.axhline(primary.loc[control,metric],ls='--',color=COLORS[control],label=names[control])
        ax.axvline(tau,ls=':',color=COLORS['adaptive']);ax.set_xlabel('信号阈值 τ');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
        if metric!='elapsed_seconds':ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=9)
    fig.suptitle('首小时每秒任务：成本之外，还要看等待和兜底',fontsize=15)
    fig.tight_layout(rect=[0,.07,1,.92]);export(fig,'03_完成时间与兜底比例',foot)
    # Per-date paired gains. Positive saves cost, negative worsens it.
    daily=d[d.policy.eq(diagnostic)&d.minutes.eq(60)&d.grid.eq('second')&d.subset.eq('all')&d.side.eq('both')].sort_values('trade_date')
    fig,axes=plt.subplots(2,1,figsize=(16,8),sharex=True)
    for ax,metric,title in zip(axes,['saving_vs_C19','saving_vs_L0'],['相对原 C19','相对有利信号全挂 LastPrice']):
        y=daily[metric].to_numpy(); xx=np.arange(len(y))
        ax.bar(xx,y,color=np.where(y>=0,'#238A7F','#C86465'));ax.axhline(0,color='#555',lw=.8)
        ax.axhline(y.mean(),color='#2A6685',ls='--',label=f'日期等权平均 {y.mean():+.4f} bp')
        ax.set_title(title);ax.set_ylabel('节约（bp）');ax.legend();ax.grid(axis='y',alpha=.15)
    axes[-1].set_xticks(np.arange(len(daily)),daily.trade_date.str[5:],rotation=55,ha='right')
    fig.suptitle(f'τ={diagnostic_tau:g} 的逐日改善是否稳定？首小时每秒任务',fontsize=15)
    fig.tight_layout(rect=[0,.065,1,.94]);export(fig,'04_逐日配对节约',foot)
    fig,axes=plt.subplots(1,2,figsize=(14,6.3),sharey=True)
    show=base[base.policy.eq(diagnostic)].sort_values(['grid','minutes'])
    labels=[f'前{r.minutes}分钟 / '+('每分钟' if r.grid=='minute' else '每秒') for r in show.itertuples()]
    for ax,metric,title in zip(axes,['saving_vs_C19','saving_vs_L0'],['相对原 C19','相对全挂 LastPrice 控制']):
        yy=np.arange(len(show));value=show[metric].to_numpy()
        ax.hlines(yy,show[metric+'_low'],show[metric+'_high'],color=COLORS['adaptive'],lw=2)
        ax.scatter(value,yy,color=COLORS['adaptive']);ax.axvline(0,color='#555',ls='--');ax.set_title(title)
        ax.set_yticks(yy,labels);ax.set_xlabel('节约（bp）与95%日期块区间');ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis();fig.suptitle(f'同一阈值 τ={diagnostic_tau:g} 的8组配对结果',fontsize=15)
    fig.tight_layout(rect=[0,.065,1,.94]);export(fig,'05_八组节约与区间',foot)
    report(s,d,audit,replay,primary,best,names,diagnostic)
    bundle()


def report(s,d,audit,replay,primary,best,names,diagnostic):
    tau=audit['selected_threshold']
    frame=s[s.policy.isin(['C19','L0',diagnostic])&s.side.eq('both')&s.subset.eq('all')]
    rows=[]
    for (minutes,grid),g in frame.groupby(['minutes','grid'],sort=True):
        g=g.set_index('policy');r=g.loc[diagnostic]
        rows.append(dict(范围=f'前{minutes}分钟',任务='每分钟' if grid=='minute' else '每秒',有效日期=int(r.days),任务数=int(r.tasks),
            **{'C19成本bp':g.loc['C19','cost_bp'],'有利信号全挂LastPrice成本bp':g.loc['L0','cost_bp'],
               f'阈值{diagnostic[1:]}成本bp':r.cost_bp,'较C19节约bp':r.saving_vs_C19,
               '较LastPrice节约bp':r.saving_vs_L0,
               '较C19节约95%区间bp':f'[{r.saving_vs_C19_low:+.4f}, {r.saving_vs_C19_high:+.4f}]'}))
    main=pd.DataFrame(rows);main.to_csv(DEST/'给leader的8组结果.csv',index=False,encoding='utf-8-sig')
    scope=s[s.minutes.eq(60)&s.grid.eq('second')&s.side.eq('both')&s.policy.isin(['C19','L0',diagnostic])].copy()
    scope['策略']=scope.policy.map(names);scope['范围']=scope.subset.map({'all':'全部订单','survivor3':'3秒仍未成交'})
    scope['兜底比例']=scope.deadline_market*100
    sel=primary.loc[best];loo=pd.read_csv(DEST/'留一日期阈值敏感性.csv');freq=pd.read_csv(DEST/'日期块阈值选择频率.csv')
    scan=primary.loc[[f'T{t:g}' for t in THRESHOLDS]+['L0']].reset_index()
    scan['timeout_pct']=scan.deadline_market*100
    forced=s[s.minutes.eq(60)&s.grid.eq('second')&s.side.eq('both')&s.subset.eq('survivor3')&s.policy.str.startswith('R_')].copy()
    forced['动作']=forced.policy.map({'R_market':'全部转市价','R_lastprice':'全部挂LastPrice','R_passive':'全部继续19档'})
    forced['timeout_pct']=forced.deadline_market*100
    small=primary.loc['T0.025']; c19=primary.loc['C19']; passive=primary.loc['R_passive']
    opening=s[s.policy.eq('T0.05')&s.minutes.eq(1)&s.grid.eq('second')&s.side.eq('both')&s.subset.eq('all')].iloc[0]
    scan[['policy','cost_bp','saving_vs_C19','elapsed_seconds','timeout_pct']].rename(columns={
        'policy':'规则','cost_bp':'平均成本bp','saving_vs_C19':'较C19节约bp','elapsed_seconds':'平均完成时间秒',
        'timeout_pct':'10秒兜底比例%'}).to_csv(DEST/'给leader的首小时阈值扫描.csv',index=False,encoding='utf-8-sig')
    content=f'''# 第3秒三动作阈值实验

## 已完成的四项

1. 同一笔订单独立回放第3秒转市价、改挂LastPrice、继续被动19档，按预设方向强度分箱比较。
2. 固定阈值0、0.025、0.05、0.075、0.10、0.15、0.20、0.30扫描，保存逐日成本、完成时间、兜底比例及日期敏感性。
3. 对照原C19及“预测不利则市价、其余全挂LastPrice”的控制规则；M/A/B也保存在完整表中。
4. 主结果含全部任务；另列第3秒仍未成交订单。早成交订单没有被删除。

## 固定规则

本轮C类策略初始按19档挂限价，买入LastPrice−3.8、卖出LastPrice＋3.8；M/A/B为历史参考，沿用各自初始规则。
T+3之前或同刻已成交的任务直接结束。其余订单：预测方向不利于本次执行→转市价；预测不变→LastPrice；预测方向有利且F≥τ→被动19档，否则LastPrice。
买入F=p跌−p涨，卖出F=p涨−p跌。F是概率差，不是预期跌幅，也不是经过独立校准的成交概率。
新限价锚定T+3当时LastPrice，不使用未来价格。新单/改单双快照延迟；原单在途仍可成交；第10秒提交市价兜底，不重新计时。
模型只预测观察结束后3秒方向，不能把概率直接解释为余下7秒的跌幅或穿价概率。
限价触发、结算及缺口规则沿用原撮合：限价单由对手价触达或有成交量的LastPrice严格穿价触发，均按限价结算；市价单按到达时对手价结算。未模拟排队、部分成交、费用与冲击。

## 样本及复核

40个源日期统一用最新保存W3第四折模型，未重训。35个模型训练日包含在本轮回放中，仅3个日期晚于训练截止，但此前均已研究。
07-20无前一小时行情；07-27从09:43:16.5开始；07-28首条09:30:01。各范围实际有效37–39日，不能称40个完整交易日。
原共同任务{replay['source_tasks']:,}，三动作交集{replay['common_tasks']:,}，额外剔除{replay['extra_removed_tasks']:,}。
独立核对最早成交{replay['verified_fills']:,}笔；旧C19逐笔精确复现{replay['exact_C_replay']:,}笔；提前成交三分支一致检查{replay['early_fill_checks']:,}次。
主结果每个任务买卖各1笔，先逐日期、方向求均值，再日期等权与买卖各半。
3秒未成交子集为{audit['survivor_orders']:,}笔方向订单，提前成交{audit['early_orders']:,}笔；子集仍按日期内买卖各半汇总，缺少任一方向的日期单列排除数量，不能把条件结果直接当成全策略结果。
强度分箱表另按各日该箱实际存活买卖构成平均，不声称每箱买卖各半；分箱内三动作严格同单配对。

## 统一阈值的8组结果

在计算前固定以“首小时每秒任务、全部订单、日期等权成本最低”选择一个阈值，再原样应用8组。
网格内最低阈值为τ={tau:g}。这是本批历史最低点，并不保证优于原C19或LastPrice控制，也不证明下批最优。
本批没有预测不变，τ=0逐单等同原C19。下表的“阈值策略”专门展示最小正阈值{diagnostic[1:]}，用来评估引入LastPrice中间档后的变化，并不是推荐参数。
成本及节约单位均为bp，成本越低越好；节约为对照成本减策略成本。

{table(main,list(main.columns))}

## 首小时完整阈值扫描

{table(scan,['policy','cost_bp','saving_vs_C19','elapsed_seconds','timeout_pct'],['规则','成本bp','较C19节约bp','完成时间秒','兜底%'])}

首小时所有正阈值均未比C19省钱。最小正阈值0.025增加成本{-small.saving_vs_C19:.6f}bp，95%配对节约区间[{small.saving_vs_C19_low:+.6f},{small.saving_vs_C19_high:+.6f}]bp；{int(small.days)}日中{int(small.days_better_C19)}日改善、{int(small.days_worse_C19)}日恶化，但平均完成时间缩短约{c19.elapsed_seconds-small.elapsed_seconds:.2f}秒。
首分钟逐秒任务存在局部例外：阈值0.05成本{opening.cost_bp:.6f}bp，较C19节约{opening.saving_vs_C19:.6f}bp，区间[{opening.saving_vs_C19_low:+.6f},{opening.saving_vs_C19_high:+.6f}]bp跨0；这是扫描后的局部结果，尚无清楚的稳定改善证据，不据此单独改开盘规则。
图02显示，在首小时有利方向各分箱内，LastPrice的平均成本均高于19档；19档相对第3秒市价的节约没有随概率差单调增强。方向置信度尚不能直接当作挂单深度依据。

## 三动作固定回放：仅3秒未成交子集

下表不使用信号选择动作；同一批尚未成交订单全部尝试同一个动作，成本仍以T时刻LastPrice为基准。

{table(forced,['动作','orders','cost_bp','elapsed_seconds','timeout_pct'],['动作','方向订单数','成本bp','完成时间秒','兜底%'])}

将早成交订单放回，固定“第3秒不看信号、全部按当时LastPrice被动19档更新”成本为{passive.cost_bp:.6f}bp，较原C19节约{passive.saving_vs_C19:.6f}bp，区间[{passive.saving_vs_C19_low:.6f},{passive.saving_vs_C19_high:.6f}]bp；但平均完成时间{passive.elapsed_seconds:.6f}秒、10秒兜底比例{passive.deadline_market*100:.4f}%，而C19为{c19.elapsed_seconds:.6f}秒、{c19.deadline_market*100:.4f}%。这是固定动作消融的发现，没有把它纳入预登记阈值候选，也不是说方向模型带来了该项改善。
因此本轮未支持新增LastPrice中间档来降低成本。更值得进一步检验的是执行成本与等待时间的取舍，以及“何时需要市价”的判断；不能只用涨跌预测概率替代动作收益判断。

## 全部订单与存活子集

{table(scope,['范围','策略','days','orders','cost_bp','elapsed_seconds','兜底比例','saving_vs_C19','saving_vs_L0'],['范围','策略','日期','方向订单数','成本bp','完成时间秒','兜底%','较C19节约bp','较LastPrice节约bp'])}

## 阈值稳定性

首小时逐秒，τ={tau:g}较C19节约{sel.saving_vs_C19:+.6f}bp，95%日期块区间[{sel.saving_vs_C19_low:+.6f},{sel.saving_vs_C19_high:+.6f}]；
较LastPrice控制节约{sel.saving_vs_L0:+.6f}bp，区间[{sel.saving_vs_L0_low:+.6f},{sel.saving_vs_L0_high:+.6f}]。
逐日较C19改善/相同/恶化={int(sel.days_better_C19)}/{int(sel.days_equal_C19)}/{int(sel.days_worse_C19)}。
每次去掉一个日期后重新选阈值，选择次数：{audit['LOO_choices']}；留出日平均较C19节约{loo.saving_vs_C19.mean():+.6f}bp、较LastPrice节约{loo.saving_vs_L0.mean():+.6f}bp。
这只是阈值对日期的敏感性诊断，冻结模型和前期研究已接触这些日期，不能当成独立验证。

日期块重复选阈值的频率：

{table(freq,list(freq.columns))}

## 文件与限制

01为8组阈值曲线；02为3动作强度分箱；03为时间与兜底；04为逐日增益；05为8组配对区间。均有PNG与SVG。
所有结果来自同一份汇总，完整买入、卖出、合并及存活子集见全策略8组分方向汇总.csv；逐单动作、价格、时刻、成本与概率分别保存在Parquet中。
区间使用有序有效日期的循环5日块、5000次重采样、固定种子20260923。未校正19档选择、阈值扫描与此前反复研究。
不从每笔事后最低成本动作构造可执行策略；所报告阈值决策只依赖当时模型信号。
日内任务重叠，订单数多并不等于独立样本很多；本轮结果属于描述性探索，不能据此认定实盘收益。
'''
    (DEST/'三动作阈值实验说明.md').write_text(content,encoding='utf-8')


def bundle():
    archive=ROOT/'result/三动作阈值实验_40日覆盖.zip'
    files=sorted(p for p in DEST.iterdir() if p.is_file())
    manifest={p.name:sha(p) for p in files}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,'三动作阈值实验/'+p.name)
        z.writestr('三动作阈值实验/文件校验.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    import hashlib
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read('三动作阈值实验/'+name)).hexdigest()==digest
    print('Packaged',len(files),'files',archive,flush=True)


if __name__=='__main__':main()
