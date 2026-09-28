"""Static evidence for fill-time shape and opening-minute cost attribution."""
from scripts.analyze_limit_fill_timing import OUT,ROOT,SOURCE,TASK,NAMES,STAGES,sha,save_json
import json,zipfile,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
    'axes.unicode_minus':False,'svg.fonttype':'path','font.size':10.5})
SCOPES=['first_minute','rest_59']
BLUE='#337ca0';GREEN='#258a7d';RED='#bb625a';ORANGE='#c89133';GRAY='#7e8894'
LABEL={'initial_limit':'初始限价','limit3':'3秒更新限价','market3':'3秒转市价','market10':'10秒兜底'}


def read(name):return pd.read_csv(OUT/(name+'.csv'))


def savefig(fig,name,title,footer,rect=(0,.085,1,.81)):
    fig.suptitle(title,fontsize=17,y=.988)
    fig.get_layout_engine().set(rect=rect)
    fig.text(.5,.025,footer,ha='center',fontsize=10,linespacing=1.55)
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=210,facecolor='white')
    plt.close(fig)


def shape_figure():
    hist=read('成交时间直方图');qq=read('正态QQ数据');km=read('含删失成交分布')
    stats=read('正态形状与分位统计');counts=read('群体与删失比例')
    fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained')
    for i,scope in enumerate(SCOPES):
        h=hist[hist.scope.eq(scope)&hist.side.eq('both')]
        q=qq[qq.scope.eq(scope)&qq.side.eq('both')]
        k=km[km.scope.eq(scope)&km.side.eq('both')&km.cohort.eq('activated_new_limit')]
        s=stats[stats.scope.eq(scope)&stats.side.eq('both')&stats.metric.eq('active_wait')].iloc[0]
        c=counts[counts.scope.eq(scope)&counts.side.eq('both')&counts.cohort.eq('activated_new_limit')].iloc[0]
        ax=axes[i,0];x=(h.left+h.right)/2
        ax.bar(x,h.observed_probability*100,width=.43,color=BLUE,label='已成交者：实际分布')
        ax.plot(x,h.normal_probability*100,color=ORANGE,lw=2,label='同均值/方差正态参照')
        ax.set(xlim=(-.5,8.5),ylim=(0,18),xlabel='新限价生效后等待（秒）',ylabel='成功成交者占比（每0.5秒箱，%）',
               title=f'{NAMES[scope]}：{int(c.limit_fills):,}笔成功 / {int(c.cohort_orders):,}笔生效')
        ax.text(.97,.96,f'中位数 {s.q50:g}秒  |  95%分位 {s.q95:g}秒\n超额峰度 {s.excess_kurtosis:+.2f}（正态为0）',
                transform=ax.transAxes,ha='right',va='top',fontsize=9)
        ax.legend(loc='upper right',bbox_to_anchor=(1,.79),fontsize=8.5);ax.grid(axis='y',alpha=.15)
        ax=axes[i,1]
        ax.plot(q.theoretical_z,s['mean']+s['sd']*q.theoretical_z,color=ORANGE,lw=2,label='正态参照直线')
        ax.scatter(q.theoretical_z,q.actual_seconds,s=12,color=BLUE,label='经验分位点',zorder=3)
        ax.axhline(0,color=GRAY,lw=.7,ls=':')
        ax.set(xlabel='标准正态分位数',ylabel='实际等待分位数（秒）',title='Q–Q图：两端明显偏离直线')
        ax.legend(fontsize=8.5);ax.grid(alpha=.15)
        ax=axes[i,2]
        ax.step(np.r_[0,k.time],np.r_[0,k.cdf*100],where='post',color=GREEN,lw=2,label='含删失的KM累计曲线')
        mass=k.event_weight.sum()+k.censor_weight.sum()
        ax.step(np.r_[0,k.time],np.r_[0,k.event_weight.cumsum()/mass*100],where='post',color=BLUE,lw=1,ls=':',label='实际已成交 / 全部生效单')
        cens=k[k.censor_weight.gt(1e-10)]
        ax.scatter(cens.time,cens.cdf*100,marker='|',s=120,color=RED,label='兜底：停止观察限价成交')
        ax.text(.04,.95,f'最终限价成交 {c.limit_fill_rate:.2%}\n最终兜底 {c.censor_rate:.2%}',transform=ax.transAxes,va='top',fontsize=11)
        ax.set(xlim=(0,8.3),ylim=(0,100),xlabel='新限价生效后等待（秒）',ylabel='累计限价成交比例（%）',title='所有生效单：兜底不是小尾部')
        ax.legend(fontsize=8.3,loc='center right');ax.grid(alpha=.15)
    savefig(fig,'限价成交时间分布','第3秒更新限价：已成交部分也不宜按正态分布裁尾\n观察3秒 / 预测1秒，初始19档、更新19档；两行对应不同发单时段',
        '仅纳入第3秒新限价实际生效的订单；到达前原单已成交者另表保留。左、中图只描述成功者，右图包含未成交者。\n兜底实际到达才停止观察，不把T+10提交市价当作限价成交；快照离散、期限截断，未知尾部不外推。18日，按日期权重后条件化。')


def retention_intervals():
    f=read('等待时点保留比例');daily=read('逐日等待保留比例')
    dates=sorted(read('逐日执行范围').trade_date.unique());n=len(dates)
    rng=np.random.default_rng(20260928);starts=rng.integers(0,n,(5000,int(np.ceil(n/5))))
    ids=((starts[...,None]+np.arange(5))%n).reshape(5000,-1)[:,:n]
    rows=[]
    for r in f.itertuples(index=False):
        a=daily[daily.scope.eq(r.scope)&daily.side.eq(r.side)&daily.clock.eq(r.clock)].set_index('trade_date').reindex(dates,fill_value=0)
        summed=a[['cohort_mass','filled_mass','eventual_fill_mass']].to_numpy()[ids].sum(axis=1)
        d=r._asdict()
        for name,denom in [('filled_fraction_of_branch',0),('retained_eventual_limit_fills',2)]:
            lo,hi=np.quantile(summed[:,1]/summed[:,denom],[.025,.975]);d[name+'_low']=lo;d[name+'_high']=hi
        rows.append(d)
    f=pd.DataFrame(rows);f.to_csv(OUT/'等待时点保留比例_日期块区间.csv',index=False,encoding='utf-8-sig')
    return f


def retention_figure(f):
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    for i,scope in enumerate(SCOPES):
        g=f[f.scope.eq(scope)&f.side.eq('both')]
        ax=axes[0,i]
        for metric,label,color in [('retained_eventual_limit_fills','已保住 / 原本最终会限价成交',GREEN),
                                   ('filled_fraction_of_branch','已限价成交 / 全部继续限价单',BLUE)]:
            ax.plot(g.clock,g[metric]*100,'o-',color=color,label=label,lw=2)
            ax.fill_between(g.clock,g[metric+'_low']*100,g[metric+'_high']*100,color=color,alpha=.10)
        for t in [6,9]:
            r=g[g.clock.eq(t)].iloc[0];y=r.retained_eventual_limit_fills*100
            ax.annotate(f'T+{t}：{y:.1f}%',(t,y),xytext=(0,12),textcoords='offset points',ha='center',color=GREEN)
        ax.axhline(95,color=GRAY,ls=':',lw=1)
        ax.set(title=NAMES[scope]+'：两个分母不能混用',xlabel='从最初发单T起的时刻（秒）',ylabel='比例（%）',xticks=range(4,11),ylim=(0,112))
        ax.legend(loc='lower right',fontsize=9);ax.grid(alpha=.15)
        ax=axes[1,i];g=g[g.clock.isin([6,9,10])]
        a=g.later_limit_fraction_among_waiting.to_numpy()*100;b=g.eventual_fallback_fraction_among_waiting.to_numpy()*100
        ax.bar(range(3),a,color=GREEN,label='原本后来仍能限价成交',width=.58)
        ax.bar(range(3),b,bottom=a,color=GRAY,label='原本最终兜底',width=.58)
        for j,(x,y) in enumerate(zip(a,b)):
            ax.text(j,x/2,f'{x:.1f}%',ha='center',va='center',color='white',fontsize=10)
            ax.text(j,x+y/2,f'{y:.1f}%',ha='center',va='center',color='white',fontsize=11)
        ax.set(title='该时刻仍在等待的订单，原本后来会怎样？',xticks=range(3),xticklabels=['T+6','T+9','T+10'],
               ylabel='仍未成交订单的事后构成（%）',ylim=(0,105))
        ax.legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2)
    savefig(fig,'等待时间与成交保留率','能否只处理尾部、保住大部分限价成交？\n第3秒未成交且决定继续限价的订单：保留原单、改单及在途成交均计入',
        '上图阴影：5日日期块、5,000次重采样的逐点95%区间；原策略不变时的等待结果，不是新截止规则回测。\n下图是事后分组，实盘当时不可直接识别。限价在兜底市价到达前仍可成交，因此T+10之后仍存在限价成交。',rect=(0,.1,1,.79))


def opening_figure():
    s=read('范围成本汇总');st=read('成交分支贡献汇总');ti=read('市价时间成本分解');bins=read('首分钟10秒段汇总')
    fig,axes=plt.subplots(2,2,figsize=(16,11),layout='constrained')
    ax=axes[0,0]
    for j,(scope,color) in enumerate(zip(SCOPES,[BLUE,ORANGE])):
        g=s[s.scope.eq(scope)].set_index('side').loc[['both','buy','sell']];x=np.arange(3)+(j-.5)*.33
        ax.bar(x,g.saving_bp,width=.3,color=color,label=NAMES[scope])
        ax.errorbar(x,g.saving_bp,yerr=np.vstack([g.saving_bp-g.low_bp,g.high_bp-g.saving_bp]),fmt='none',color=color,capsize=3)
        for xx,v in zip(x,g.saving_bp):ax.annotate(f'{v:+.3f}',(xx,v),xytext=(0,5 if v>=0 else -15),textcoords='offset points',ha='center',fontsize=9)
    ax.axhline(0,color=GRAY,lw=1);ax.set(xticks=range(3),xticklabels=['买卖各半','买入','卖出'],ylabel='相对起点立即市价节约（bp）',title='首分钟合并亏0.322bp；卖出点估计更差')
    ax.legend(fontsize=9);ax.grid(axis='y',alpha=.15);ax.margins(y=.2)
    ax=axes[0,1]
    for j,(scope,color) in enumerate(zip(SCOPES,[BLUE,ORANGE])):
        g=st[st.scope.eq(scope)&st.side.eq('both')].set_index('stage').loc[STAGES];x=np.arange(4)+(j-.5)*.33
        ax.bar(x,g.saving_contribution_bp,width=.3,color=color,label=NAMES[scope])
        for xx,v in zip(x,g.saving_contribution_bp):ax.annotate(f'{v:+.3f}',(xx,v),xytext=(0,5 if v>=0 else -15),textcoords='offset points',ha='center',fontsize=9)
    ax.axhline(0,color=GRAY,lw=1);ax.set(xticks=range(4),xticklabels=[LABEL[k].replace('更新','\n更新').replace('转市','\n转市') for k in STAGES],
        ylabel='全体订单分母下的节约贡献（bp）',title='损失不只在兜底：3秒转市价贡献更差')
    ax.legend(fontsize=9);ax.grid(axis='y',alpha=.15);ax.margins(y=.18)
    ax=axes[1,0];g=bins[bins.side.eq('both')]
    ax.bar(range(6),g.saving_bp,color=[GREEN if v>0 else RED for v in g.saving_bp],width=.65)
    ax.errorbar(range(6),g.saving_bp,yerr=np.vstack([g.saving_bp-g.low_bp,g.high_bp-g.saving_bp]),fmt='none',color='#555',capsize=3)
    for j,v in enumerate(g.saving_bp):ax.annotate(f'{v:+.3f}',(j,v),xytext=(0,7 if v>=0 else -16),textcoords='offset points',ha='center',fontsize=10)
    ax.axhline(0,color=GRAY,lw=1);ax.set(xticks=range(6),xticklabels=['00–09','10–19','20–29','30–39','40–49','50–59'],
        xlabel='09:30内的名义发单秒数；每段180个任务',ylabel='相对起点立即市价节约（bp）',title='最前10秒任务最差，并非整分钟均匀亏损')
    ax.grid(axis='y',alpha=.15);ax.margins(y=.18)
    ax=axes[1,1];g=ti[ti.scope.eq('first_minute')&ti.side.eq('both')].set_index(['stage','component'])
    components=[('wait_to3_cost_bp','起点市价成交→T+3',BLUE),('signal3_delay_cost_bp','T+3→市价指令到达',ORANGE),
                ('from3_to10_cost_bp','T+3→T+10继续等待',RED),('fallback_delay_cost_bp','T+10→兜底到达',GRAY)]
    positive=np.zeros(2);negative=np.zeros(2)
    for key,label,color in components:
        values=np.array([g.loc[(stage,key),'extra_contribution_bp'] if (stage,key) in g.index else 0 for stage in ['market3','market10']])
        bottom=np.where(values>=0,positive,negative)
        ax.barh([0,1],values,left=bottom,color=color,height=.5,label=label)
        for j,v in enumerate(values):
            if abs(v)>.06:ax.text(bottom[j]+v/2,j,f'{v:+.3f}',ha='center',va='center',fontsize=10,color='white')
        positive+=np.maximum(values,0);negative+=np.minimum(values,0)
    ax.axvline(0,color=GRAY,lw=1);ax.set(yticks=[0,1],yticklabels=['3秒转市价组','10秒兜底组'],ylim=(-.5,1.6),
        xlabel='额外成本贡献（bp，正数表示损失）',title='首分钟损失发生在哪段时间？')
    ax.legend(fontsize=8.5,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.22));ax.grid(axis='x',alpha=.15);ax.margins(x=.09)
    savefig(fig,'开盘首分钟差异归因','首分钟难点：价格移动快，等到信号再追单往往已经付出代价\nC3仅第3秒判断，观察3秒 / 预测1秒；初始与后续限价均被动19档',
        '18日，首分钟1,080任务；随后59分钟61,819任务。两侧完整配对，日期等权。成本参照均是含相同延迟的起点市价。\n左上/左下区间为95%日期块区间。右侧按最终成交类型事后归因，不能当作决策时已知分类或策略修改的因果收益。',rect=(0,.095,1,.805))


def table(df,columns):
    return '\n'.join('|'+ '|'.join(str(fn(r)) for fn in columns)+'|' for r in df.itertuples(index=False))


def report():
    counts=read('群体与删失比例');stats=read('正态形状与分位统计');lm=read('等待时点保留比例')
    s=read('范围成本汇总');st=read('成交分支贡献汇总');ti=read('市价时间成本分解');a=read('预测与行情诊断');bins=read('首分钟10秒段汇总')
    ct=counts[counts.side.eq('both')&counts.cohort.eq('activated_new_limit')]
    nt=stats[stats.side.eq('both')&stats.metric.eq('active_wait')]
    bt=counts[counts.side.eq('both')&counts.cohort.eq('limit_branch')]
    nr=table(nt,[lambda r:NAMES[r.scope],lambda r:r.n,lambda r:f'{r.q50:g}',lambda r:f'{r.q90:g}',lambda r:f'{r.q95:g}',
        lambda r:f'{r.skew:+.3f} [{r.skew_low:+.3f},{r.skew_high:+.3f}]',
        lambda r:f'{r.excess_kurtosis:+.3f} [{r.excess_kurtosis_low:+.3f},{r.excess_kurtosis_high:+.3f}]',lambda r:f'{r.normal_cdf_gap:.3f}'])
    cr=table(ct,[lambda r:NAMES[r.scope],lambda r:r.cohort_orders,lambda r:r.limit_fills,lambda r:r.censored,
        lambda r:f'{r.limit_fill_rate:.2%}',lambda r:f'{r.censor_rate:.2%}'])
    br=table(bt,[lambda r:NAMES[r.scope],lambda r:r.cohort_orders,lambda r:r.limit_fills,lambda r:r.censored,
        lambda r:f'{r.limit_fill_rate:.2%}',lambda r:f'{r.censor_rate:.2%}'])
    lr=table(lm[lm.side.eq('both')&lm.scope.isin(SCOPES)&lm.clock.isin([6,9,10])],[lambda r:NAMES[r.scope],lambda r:r.clock,
        lambda r:f'{r.retained_eventual_limit_fills:.2%}',lambda r:f'{r.filled_fraction_of_branch:.2%}',
        lambda r:f'{r.later_limit_fraction_among_waiting:.2%}',lambda r:f'{r.eventual_fallback_fraction_among_waiting:.2%}'])
    sr=table(s,[lambda r:NAMES[r.scope],lambda r:r.side,lambda r:r.orders//(2 if r.side=='both' else 1),
        lambda r:f'{r.cost_bp:.6f}',lambda r:f'{r.market_cost_bp:.6f}',lambda r:f'{r.saving_bp:+.6f}',lambda r:f'[{r.low_bp:+.6f},{r.high_bp:+.6f}]'])
    st=st[st.side.eq('both')&st.scope.isin(SCOPES)]
    strr=table(st,[lambda r:NAMES[r.scope],lambda r:LABEL[r.stage],lambda r:f'{r.rate:.2%}',lambda r:f'{r.conditional_saving_bp:+.6f}',lambda r:f'{r.saving_contribution_bp:+.6f}'])
    ar=table(a[a.group.eq('all_tasks')],[lambda r:NAMES[r.scope],lambda r:f'{r.accuracy:.2%}',lambda r:f'{r.flat_rate:.2%}',
        lambda r:f'{r.abs_move1_ticks:.2f}',lambda r:f'{r.abs_move3_observed_ticks:.2f}',lambda r:f'{r.spread0_ticks:.2f}'])
    binr=table(bins[bins.side.eq('both')],[lambda r:f'{r.bin_start:02d}–{r.bin_start+9:02d}',lambda r:f'{r.saving_bp:+.6f}',lambda r:f'[{r.low_bp:+.6f},{r.high_bp:+.6f}]'])
    text=f'''# 第3秒限价成交时间分布与开盘首分钟诊断

完成：2026-09-28。**不支持把兜底订单理解成正态分布的小尾巴，然后单靠时间剪掉。开盘首分钟表现差，主要与等待期间价格位移的代价有关；不是简单的预测准确率更低。**

## 本次口径

- 复用最近C3（只在第3秒判断）：观察3秒、预测1秒、冻结四折五因子模型；初始19档、第3秒需要限价也19档，10秒兜底。没有启用新增6/9秒规则，没有重训或改撮合。
- 18日首小时逐秒62,899共同任务、125,798买卖订单。首分钟1,080任务、2,160订单；随后59分钟61,819任务，两个时段互不重叠。
- 任务时段按名义发单秒0–59、60–3599划分；实际起点T为整秒后的首条合格快照。成交可越过时段边界。
- 均值和成本先日内后日期等权、买卖各半。分布先给各日期全部任务等权，再条件化到待研究群体；不把没有该类成交的日期伪造成0秒成交。
- 初始和更新限价的原单在撤换到达前有效；相同限价继续保留原单。两条严格未来快照延迟不变。T+10是提交兜底时间，实际成交通常约T+11，期间限价仍可成交。
- 主分布研究“第3秒新限价实际生效”的订单，从实际生效时刻计时。另保留“第3秒仍未成交且决定继续限价”的全部订单，包括原单保留和改单在途期间原单成交，从T+3计时。

## 1. 不能把成功者的直方图当成所有订单的成交分布

主群体：第3秒新限价实际生效。

|任务时段|生效订单|最终限价成交|最终兜底|限价成交率|兜底比例|
|---|---:|---:|---:|---:|---:|
{cr}

首分钟的新限价269笔成功、427笔兜底：**61.35%未等到限价成交，不是小尾部**。随后59分钟兜底比例更高。

只看最终成功者，从新限价生效后计时：

|任务时段|成功笔数|中位秒|90%分位秒|95%分位秒|偏度及95%区间|超额峰度及95%区间|经验CDF与正态最大距离|
|---|---:|---:|---:|---:|---|---|---:|
{nr}

正态的偏度与超额峰度均为0。这里Q–Q图两端明显偏离直线、超额峰度区间远离0；首分钟同均值/方差正态还会把约7.23%的概率放在不可能的负等待时间上。可观察的成功者分布是离散、被期限截断并经过成交筛选的分布，不适合直接用均值加标准差确定尾部。

**没有对这些密集重叠的逐秒任务强行套用独立样本正态检验p值。** 上述矩区间采用5日循环日期块、5,000次重采样，18个日期全部保留，5000次均有效。该结果描述的是已成交部分，不是对未知、无限等待成交时间分布的正式拒绝检验。

兜底市价到达时，尚未观察到限价成交，属于右删失；不能把市价时间记成限价成交时间。含删失曲线采用KM估计，只作可观察区间描述；少量订单存在不同截止时间，末端在险数很低，不外推最终一定成交或未知尾部为正态。[NIST删失说明](https://www.itl.nist.gov/div898/handbook/apr/section1/apr131.htm)、[KM方法](https://itl.nist.gov/div898/handbook/apr/section2/apr215.htm)、[正态分位图](https://www.itl.nist.gov/div898/handbook/eda/section3/normprpl.htm)。

![分布](限价成交时间分布.png)

## 2. 想保住大部分限价成交，不能过早按时间一刀切

更接近实际执行的群体：第3秒未成交且信号决定继续限价，包含原单保留/在途成交。

|任务时段|继续限价订单|最终限价成交|最终兜底|限价成交率|兜底比例|
|---|---:|---:|---:|---:|---:|
{br}

|任务时段|从发单T起经过秒数|已保住原本最终限价成交的比例|已成交占全部继续限价订单|当时仍等的订单中后来限价成交|当时仍等的订单中后来兜底|
|---|---:|---:|---:|---:|---:|
{lr}

首分钟T+6仅保住原本最终限价成交的60.63%，T+9为88.45%，T+10为94.49%。要在原路径上保住95%，需等到约T+10.5，已经接近原兜底到达。随后59分钟相应95%分位为T+11。

这里“保住95%”的分母是**原本最后会限价成交的订单**，不是全部订单。T+9首分钟余下未成交单中仍有8.94%后来成功限价成交，剩余91.06%最终兜底；这是事后组成，当时并不能直接识别哪笔属于哪类。

该表是原策略轨迹上的保留比例，**不是T+6/T+9改市价的策略收益，也没有扣除新指令到达前还能成交的情况**。若提交更早的市价，延迟期间原单仍能成交，真实成交保留率和成本须另做完整配对回放。

![等待保留率](等待时间与成交保留率.png)

## 3. 首分钟执行差在哪里

参照都是相同延迟的“起点立即市价”，正节约表示更好。

|任务时段|方向|任务数|C3成本bp|市价成本bp|节约bp|95%日期块区间|
|---|---|---:|---:|---:|---:|---|
{sr}

首分钟合并比市价贵0.321719bp；随后59分钟省0.063661bp。首分钟卖出点估计更差，但单侧区间均跨0，不把买卖差异宣称为确定规律。

|任务时段|最终成交分支|订单占比|该组单笔平均节约bp|占全部订单的节约贡献bp|
|---|---|---:|---:|---:|
{strr}

首分钟限价成交合计36.53%，比随后59分钟14.44%更高，兜底比例反而更低。问题在于追单和兜底的**单笔损失更大**：3秒市价组每笔多花2.618bp，随后59分钟为0.878bp；兜底组每笔多花3.352bp，随后59分钟为0.578bp。

首分钟两类市价损失精确拆账，分母均为全部订单：

- 3秒转市价总损失1.118863bp，其中起点市价实际成交→T+3的价格变化贡献0.983425bp（87.89%）；T+3→该市价实际到达贡献0.135438bp（12.11%）。**主要损失已在形成信号前发生。**
- 10秒兜底总损失0.695287bp：起点市价成交→T+3反而贡献−0.116753bp，随后T+3→T+10贡献+0.796680bp，兜底执行延迟再贡献+0.015360bp。
- 这些段落均使用同方向可立即执行的对手报价变化，逐笔相加精确等于相对起点市价额外成本；它是事后成本账目，不是随机干预或提前行动的净收益。

|任务时段|1秒三分类准确率|实际不变比例|未来1秒平均绝对价变tick|观察3秒平均绝对净位移tick|起点价差tick|
|---|---:|---:|---:|---:|---:|
{ar}

**首分钟预测准确率63.33%，高于随后59分钟55.17%，并非信号整体更不准。** 首分钟观察3秒价格净位移约14.39tick，对比7.22tick；未来1秒绝对变动7.20tick，对比3.98tick；盘口价差也更大。方向预测正确不等于等待后执行便宜，错误时移动幅度也未被准确率表达。两条快照延迟约1秒，与预测1秒相近，信号形成后可执行时点的价格同样重要；本次未做零延迟反事实，也未重新选择观察期。

开盘首分钟内继续按发单时间切分：

|09:30内发单秒数|相对立即市价节约bp|95%日期块区间|
|---|---:|---|
{binr}

每段180任务、360买卖订单。最前10秒任务比市价贵1.248689bp，是最突出的损失区域；30–39秒也较差，并非严格单调改善。分段区间未校正事后挑选，不据此直接宣布交易禁区。

![首分钟差异](开盘首分钟差异归因.png)

## 4. 对下一步的含义

你的目标“保留可成交订单，只处理代价高的剩余订单”值得继续，但**不需要正态分布作为前提，也不能只看时间是否处于尾部**。需要在每个仍未成交时点，用当时可见信息同时估计：继续挂单的剩余成交机会、未成交时继续等待的成本、现在转市价的成本。目标应是条件执行成本，不是仅预测涨跌类别。

首分钟应单独评估，尤其最前10秒：它既有更强的预测信号，也有更大的等待代价。先保留原19/19、3秒方案作基准。后续若研究尾部规则，可优先检查距限价距离、相对起点已错失价差、近段报价/OFI变化等是否区分高代价未成交订单，避免把原本有利的晚成交也统一取消。本次只诊断，尚未训练或选出这种新规则。

## 5. 交付与验收

- 3张PNG/SVG，完整分方向/逐日/条件群体CSV，125,798笔基准成交与报价时间成本分解Parquet，分析口径及输入SHA256。
- 6组首分钟/首小时×方向成本与既有C3汇总一致；原始行情、模型、原成交路径及下单记录哈希不变。
- 市价价格与原行情逐笔核对；市价各时段成本相加、成交分支占比/贡献相加均精确复现总账。首分钟与随后59分钟按日期配对差异另表保存。
- 概率曲线包含删失；KM同快照事件优先于删失，零等待成交保留。单位测试160项通过（包括5项新增统计边界测试）。
- 研究沿用已看过的18日、L1撮合代理；无排队/冲击/费用/部分成交。区间为已有日期描述，不是未见日期或经过选择校正的盈利保证。

复现：`python -m scripts.analyze_limit_fill_timing`，`python scripts/test_opening.py result/opening_execution/initial_offset_selection/limit_fill_timing`，`python -m scripts.deliver_limit_fill_timing`。
'''
    (OUT/'3秒限价成交分布与首分钟诊断.md').write_text(text,encoding='utf-8')


def verify():
    counts=read('群体与删失比例');lm=read('等待时点保留比例')
    assert (counts.cohort_orders==counts.limit_fills+counts.censored).all()
    np.testing.assert_allclose(counts.limit_fill_rate+counts.censor_rate,1,atol=1e-12)
    np.testing.assert_allclose(lm.later_limit_fraction_among_waiting+lm.eventual_fallback_fraction_among_waiting,1,atol=1e-12)
    for key,g in lm.groupby(['scope','side']):
        assert g.sort_values('clock').retained_eventual_limit_fills.diff().dropna().ge(-1e-12).all()
    stats=read('正态形状与分位统计')
    assert stats.bootstrap_dates.eq(18).all() and stats.bootstrap_valid_draws.eq(5000).all()
    hist=read('成交时间直方图')
    np.testing.assert_allclose(hist.groupby(['scope','side']).observed_probability.sum(),1,atol=1e-12)
    km=read('含删失成交分布')
    for key,g in km.groupby(['scope','side','cohort']):
        assert g.cdf.between(-1e-12,1+1e-12).all() and g.cdf.diff().dropna().ge(-1e-12).all()
    summary=read('范围成本汇总')
    for scope,g in summary.groupby('scope'):
        g=g.set_index('side')
        for metric in ['cost_bp','market_cost_bp','saving_bp']:
            np.testing.assert_allclose((g.loc['buy',metric]+g.loc['sell',metric])/2,g.loc['both',metric],atol=1e-12)
    tests=json.loads((OUT/'unit_test_verification.json').read_text(encoding='utf-8'))
    assert tests['passed'] and tests['tests_run']==160
    save_json(OUT/'交付验收.json',dict(cohort_rows=len(counts),retention_rows=len(lm),normal_diagnostic_rows=len(stats),
        cohort_counts_exact=True,proportions_sum_to_one=True,retention_monotone=True,bootstrap_all_18_dates=True,
        no_new_strategy_or_model=True,test_verification=tests))


def package():
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='文件清单.csv')
    manifest=pd.DataFrame([dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    manifest.to_csv(OUT/'文件清单.csv',index=False,encoding='utf-8-sig')
    archive=ROOT/'result/3秒限价成交分布与首分钟诊断.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in [*files,OUT/'文件清单.csv']:z.write(p,arcname=p.name)
    with zipfile.ZipFile(archive) as z:
        assert len(z.infolist())==len(files)+1
        for r in manifest.itertuples():
            assert z.getinfo(r.path).file_size==r.bytes
            with z.open(r.path) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r.sha256
    save_json(archive.with_suffix('.verification.json'),dict(archive=str(archive),entries=len(files)+1,sha256=sha(archive),all_verified=True))
    print(json.dumps(dict(archive=str(archive),entries=len(files)+1,bytes=archive.stat().st_size),ensure_ascii=False))


def main():
    verify();f=retention_intervals();shape_figure();retention_figure(f);opening_figure();report();package()


if __name__=='__main__':main()
