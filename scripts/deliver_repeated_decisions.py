"""Paired attribution, static figures, report and archive for repeated decisions."""
from scripts.run_repeated_decisions import OUT,ROOT,SOURCE,KEY,TASK,FEATURES,PCOLS,sha,save_json
from utils.opening_repeated import NAMES
from utils.opening_two_stage import date_block_interval
import json,zipfile,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'svg.fonttype':'path','font.size':11})
CODES=['M','C3','C36','C369']
SHORT={'M':'立即市价','C3':'3秒','C36':'3、6秒','C369':'3、6、9秒'}
COLOR={'M':'#888888','C3':'#3c71a0','C36':'#d59238','C369':'#269586'}
SCOPE={60:'开盘后首小时',1:'开盘首分钟'}
ORIGIN={'initial_limit':'原初始限价成交','limit3':'原3秒限价成交','market3':'原3秒市价成交','market10':'原10秒兜底成交'}


def figsave(fig,name):
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=210,facecolor='white')
    plt.close(fig)


def diagnose(orders,summary):
    base=orders[orders.strategy.eq('C3')][KEY+['stage','cost_bp','fill_seconds']].rename(columns={'stage':'original_stage','cost_bp':'original_cost','fill_seconds':'original_fill_seconds'})
    paired=orders[orders.strategy.isin(['C36','C369'])].merge(base,on=KEY,validate='many_to_one')
    paired['gain_vs_C3_bp']=paired.original_cost-paired.cost_bp
    paired.to_parquet(OUT/'与原方案逐单配对归因.parquet',index=False)
    rows=[];trans=[]
    for minutes in [1,60]:
        for side,name in [(0,'both'),(1,'buy'),(-1,'sell')]:
            scope=paired[paired.nominal_second.lt(minutes*60)]
            if side:scope=scope[scope.direction.eq(side)]
            for (date,strategy),g in scope.groupby(['trade_date','strategy']):
                for stage in ORIGIN:
                    mask=g.original_stage.eq(stage)
                    rows.append(dict(trade_date=date,strategy=strategy,minutes=minutes,side=name,original_stage=stage,
                        original_rate=float(mask.mean()),gain_contribution_bp=float(g.gain_vs_C3_bp.where(mask,0).mean())))
                    for new_stage in ['initial_limit','limit3','limit6','limit9','market3','market6','market9','market10']:
                        selected=mask & g.stage.eq(new_stage)
                        trans.append(dict(trade_date=date,strategy=strategy,minutes=minutes,side=name,
                            original_stage=stage,new_stage=new_stage,rate=float(selected.mean()),
                            gain_contribution_bp=float(g.gain_vs_C3_bp.where(selected,0).mean())))
    daily=pd.DataFrame(rows);transition=pd.DataFrame(trans)
    daily.to_csv(OUT/'原方案成交群体_逐日配对贡献.csv',index=False,encoding='utf-8-sig')
    keys=['minutes','side','strategy','original_stage']
    grouped=daily.groupby(keys,as_index=False)[['original_rate','gain_contribution_bp']].mean()
    grouped['conditional_gain_bp']=grouped.gain_contribution_bp/grouped.original_rate.replace(0,np.nan)
    grouped.to_csv(OUT/'原方案成交群体_配对贡献.csv',index=False,encoding='utf-8-sig')
    transition.groupby(keys+['new_stage'],as_index=False)[['rate','gain_contribution_bp']].mean().to_csv(OUT/'成交方式转移表.csv',index=False,encoding='utf-8-sig')
    for (minutes,side,strategy),g in grouped.groupby(['minutes','side','strategy']):
        ref=summary[summary.minutes.eq(minutes)&summary.side.eq(side)&summary.strategy.eq(strategy)].iloc[0]
        np.testing.assert_allclose(g.original_rate.sum(),1,atol=1e-12)
        np.testing.assert_allclose(g.gain_contribution_bp.sum(),ref.saving_vs_C3_bp,atol=1e-12)
    signals=pd.read_parquet(OUT/'三时点信号与特征.parquet')
    commands=pd.read_parquet(OUT/'全部下单记录.parquet')
    missing=[]
    for code in ['C3','C36','C369']:
        z=orders[orders.strategy.eq(code)]
        for second in [3,6,9]:
            n=int(z.missing_decisions.fillna('').str.split(',').map(lambda a:str(second) in a).sum())
            missing.append(dict(strategy=code,decision_second=second,missing_reached_orders=n))
    pd.DataFrame(missing).to_csv(OUT/'实际到达判断点的信号缺失.csv',index=False,encoding='utf-8-sig')
    diag=[]
    for (code,second),g in commands.groupby(['strategy','decision_second']):
        diag.append(dict(strategy=code,decision_second=second,submissions=len(g),
            realized_arrivals=int(g.realized_arrival.sum()),cancelled_by_earlier_fill=int((~g.realized_arrival).sum()),
            after_T10=int(g.arrival_seconds.gt(g.task_seconds+10+1e-10).sum()),
            mean_scheduled_delay=float((g.arrival_seconds-g.submitted_seconds).mean())))
    pd.DataFrame(diag).to_csv(OUT/'指令延迟诊断.csv',index=False,encoding='utf-8-sig')
    # Explicitly audit all recorded boundaries, including the initially cached days.
    np.testing.assert_array_equal(signals.window_start,signals.task_seconds+(signals.decision_second-3))
    np.testing.assert_array_equal(signals.window_start+3,signals.window_end)
    save_json(OUT/'时间边界验收.json',dict(signal_rows=len(signals),start_exact=True,end_exact=True,
        repair='Preserve origin as t+(s-3), not (t+s)-3; all stored boundaries checked'))
    return grouped,pd.DataFrame(diag),pd.DataFrame(missing)


def cost_figure(summary,pairs):
    fig,axes=plt.subplots(2,2,figsize=(15,10.5),layout='constrained')
    for row,minutes in enumerate([60,1]):
        g=summary[summary.minutes.eq(minutes)&summary.side.eq('both')].set_index('strategy').loc[CODES]
        ax=axes[row,0]
        ax.bar(np.arange(4),g.cost_bp,color=[COLOR[c] for c in CODES],width=.62)
        for i,r in enumerate(g.itertuples()):
            ax.text(i,r.cost_bp,f'{r.cost_bp:.4f}',ha='center',va='bottom',fontsize=11)
        ax.axhline(g.loc['M','cost_bp'],color='#888',ls='--',lw=1)
        ax.set(xticks=np.arange(4),xticklabels=[SHORT[c] for c in CODES],ylabel='平均执行成本（bp，越低越好）',
               title=f'{SCOPE[minutes]}：{int(g.tasks.iloc[0]):,}个共同任务')
        ax.set_ylim(0,g.cost_bp.max()*1.19);ax.grid(axis='y',alpha=.15)
        ax=axes[row,1]
        q=pairs[pairs.minutes.eq(minutes)&pairs.side.eq('both')].reset_index(drop=True)
        for i,r in q.iterrows():
            color=COLOR[r.candidate]
            ax.plot([r.low_bp,r.high_bp],[i,i],color=color,lw=3)
            ax.scatter(r.saving_bp,i,color=color,s=65,zorder=3)
            ax.annotate(f'{r.saving_bp:+.4f} [{r.low_bp:+.4f}, {r.high_bp:+.4f}]',
                        (r.saving_bp,i),xytext=(0,15),textcoords='offset points',ha='center',fontsize=9)
        ax.axvline(0,color='#888',lw=1)
        ax.set(yticks=np.arange(len(q)),yticklabels=[f'{SHORT[r.candidate]}\n相对{SHORT[r.reference]}' for r in q.itertuples()],
               xlabel='配对节约（bp，正数更好）；95%日期块区间',title='新增判断点带来了多少改善？',ylim=(-.6,2.8))
        ax.grid(axis='x',alpha=.15);ax.margins(x=.30)
    fig.suptitle('初始19档：增加第6、9秒判断，能否节约成本？\n每次回看最近3秒、预测未来1秒；限价仍为当时LastPrice被动19档',fontsize=17,y=.99)
    fig.get_layout_engine().set(rect=(0,.085,1,.815))
    fig.text(.5,.023,'18日、日期等权、买卖各半；保留提前成交订单。两条后续快照延迟、10秒兜底，全部方案严格同任务配对。\n第9秒订单按实际到达时刻生效；区间为5日日期块、5,000次重采样，属于已有日期探索，未校正此前筛选。',ha='center',fontsize=10)
    figsave(fig,'重复判断_成本与配对节约')


def attribution_figure(summary,contribution,paired):
    fig,axes=plt.subplots(2,2,figsize=(15,10.5),layout='constrained')
    for row,minutes in enumerate([60,1]):
        ax=axes[row,0];codes=['C3','C36','C369'];bottom=np.zeros(3)
        dc=contribution[contribution.minutes.eq(minutes)&contribution.side.eq('both')]
        kinds=[('限价成交',['initial_limit','limit3','limit6','limit9'],'#3c71a0'),
               ('3秒市价',['market3'],'#d59238'),('6秒市价',['market6'],'#a46fa1'),
               ('9秒市价',['market9'],'#ce929d'),('10秒兜底',['market10'],'#89939d')]
        for label,stages,color in kinds:
            vals=np.array([dc[dc.strategy.eq(c)&dc.stage.isin(stages)].rate.sum()*100 for c in codes])
            ax.bar(range(3),vals,bottom=bottom,color=color,label=label,width=.6)
            for i,v in enumerate(vals):
                if v>3:ax.text(i,bottom[i]+v/2,f'{v:.1f}%',ha='center',va='center',fontsize=10,color='white')
            bottom+=vals
        np.testing.assert_allclose(bottom,100,atol=1e-10)
        ax.set(xticks=range(3),xticklabels=[SHORT[c] for c in codes],ylim=(0,109),ylabel='最终成交构成（%）',
               title=f'{SCOPE[minutes]}：兜底减少，限价成交也减少')
        ax.legend(fontsize=9,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.09))
        ax=axes[row,1]
        q=paired[paired.minutes.eq(minutes)&paired.side.eq('both')&paired.strategy.eq('C369')].set_index('original_stage').loc[list(ORIGIN)]
        vals=q.gain_contribution_bp.to_list()+[q.gain_contribution_bp.sum()]
        ax.bar(np.arange(5),vals,color=['#269586' if v>=0 else '#ba605c' for v in vals],width=.62)
        for i,v in enumerate(vals):ax.annotate(f'{v:+.4f}',(i,v),xytext=(0,5 if v>=0 else -15),textcoords='offset points',ha='center',fontsize=10)
        ax.axhline(0,color='#888',lw=1)
        ax.set(xticks=np.arange(5),xticklabels=['原初始\n限价成交','原3秒\n限价成交','原3秒\n市价成交','原10秒\n兜底成交','合计'],
               ylabel='新增6、9秒操作的节约贡献（bp）',title=f'{SCOPE[minutes]}：按原方案成交群体做同单配对')
        ax.margins(y=.3);ax.grid(axis='y',alpha=.15)
    fig.suptitle('兜底少了，为什么总成本反而更高？\n同一批订单：挽回原兜底成本，也失去部分原限价成交机会',fontsize=17,y=.99)
    fig.get_layout_engine().set(rect=(0,.10,1,.80))
    fig.text(.5,.023,'右图：原3秒方案成本−新3/6/9秒方案成本，按原方案的最终成交方式分组；所有贡献以全体任务为分母。\n这些是事后配对归因，实盘决策时不知道订单原本会在哪一类成交；不能直接拿分组结果充当可用信号。',ha='center',fontsize=10)
    figsave(fig,'重复判断_成交构成与损失来源')


def report(summary,pairs,attribution,diag,missing):
    audit=json.loads((OUT/'撮合验收.json').read_text(encoding='utf-8'))
    tests=json.loads((OUT/'unit_test_verification.json').read_text(encoding='utf-8'))
    def table(minutes):
        g=summary[summary.minutes.eq(minutes)&summary.side.eq('both')].set_index('strategy').loc[CODES]
        return '\n'.join(f'|{NAMES[c]}|{r.cost_bp:.6f}|{r.saving_vs_market_bp:+.6f}|{r.saving_vs_C3_bp:+.6f}|{r.elapsed_seconds:.3f}|{r.fallback_rate:.2%}|' for c,r in g.iterrows())
    header='|策略|成本bp|相对立即市价节约bp|相对只在3秒判断节约bp|平均完成秒|10秒兜底成交比例|\n|---|---:|---:|---:|---:|---:|'
    pairtable='\n'.join(f'|{SCOPE[r.minutes]}|{SHORT[r.candidate]} 相对 {SHORT[r.reference]}|{r.saving_bp:+.6f}|[{r.low_bp:+.6f}, {r.high_bp:+.6f}]|{r.positive_days}/{r.negative_days}/{r.zero_days}|' for r in pairs[pairs.side.eq('both')].itertuples())
    attrtable='\n'.join(f'|{SCOPE[r.minutes]}|{ORIGIN[r.original_stage]}|{r.original_rate:.2%}|{r.gain_contribution_bp:+.6f}|' for r in attribution[attribution.side.eq('both')&attribution.strategy.eq('C369')].itertuples())
    sidetable='\n'.join(f'|{SCOPE[r.minutes]}|{NAMES[r.strategy]}|{r.side}|{r.saving_vs_market_bp:+.6f}|{r.saving_vs_C3_bp:+.6f}|' for r in summary[summary.side.ne('both')&summary.strategy.ne('M')].itertuples())
    folds=pd.read_csv(OUT/'四折结果.csv')
    fp=folds[folds.minutes.eq(60)&folds.side.eq('both')].pivot(index='fold',columns='strategy',values='cost_bp')
    foldtable='\n'.join(f'|{i}|{r.C3:.6f}|{r.C36:.6f}|{r.C369:.6f}|{r.C3-r.C369:+.6f}|' for i,r in fp.iterrows())
    ms=pd.read_csv(OUT/'信号缺失原因.csv')
    missingtable='\n'.join(f'|{r.decision_second}|{r.missing_reason}|{r.tasks}|' for r in ms.itertuples())
    diag9=diag[diag.strategy.eq('C369')&diag.decision_second.eq(9)].iloc[0]
    txt=f'''# 初始19档：增加第6、9秒执行判断

完成时间：2026-09-28。结论：**本轮没有改善执行成本，保留只在第3秒判断作为成本基准。** 新增判断缩短等待，但兜底减少不足以代表总成本下降。

## 固定口径

- 18个日期，原62,899首小时逐秒共同任务；开盘首分钟1,080任务。起点是各名义整秒之后0.5秒内首条快照的实际时间T。首小时为开盘后09:30–10:30。
- 冻结上一轮观察3秒、预测1秒的四折五因子模型。使用最新OFI、报价位移、末价盘口位置、OFI方向、涨跌次数失衡；没有重训或调参。
- 初始买限价为LastPrice−19×0.2，卖为LastPrice+19×0.2。在第3／6／9秒，分别用[T,T+3]／[T+3,T+6]／[T+6,T+9]形成信号；各次均预测其观察结束后未来1秒。
- 买遇预测涨、卖遇预测跌，提交市价；其余信号按该次当时LastPrice被动19档限价。相同限价不重复提交。原来只在T+3判断的方案也采用同一1秒模型，不是更早预测3秒的旧C。
- 比较M立即市价、C3仅第3秒、C36第3/6秒、C369第3/6/9秒四组。每组保持10秒兜底，单位订单、两条严格未来快照延迟、限价结算及原单优先规则。
- 信号无法计算时保留当时原单，等下次判断或T+10兜底；不填补、不将缺失当作不变信号，不因后来信号缺失丢弃早成交订单。
- 第9秒限价命令若仍在途，10秒照常提交市价兜底，两条命令按各自到达时刻执行；旧单在替换到达前仍有效。若已有市价在途，10秒不重复提交市价。10秒不是强行成交时间。

## 开盘后首小时

{header}
{table(60)}

新增6/9秒判断后相对原C3贵0.016620bp，平均完成时间缩短约0.902秒，兜底比例从37.22%降至9.39%。仍比立即市价节约0.040462bp，但低于原C3节约0.057082bp。

## 开盘首分钟

{header}
{table(1)}

首分钟C369比原C3贵0.017943bp，配对区间跨0；没有看到首分钟问题得到改善。三种限价策略均比该范围立即市价平均成本高。

![成本对照](重复判断_成本与配对节约.png)

## 配对差异及日期稳定性

|范围|比较|平均节约bp|95%日期块区间bp|改善/变差/相同日期|
|---|---|---:|---|---:|
{pairtable}

首小时第6秒新增判断平均贵0.013762bp，增加第9秒再贵0.002858bp。均为原订单严格配对、逐日等权；首小时C369相对C3在14/18日成本更高。

|模型折|C3成本bp|C36成本bp|C369成本bp|C369相对C3节约bp|
|---|---:|---:|---:|---:|
{foldtable}

区间使用5日循环日期块、5,000次配对重采样，种子20260928。所有日期之前均已参与研究、1秒预测期限也曾用这些执行结果选择；区间仅为描述性探索，没有做历史选择校正。

## 你的假设哪一部分成立？

原C3中，10秒兜底订单对全体任务的相对市价贡献约−0.225145bp；第3秒市价订单贡献约−0.433979bp。不能将所有不利贡献都解释为“等待到10秒未成交”。

更有用的配对方法是固定原C3的最终成交群体，看相同订单改用C369后改变多少：

|范围|原C3中的最终成交群体|该群体原占比|改为C369对全体任务的节约贡献bp|
|---|---|---:|---:|
{attrtable}

首小时：原本会10秒兜底的订单，新增操作确实带来+0.055628bp的总贡献；但原本会初始限价或第3秒限价成交的订单合计损失约−0.072248bp。净值−0.016620bp。因此“救回部分兜底订单”与“打断原限价成交机会”同时发生，后者超过前者。

这个归因使用原方案的事后成交类型，决策当时不知道订单最终属于哪个群体，不能直接变成只对某群体干预的可执行规则。也不能因原兜底比例下降而把降低了多少风险说成已得到因果证明。

![成交归因](重复判断_成交构成与损失来源.png)

## 买卖分开

|范围|策略|方向|相对同方向市价节约bp|相对同方向C3节约bp|
|---|---|---|---:|---:|
{sidetable}

主表按日期等权、买卖各半；未对不同方向另选规则。

## 缺失、延迟和验收

|判断秒|缺失原因|潜在信号任务数|
|---:|---|---:|
{missingtable}

第6秒197个、第9秒201个任务的潜在信号缺失；这些包含之前已成交、不再需要判断的任务。实际到达判断点的缺失订单另存`实际到达判断点的信号缺失.csv`，均继续保留原限价、按下一次判断或截止时间处理。所有原62,899任务保留，本轮新增排除{audit['original_tasks']-audit['common_tasks']}。

C369第9秒实际提交{int(diag9.submissions):,}条命令，其中{int(diag9.after_T10)}条计划到达晚于T+10；旧单可能在命令到达前成交。没有把这类改单当作即时生效，完整记录命令是否实际到达。

- {sum(r['single_decision_exact'] for r in audit['per_day']):,}笔C3逐笔复现前轮；同数立即市价订单从原行情复核。
- {sum(r['independently_checked_orders'] for r in audit['per_day']):,}笔限价策略订单使用独立最早成交扫描复核；已成交订单在新增判断点之前保持一致。
- {sum(r['causal_checks'] for r in audit['per_day']):,}次窗口截断与窗口外扰动检查；188,697条三时点记录的起止边界逐条核实；保存模型参数可复算有效概率和信号。
- 发现并修复一个新脚本的浮点时间计算问题：用(T+s)−3可能轻微移动起始边界，现改为T+(s−3)，第3秒直接保留原始T。第3秒概率逐条与前轮一致，所有最终保存的窗口边界均符合当前表达；失败的回放未进入汇总。旧实验未受影响。
- {tests['tests_run']}项测试通过；成本从成交价格复算、成交比例和贡献加总、买卖合并及同任务配对均校验。原行情、模型、旧结果与公共两阶段撮合文件的SHA256保持不变。
- L1模拟保留无排队、无冲击、无费用、无部分成交限制；本实验只证明本轮规则的历史表现，不代表任何新增判断规则都不会有效。

## 文件与复现

本目录保存两张PNG/SVG、完整汇总与分方向CSV、逐日表现、成交群体配对贡献、成交方式转移表、全部逐任务成交、信号/特征、指令及轨迹样例、冻结模型和验收。独立包为`result/初始19档_增加6秒9秒判断.zip`，包含SHA256清单。

运行顺序：`python -m scripts.run_repeated_decisions`、`python scripts/test_opening.py result/opening_execution/initial_offset_selection/repeated_decisions`、`python -m scripts.deliver_repeated_decisions`。不重训模型，不更新旧实验，不部署策略，不上传远端。
'''
    (OUT/'增加6秒9秒判断报告.md').write_text(txt,encoding='utf-8')


def verify(orders,summary,pairs):
    old=pd.read_parquet(SOURCE/'四组逐任务成交.parquet')
    old=old[old.strategy.isin(['M','fixed_1s'])].copy();old['strategy']=old.strategy.replace({'fixed_1s':'C3'})
    joined=orders[orders.strategy.isin(['M','C3'])].merge(old,on=KEY+['strategy'],suffixes=('','_old'),validate='one_to_one')
    assert len(joined)==251596
    for col in ['task_seconds','fill_seconds','fill_ticks','cost_bp','elapsed_seconds']:
        np.testing.assert_allclose(joined[col],joined[col+'_old'],atol=0,rtol=0)
    for (minutes,side,strategy),g in summary.groupby(['minutes','side','strategy']):
        z=orders[orders.nominal_second.lt(minutes*60)&orders.strategy.eq(strategy)]
        if side!='both':z=z[z.direction.eq(1 if side=='buy' else -1)]
        np.testing.assert_allclose(z.groupby('trade_date').cost_bp.mean().mean(),g.cost_bp.iloc[0],atol=1e-12)
    for r in pairs.itertuples():
        z=orders[orders.nominal_second.lt(r.minutes*60)]
        if r.side!='both':z=z[z.direction.eq(1 if r.side=='buy' else -1)]
        daily=z.groupby(['trade_date','strategy']).cost_bp.mean().unstack().sort_index()
        gain=daily[r.reference]-daily[r.candidate]
        np.testing.assert_allclose(gain.mean(),r.saving_bp,atol=1e-12)
        np.testing.assert_allclose(date_block_interval(gain,seed=20260928),[r.low_bp,r.high_bp],atol=1e-12)
    save_json(OUT/'交付验收.json',dict(baseline_orders_reproduced=len(joined),summary_rows=len(summary),paired_intervals=len(pairs),all_verified=True))


def package():
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and '逐日回放' not in p.parts and p.name!='文件清单.csv')
    # Include compact trace examples; aggregate parquet files already include all daily rows.
    files+=sorted((OUT/'逐日回放').glob('*_traces.json'))
    manifest=pd.DataFrame([dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    manifest.to_csv(OUT/'文件清单.csv',index=False,encoding='utf-8-sig')
    archive=ROOT/'result/初始19档_增加6秒9秒判断.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in [*files,OUT/'文件清单.csv']:z.write(p,arcname=p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert len(z.infolist())==len(files)+1
        for r in manifest.itertuples():
            assert z.getinfo(r.path).file_size==r.bytes
            with z.open(r.path) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r.sha256
    save_json(archive.with_suffix('.verification.json'),dict(path=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,
        sha256=sha(archive),entries=len(files)+1,all_entry_sizes_and_sha256_verified=True))
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,entries=len(files)+1),ensure_ascii=False),flush=True)


def main():
    orders=pd.read_parquet(OUT/'全部逐任务成交.parquet')
    summary=pd.read_csv(OUT/'成本与节约汇总.csv')
    pairs=pd.read_csv(OUT/'配对改善.csv')
    contribution=pd.read_csv(OUT/'成交方式贡献汇总.csv')
    verify(orders,summary,pairs)
    attribution,diag,missing=diagnose(orders,summary)
    cost_figure(summary,pairs)
    attribution_figure(summary,contribution,attribution)
    report(summary,pairs,attribution,diag,missing)
    package()


if __name__=='__main__':main()
