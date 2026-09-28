"""Static comparison and paired accounting for opening-minute early execution."""
from scripts.run_early_execution import OUT,ROOT,BASE,CODES,NAMES,SCOPES,SCOPE_NAMES,KEY,NUMERIC,sha,save_json
import json,zipfile,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
    'axes.unicode_minus':False,'svg.fonttype':'path','font.size':10.5})
BLUE='#337ca0';GREEN='#258a7d';RED='#bb625a';ORANGE='#c89133';GRAY='#7e8894'
COLORS={'M':GRAY,'C3':BLUE,'E1':ORANGE,'E2':GREEN,'E1_10':'#d7b170','E2_10':'#76aaa1','M10':'#8466a2'}
SHORT={'M':'立即市价','C3':'原3秒','E1':'加1秒检查','E2':'加2秒检查','E1_10':'前10秒\n加1秒检查','E2_10':'前10秒\n加2秒检查','M10':'前10秒\n直接市价'}


def read(name):return pd.read_csv(OUT/(name+'.csv'))


def savefig(fig,name,title,footer):
    fig.suptitle(title,fontsize=17,y=.99);fig.get_layout_engine().set(rect=(0,.09,1,.81))
    fig.text(.5,.025,footer,ha='center',fontsize=10,linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=210,facecolor='white')
    plt.close(fig)


def figures(s):
    fig,axes=plt.subplots(2,2,figsize=(16,11),layout='constrained')
    allg=s[s.scope.eq('first_minute')&s.side.eq('both')].set_index('strategy').loc[CODES]
    ax=axes[0,0];ax.bar(range(7),allg.cost_bp,color=[COLORS[c] for c in CODES],width=.65)
    ax.axhline(allg.loc['M','cost_bp'],color=GRAY,ls='--')
    for i,r in enumerate(allg.itertuples()):ax.text(i,r.cost_bp+.02,f'{r.cost_bp:.3f}',ha='center',fontsize=10)
    ax.set(xticks=range(7),xticklabels=[SHORT[c] for c in CODES],ylim=(0,1.52),ylabel='首分钟平均执行成本（bp，低为好）',title='提前检查并未直接解决执行成本问题')
    ax.tick_params(axis='x',labelsize=9);ax.grid(axis='y',alpha=.15)
    ax=axes[0,1];cand=['E1','E2','E1_10','E2_10','M10']
    for i,c in enumerate(cand):
        r=allg.loc[c];ax.plot([r.vs_C3_low,r.vs_C3_high],[i,i],color=COLORS[c],lw=3)
        ax.scatter(r.saving_vs_C3_bp,i,color=COLORS[c],s=55,zorder=3)
        ax.annotate(f'{r.saving_vs_C3_bp:+.3f} [{r.vs_C3_low:+.3f}, {r.vs_C3_high:+.3f}]',
            (r.saving_vs_C3_bp,i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
    ax.axvline(0,color=GRAY,lw=1);ax.set(yticks=range(5),yticklabels=[SHORT[c] for c in cand],ylim=(-.5,4.65),
        xlabel='相对原3秒方案节约（bp，正为好）',title='配对95%日期块区间：早检查仍有不确定性')
    ax.margins(x=.3);ax.grid(axis='x',alpha=.15)
    ax=axes[1,0];g=s[s.scope.eq('first10')&s.side.eq('both')].set_index('strategy');codes=['M','C3','E1','E2']
    vals=g.loc[codes,'cost_bp'];ax.bar(range(4),vals,color=[COLORS[c] for c in codes],width=.6)
    for i,v in enumerate(vals):ax.text(i,v+.04,f'{v:.3f}',ha='center')
    ax.set(xticks=range(4),xticklabels=[SHORT[c] for c in codes],ylim=(0,3.05),ylabel='最前10秒任务平均执行成本（bp）',title='只看最前10秒：等1秒或2秒仍明显贵于市价')
    ax.axhline(g.loc['M','cost_bp'],color=GRAY,ls='--');ax.grid(axis='y',alpha=.15)
    ax=axes[1,1];b=read('10秒段结果')
    for code in ['C3','E1','E2','M10']:
        q=b[b.strategy.eq(code)].sort_values('bin_start')
        ax.plot(q.bin_start+4.5,q.saving_bp,'o-',color=COLORS[code],label=SHORT[code].replace('\n',''),lw=2)
    ax.axhline(0,color=GRAY,lw=1);ax.set(xticks=np.arange(6)*10+4.5,xticklabels=['00–09','10–19','20–29','30–39','40–49','50–59'],
        xlabel='09:30内名义发单秒数',ylabel='相对立即市价节约（bp）',title='每10秒任务分布：提前行动的收益并不均匀')
    ax.legend(fontsize=9,ncol=2,loc='lower right');ax.grid(alpha=.15)
    savefig(fig,'提前检查_成本与时段对照','优化等待代价：第1秒 / 第2秒加检查，是否比原3秒方案更好？\n短窗模型均预测未来1秒；仅不利信号提前市价，否则保留原限价与3秒流程',
        '18日，首分钟1,080任务、买卖各半、日期等权；最前10秒180任务。初始/后续19档、两快照延迟、10秒兜底不变。\n前10秒指09:30:00–09:30:09发起的任务，不是每笔订单自己的前10秒。区间未校正此前模型与时段探索。')
    fig,axes=plt.subplots(1,2,figsize=(16,6.6),layout='constrained')
    attr=read('原群体配对贡献');labels=['initial_limit','limit3','market3','market10']
    for j,c in enumerate(['E1','E2']):
        ax=axes[j];g=attr[attr.scope.eq('first_minute')&attr.side.eq('both')&attr.strategy.eq(c)].set_index('c3_stage')
        v=[g.loc[k,'gain_contribution_bp'] for k in labels];v.append(sum(v))
        ax.bar(range(5),v,color=[GREEN if x>=0 else RED for x in v],width=.65)
        for i,x in enumerate(v):ax.annotate(f'{x:+.3f}',(i,x),xytext=(0,6 if x>=0 else -16),textcoords='offset points',ha='center')
        ax.axhline(0,color=GRAY,lw=1);ax.set(xticks=range(5),xticklabels=['原初始\n限价成交','原3秒\n限价成交','原3秒\n市价成交','原10秒\n兜底成交','合计'],
            ylabel='相对原3秒方案的节约贡献（bp）',title=NAMES[c],ylim=(-.65,.59));ax.grid(axis='y',alpha=.15)
    savefig(fig,'提前检查_改善与损失来源','提前追单确实挽回一部分等待损失，也损失原本有利的限价成交\n所有分组使用同一笔任务配对，贡献分母为首分钟全部买卖订单',
        '以原C3最终成交类型作事后归因；实盘当时不知道一笔订单原本属于哪类，不能用这个分类充当决策信号。\n没有从任务中剔除早成交或缺少早期信号的订单。短窗缺失时继续原方案。')


def verify(o,s):
    assert len(o)==15120 and o.groupby(KEY).strategy.nunique().eq(7).all()
    originals=pd.read_parquet(BASE/'全部逐任务成交.parquet')
    originals=originals[originals.nominal_second.lt(60)&originals.strategy.isin(['M','C3'])]
    joined=o[o.strategy.isin(['M','C3'])].merge(originals,on=KEY+['strategy'],suffixes=('','_old'),validate='one_to_one')
    assert len(joined)==4320
    for k in NUMERIC:np.testing.assert_allclose(joined[k],joined[k+'_old'],atol=0,rtol=0)
    indexed=o.set_index(KEY+['strategy'])
    for code,parent in [('E1_10','E1'),('E2_10','E2'),('M10','M')]:
        for r in o[o.strategy.eq(code)].itertuples(index=False):
            ref=indexed.loc[(r.trade_date,r.nominal_second,r.direction,parent if r.nominal_second<10 else 'C3')]
            for k in NUMERIC:assert getattr(r,k)==ref[k]
    daily=read('逐日成本');checked=0
    for r in s.itertuples(index=False):
        lo,hi=SCOPES[r.scope];g=o[o.nominal_second.ge(lo)&o.nominal_second.lt(hi)&o.strategy.eq(r.strategy)]
        if r.side!='both':g=g[g.direction.eq(1 if r.side=='buy' else -1)]
        np.testing.assert_allclose(g.groupby('trade_date').cost_bp.mean().mean(),r.cost_bp,atol=1e-12)
        d=daily[daily.scope.eq(r.scope)&daily.side.eq(r.side)&daily.strategy.eq(r.strategy)].sort_values('trade_date')
        from utils.opening_two_stage import date_block_interval
        for ref in ['market','C3']:
            np.testing.assert_allclose(date_block_interval(d[f'saving_vs_{ref}_bp'],seed=20260928),[getattr(r,f'vs_{ref}_low'),getattr(r,f'vs_{ref}_high')],atol=1e-12)
            checked+=1
    tests=json.loads((OUT/'unit_test_verification.json').read_text(encoding='utf-8'))
    assert tests['passed'] and tests['tests_run']==168
    save_json(OUT/'交付验收.json',dict(baseline_orders_exact=len(joined),clock_policy_orders_exact=6480,summary_rows=len(s),
        paired_intervals=checked,all_verified=True,tests=tests))


def report(s):
    g=s[s.scope.eq('first_minute')&s.side.eq('both')].set_index('strategy')
    rows='\n'.join(f'|{NAMES[c]}|{g.loc[c,"cost_bp"]:.6f}|{g.loc[c,"saving_vs_C3_bp"]:+.6f}|[{g.loc[c,"vs_C3_low"]:+.6f},{g.loc[c,"vs_C3_high"]:+.6f}]|{g.loc[c,"saving_vs_market_bp"]:+.6f}|{g.loc[c,"elapsed_seconds"]:.3f}|' for c in CODES)
    q=s[s.scope.eq('first10')&s.side.eq('both')].set_index('strategy')
    first='\n'.join(f'|{NAMES[c]}|{q.loc[c,"cost_bp"]:.6f}|{q.loc[c,"saving_vs_market_bp"]:+.6f}|{q.loc[c,"saving_vs_C3_bp"]:+.6f}|' for c in ['M','C3','E1','E2'])
    side='\n'.join(f'|{NAMES[r.strategy]}|{r.side}|{r.saving_vs_C3_bp:+.6f}|[{r.vs_C3_low:+.6f},{r.vs_C3_high:+.6f}]|{r.saving_vs_market_bp:+.6f}|' for r in s[s.scope.eq('first_minute')&s.side.ne('both')&s.strategy.isin(['E1','E2','M10'])].itertuples())
    f=read('四折结果');f=f[f.scope.eq('first_minute')&f.side.eq('both')&f.strategy.isin(['E1','E2','M10'])]
    folds='\n'.join(f'|{r.fold}|{NAMES[r.strategy]}|{r.saving_vs_C3_bp:+.6f}|{r.saving_vs_market_bp:+.6f}|' for r in f.itertuples())
    a=json.loads((OUT/'回放验收.json').read_text(encoding='utf-8'))
    text=f'''# 开盘首分钟：提前检查能否降低等待损失

完成：2026-09-28。结论：**简单加第1/2秒不利信号检查尚未显示稳定成本改善；最前10秒发起任务改为立即市价，历史平均改善较明显，但完整首分钟仍未击败全部立即市价。** 本轮只比较固定候选，不自动替换原基准。

## 口径与可执行规则

- 首分钟18日、1,080相同任务；每任务独立回放买卖双向，共2,160订单/方案、7方案15,120条。没有抽掉早成交、未来标签缺失或早期信号缺失任务。全部任务都完成。
- 原C3：起点初始被动19档；第3秒观察3秒/预测1秒，买遇预测涨、卖遇预测跌则转市价，其余按当时LastPrice被动19档；10秒提交兜底。
- E1/E2：初始不变，在T+1/T+2用对应W1/W2冻结五因子模型预测其观察结束后未来1秒。只有不利argmax信号才提前市价；有利、不变、不可计算均保留原单。到T+3仍未成交且无市价在途，再按原W3信号处理。**不是用未来3秒信号倒填到第1秒。**
- E1_10/E2_10：只对09:30第0–9秒发起的任务启用早期检查，其他任务仍C3。M10：该0–9秒的任务立即市价，其余C3。这里的10秒是开盘墙上时钟分组，不是每笔订单已经等待10秒。
- 全部市价均保留两条严格未来快照延迟；起点立即市价在本首分钟样本平均T+1成交。旧限价在替换到达前继续有效，同快照旧单成交优先；初始订单或早期市价仍在途时按提交顺序执行，不提前生效或重复转市价。
- 复用此前按日期向前训练的W1/W2/W3四折模型，共12个，没有重新拟合、重选因子或扫描概率阈值。各折只使用较早日期训练，所有特征仅使用本任务开始到相应检查时刻。
- 初始和第3秒限价19档不变，不同时搜索更积极的初始档位。更早的首分钟初始档位研究已提示“挂浅”并不必然更省，增加成交率不是完整成本目标。
- 所有汇总先按日求均值再日期等权，买卖各半。配对区间为循环5日日期块、5,000次、seed20260928。已看过18日和首10秒亏损后提出本轮规则，区间未经历史搜索校正，不能当全新样本外验证。

## 首分钟整体成本

|方案|成本bp|相对原C3节约bp|配对95%区间|相对立即市价节约bp|平均完成秒|
|---|---:|---:|---|---:|---:|
{rows}

- E1：只改善0.001801bp，8日改善/10日变差，4折2正2负。平均更快，但基本没有降低成本。
- E2：改善0.075172bp，区间[-0.068417,0.196999]跨0，9日改善/9日变差，4折3正1负。可留为候选，不能声称已验证更优；相对市价仍贵0.246547bp。
- E1_10/E2_10：整体分别改善0.042017/0.044189bp，相对C3的区间均跨0。
- M10：整体改善0.208115bp，区间[0.078909,0.334859]，14日改善/4日变差，4折改善均为正；相对市价仍贵0.113604bp（节约区间[-0.307784,0.056860]）。这减少原C3相对市价历史平均劣势约64.69%，不是新增了正执行收益。
- 全首分钟直接市价的平均成本仍为0.939636bp，在本次7个候选里最低。不能因为M10击败C3，就说M10击败市价或具有预测优势。

![成本](提前检查_成本与时段对照.png)

## 只看最前10秒的180个任务

|方案|成本bp|相对立即市价节约bp|相对原C3节约bp|
|---|---:|---:|---:|
{first}

加1/2秒检查虽改善该段平均成本，但仍比同段立即市价贵约1bp。说明这段价格变化很快，等待更短不自动意味着值得等；现有预测器尚未在这段抵消等待代价。M10的改进是固定时段规则的历史结果，不把0–9秒当作已验证的最佳时段边界。

## 为什么提前处理仍未明显改善总账

按原C3最终成交类型做同一订单配对归因，贡献分母均为首分钟全部订单：

|新增检查|原初始限价组|原3秒限价组|原3秒市价组|原兜底组|合计|
|---|---:|---:|---:|---:|---:|
|第1秒|-0.208920|-0.481834|+0.365004|+0.327550|+0.001801|
|第2秒|-0.045005|-0.369539|+0.064384|+0.425332|+0.075172|

提前市价确实改善了部分原追单、兜底订单；但也把一些原本会在有利价格成交的限价单提前买贵/卖便宜。这与前轮增加6/9秒判断的取舍一致。原最终成交类型在当时未知，这张归因表不能直接转成“只对兜底组提前市价”的规则。

![归因](提前检查_改善与损失来源.png)

## 买卖与逐折表现

|方案|方向|相对原C3节约bp|95%区间|相对市价节约bp|
|---|---|---:|---|---:|
{side}

|前推折|方案|相对原C3节约bp|相对市价节约bp|
|---|---|---:|---:|
{folds}

## 下一步更有针对性的优化

1. 保留原C3、E2及M10作为固定对照，不把本轮历史最低自动部署。最前10秒单独评价，所有任务持续预测与完成，不靠删掉困难任务提高均值。
2. 提前决策应改为估计“现在转市价”和“保留原流程”的**条件成本差**。上涨概率只作为输入之一；还需考虑当前可执行报价相对起点的变化、价差、距原限价距离、短窗OFI/报价位移等。后续可在较早训练日生成两种动作的模拟成本标签，再在后续日期完整配对检验；标签可以用未来回放，决策输入不能使用未来。
3. 同时保留有利晚成交机会，用成本差而非方向argmax决定提前动作。按日期选规则并计入全部订单，不用原方案最终成交类型筛订单。新的成本模型尚未执行，本轮不声称已解决。

## 验收与交付

- 168项单元测试通过，新增8项覆盖早期信号缺失/报价陈旧、双向不利信号、同快照原单优先、初始/市价在途、保留原3秒规则等。
- {a['independently_checked_orders']:,}笔C3/E1/E2独立最早成交扫描；原C3与市价4,320笔逐值复核。
- {a['saved_prediction_matches']:,}条已存预测复现，432次任务开始之前/观察结束之后行情扰动与截断检查；12个模型参数可复算预测，训练日期严格早于对应任务日期。
- 1秒45任务、2秒3任务信号不全，均继续原3秒流程；无填补、无新增剔除。代码留存缺失成员和实际执行轨迹。
- 2,268笔不触发提前操作的执行与原方案一致，2,775笔在T+3之前已结束的执行不受后来信号改变影响。所有提前方案完成时间不晚于对应原C3。
- 原行情、模型、旧实验文件及公共两阶段撮合文件指纹不变。重复判断引擎新增可选早期检查，默认行为不变；不上传远端。
- 两张PNG/SVG、分方向/分时段/逐日/四折CSV、逐任务结果、短窗概率、下单记录、冻结模型、口径与校验文件保存在本目录。

运行：`python -m scripts.run_early_execution`；`python scripts/test_opening.py result/opening_execution/initial_offset_selection/early_decisions`；`python -m scripts.deliver_early_execution`。
'''
    (OUT/'提前检查与首10秒优化报告.md').write_text(text,encoding='utf-8')


def package():
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='文件清单.csv')
    m=pd.DataFrame([dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in files]);m.to_csv(OUT/'文件清单.csv',index=False,encoding='utf-8-sig')
    path=ROOT/'result/首分钟提前检查与首10秒优化.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in [*files,OUT/'文件清单.csv']:z.write(p,arcname=p.name)
    with zipfile.ZipFile(path) as z:
        assert len(z.infolist())==len(files)+1
        for r in m.itertuples():
            assert z.getinfo(r.path).file_size==r.bytes
            with z.open(r.path) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r.sha256
    save_json(path.with_suffix('.verification.json'),dict(path=str(path),entries=len(files)+1,sha256=sha(path),all_verified=True))
    print(json.dumps(dict(archive=str(path),entries=len(files)+1,bytes=path.stat().st_size),ensure_ascii=False))


def main():
    o=pd.read_parquet(OUT/'全部逐任务成交.parquet');s=read('成本与节约汇总')
    verify(o,s);figures(s);report(s);package()


if __name__=='__main__':main()
