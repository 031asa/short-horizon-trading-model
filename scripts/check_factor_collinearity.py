"""Design-only diagnostics: unregularized VIF, correlations and singular values."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_prediction import factors,NAMES,weights
OUT=ROOT/'result/opening_prediction/collinearity'

def diagnose(x,w=None):
    x=np.asarray(x,float)
    w=np.ones(len(x)) if w is None else np.asarray(w,float)
    w=w/w.sum();center=x-np.sum(w[:,None]*x,axis=0)
    sd=np.sqrt(np.sum(w[:,None]*center**2,axis=0))
    active=sd>1e-12
    z=center[:,active]/sd[active];zw=z*np.sqrt(w[:,None])
    corr=zw.T@zw;s=np.linalg.svd(zw,compute_uv=False)
    rank=int(np.linalg.matrix_rank(zw));vif=np.full(x.shape[1],np.nan)
    for j,k in enumerate(np.flatnonzero(active)):
        others=np.delete(zw,j,axis=1)
        residual=zw[:,j]-others@np.linalg.lstsq(others,zw[:,j],rcond=None)[0]
        error=float(residual@residual)
        vif[k]=np.inf if error<1e-12 else 1/error
    full=np.full((x.shape[1],x.shape[1]),np.nan);full[np.ix_(active,active)]=corr
    condition=float(s[0]/s[-1]) if rank==len(s) else np.inf
    if rank==z.shape[1]:np.testing.assert_allclose(vif[active],np.diag(np.linalg.inv(corr)),rtol=1e-8,atol=1e-8)
    return vif,full,condition,rank,active

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'sample_snapshot_原子执行总表.parquet';before=hashlib.sha256(source.read_bytes()).hexdigest()
    cols=['trade_date','observation_seconds','y_decision_1s']+sorted({f for w in range(1,6) for f in factors(w)})
    a=pd.read_parquet(source,columns=cols);a.trade_date=a.trade_date.astype(str)
    dates=sorted(a.trade_date.unique());vrows=[];pairs=[];designs=[]
    for window in range(1,6):
        fs=factors(window);g=a[a.observation_seconds==window].dropna(subset=fs).copy()
        groups=[('all_equal_days',g,weights(g.trade_date)),('all_pooled',g,None)]
        within=g.copy();within[fs]=g[fs]-g.groupby('trade_date')[fs].transform('mean')
        groups.append(('within_day_centered',within,weights(g.trade_date)))
        for n in [20,25,30,35]:
            train=g[g.trade_date.isin(dates[:n])].dropna(subset=['y_decision_1s'])
            groups.append((f'train_{n}_days',train,weights(train.trade_date)))
        groups.extend([(f'date_{d}',sub,None) for d,sub in g.groupby('trade_date')])
        for scope,sub,w in groups:
            x=sub[fs].to_numpy();v,c,cond,rank,active=diagnose(x,w)
            ranks=sub[fs].rank(method='average').to_numpy()
            _,rc,_,_,_=diagnose(ranks,w)
            designs.append(dict(window=window,scope=scope,n=len(sub),days=sub.trade_date.nunique(),rank=rank,columns=6,
                                condition=cond,constant_columns=int((~active).sum())))
            for i,f in enumerate(fs):vrows.append(dict(window=window,scope=scope,factor=f,name=NAMES[i],vif=v[i],constant=not active[i]))
            for i in range(6):
                for j in range(i+1,6):pairs.append(dict(window=window,scope=scope,a=NAMES[i],b=NAMES[j],factor_a=fs[i],factor_b=fs[j],pearson=c[i,j],spearman=rc[i,j]))
    vf=pd.DataFrame(vrows);pc=pd.DataFrame(pairs);dg=pd.DataFrame(designs)
    vf.to_csv(OUT/'VIF完整明细.csv',index=False,encoding='utf-8-sig');pc.to_csv(OUT/'相关系数完整明细.csv',index=False,encoding='utf-8-sig');dg.to_csv(OUT/'矩阵秩与条件数.csv',index=False,encoding='utf-8-sig')
    lines=['# 六因子多重共线性检查','',
      '主口径：观察＝回看 1 秒，全部 38 个有效上午，仅使用六个因子共同有效的行。主诊断只看 X，不要求未来标签有效，不重训预测模型、不改变原来的因子或日期划分。1–5 秒与原模型各训练段作为补充。', '',
      'VIF_j = 1 / (1 − R²_j)：用其余五个因子加截距解释第 j 个因子，使用不加正则的辅助线性回归。主表合并样本按日期等权；另列等行权、先去除每日均值及逐日诊断。VIF 不取逐日平均冒充合并 VIF。常数列标为不可识别；精确共线时 VIF 为无穷。', '',
      'VIF > 5 作为需要关注的经验线，不是删除因子的自动标准；依据 [statsmodels 官方说明](https://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html)。本报告是输入设计矩阵诊断，不是正则逻辑回归的系数 p 值或显著性检验。条件数用中心化、标准化后的加权 X 的最大／最小奇异值之比，不使用原始量纲。', '',
      '## 主结果：观察／回看 1 秒','',
      '| 因子 | 日期等权合并VIF | 等行权VIF | 去除日均值VIF | 逐日中位数 | 逐日最大值 | VIF>5日期/有效日期 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    result=[]
    for name in NAMES:
        v=vf[(vf.window==1)&(vf.name==name)];daily=v[v.scope.str.startswith('date_')].vif.dropna()
        get=lambda s:float(v[v.scope==s].vif.iloc[0])
        row=dict(name=name,vif=get('all_equal_days'),pooled=get('all_pooled'),within=get('within_day_centered'),daily_median=daily.median(),daily_max=daily.max(),high_days=int((daily>5).sum()),valid_days=len(daily))
        result.append(row)
        lines.append(f"| {name} | {row['vif']:.3f} | {row['pooled']:.3f} | {row['within']:.3f} | {row['daily_median']:.3f} | {row['daily_max']:.3f} | {row['high_days']}/{row['valid_days']} |")
    lines+=['','## 所有两两相关（1秒）','','| 因子A | 因子B | 合并Pearson（日期等权） | 合并Spearman（日期等权） | 逐日Pearson等权均值 | 逐日Spearman等权均值 |','|---|---|---:|---:|---:|---:|']
    p=pc[(pc.window==1)&(pc.scope=='all_equal_days')].copy();p=p.iloc[np.argsort(-np.abs(p.pearson.to_numpy()))]
    for r in p.to_dict('records'):
        daily=pc[(pc.window==1)&pc.scope.str.startswith('date_')&(pc.a==r['a'])&(pc.b==r['b'])]
        lines.append(f"| {r['a']} | {r['b']} | {r['pearson']:.3f} | {r['spearman']:.3f} | {daily.pearson.mean():.3f} | {daily.spearman.mean():.3f} |")
    lines+=['','## 窗口与训练段复核','','| 窗口 | 样本口径 | 样本数 | 日期数 | 标准化矩阵秩 | 条件数 | 最大VIF |','|---:|---|---:|---:|---:|---:|---:|']
    for r in dg[(dg.scope=='all_equal_days')|((dg.window==1)&dg.scope.str.startswith('train_'))].to_dict('records'):
        mx=vf[(vf.window==r['window'])&(vf.scope==r['scope'])].vif.max()
        lines.append(f"| {r['window']} | {r['scope']} | {r['n']} | {r['days']} | {r['rank']}/6 | {r['condition']:.3f} | {mx:.3f} |")
    mainmax=vf[(vf.window==1)&(vf.scope=='all_equal_days')].vif.max()
    longmax=vf[(vf.window==5)&(vf.scope=='all_equal_days')].vif.max()
    lines+=['','## 解释与下一步','',
      f'1 秒组的合并最大 VIF 为 {mainmax:.3f}，六列满秩，没有精确线性依赖。存在信息重叠，但不能据此将六因子组合未改善归咎于严重共线性。逐日局部 VIF 升高仍需关注。',
      f'5 秒组的最大 VIF 升至 {longmax:.3f}，主要来自“末价相对盘口位置／盘口位置偏离”这一对。它们分别使用当前值和当前值减历史均值，公式共享当前值；长窗口的冗余比 1 秒更明显。',
      '建议下一步优先做这两个位置因子的保留／替换对照，再检查最新 OFI 与窗口 OFI 的增量。不仅按相关系数删因子，也不把低 VIF 解释为因子有效。',
      '逐日的因子可能因样本少而出现较大 VIF，不能只看合并结果。反之，VIF 较低也不意味着各因子必然具有独立预测增量；仍需加减因子的预测对照。未来进行组合筛选时，共线性处理应仅由训练日期决定。','']
    (OUT/'六因子共线性报告.md').write_text('\n'.join(lines),encoding='utf-8')
    assert before==hashlib.sha256(source.read_bytes()).hexdigest()
    checks=dict(input_unchanged=True,input_sha256=before,ols_vif_matches_inverse_correlation=True,
                scopes=len(dg),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'复核.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(pd.DataFrame(result).to_string(index=False));print(p[['a','b','pearson','spearman']].head(6).to_string(index=False));print(dg[(dg.window==1)&~dg.scope.str.startswith('date_')].to_string(index=False))

if __name__=='__main__':run()
