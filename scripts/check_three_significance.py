"""Exploratory significance diagnostics, explicitly conditional on prior selection."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_prediction import SNAPSHOT,factors,fit_model
from scripts.run_opening_prediction import save_json
OUT=ROOT/'result/opening_prediction/significance'

def flip_p(v):
    v=np.asarray(v,float);n=len(v)
    signs=2*((np.arange(2**n,dtype=np.uint32)[:,None]>>np.arange(n))&1).astype(float)-1
    return float(np.mean(np.abs(signs@v)>=abs(v.sum())-1e-12))

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    data=json.loads((ROOT/'result/opening_prediction/看板数据.json').read_text(encoding='utf-8'))
    results=[]
    for h in [1,2,3]:
        a=pd.DataFrame([r for r in data['daily'] if r['model']=='snapshot' and r['sample']=='common_windows' and r['horizon']==h]).set_index('trade_date').sort_index()
        b=pd.DataFrame([r for r in data['baseline_daily'] if r['model']=='snapshot' and r['sample']=='common_windows' and r['horizon']==h]).set_index('trade_date').sort_index()
        assert a.index.equals(b.index) and len(a)==18
        for metric in ['accuracy','logloss']:
            diff=(a[metric]-b[metric]).to_numpy()*(1 if metric=='accuracy' else -1)
            rng=np.random.default_rng(20260923);boot=diff[rng.integers(0,18,size=(10000,18))].mean(1)
            # Conservative sensitivity: signs move together within each original test fold.
            blocks=np.array([diff[:5].sum(),diff[5:10].sum(),diff[10:15].sum(),diff[15:].sum()])
            results.append(dict(horizon=h,metric=metric,improvement=float(diff.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                                nominal_day_p=flip_p(diff),four_block_p=flip_p(blocks),positive_days=int((diff>0).sum())))
    order=np.argsort([r['nominal_day_p'] for r in results]);last=0.
    for i,k in enumerate(order):last=max(last,min(1.,(len(order)-i)*results[k]['nominal_day_p']));results[k]['holm_six_p']=last
    fits=json.loads((ROOT/'result/opening_prediction/各折模型参数.json').read_text(encoding='utf-8'))
    fits=[r for r in fits if r['model']=='snapshot' and r['horizon']==1]
    initial=next(r for r in fits if r['fold']==1)
    source=ROOT/'sample_snapshot_原子执行总表.parquet';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    a=pd.read_parquet(source,columns=['trade_date','observation_seconds','y_decision_1s']+factors(1));a.trade_date=a.trade_date.astype(str)
    train=a[(a.observation_seconds==1)&a.trade_date.isin(initial['train_dates'])].dropna(subset=factors(1)+['y_decision_1s'])
    groups=[g for _,g in train.groupby('trade_date')];rng=np.random.default_rng(20260923);coef=[]
    base_sd=np.asarray(initial['scale'])
    for j in range(1000):
        chunks=[groups[k].assign(bootstrap_day=i) for i,k in enumerate(rng.integers(0,len(groups),len(groups)))]
        g=pd.concat(chunks,ignore_index=True)
        m=fit_model(g[SNAPSHOT].to_numpy(),np.sign(g.y_decision_1s).to_numpy(int),g.bootstrap_day.to_numpy(),initial['C'])
        coef.append((m['model'].coef_[2]-m['model'].coef_[0])/m['scale'].scale_*base_sd)
        if (j+1)%250==0:print(f'coefficient day bootstrap {j+1}/1000',flush=True)
    coef=np.asarray(coef);point=(np.asarray(initial['coefficients'])[2]-np.asarray(initial['coefficients'])[0]);coefficient=[]
    for i,f in enumerate(SNAPSHOT):
        coefficient.append(dict(factor=f,contrast='log P(up)/P(down) per original training SD',estimate=float(point[i]),
          conditional_ci95=np.quantile(coef[:,i],[.025,.975]).tolist(),
          conditional_bonferroni_three_ci=np.quantile(coef[:,i],[.05/6,1-.05/6]).tolist(),positive_bootstrap_share=float(np.mean(coef[:,i]>0)),
          folds_up_minus_down=[float(np.asarray(r['coefficients'])[2,i]-np.asarray(r['coefficients'])[0,i]) for r in fits]))
    assert digest==hashlib.sha256(source.read_bytes()).hexdigest()
    save_json(OUT/'三因子显著性诊断.json',dict(performance=results,coefficients=coefficient,bootstrap_replicates=1000,seed=20260923,
      training_dates=initial['train_dates'],fixed_C=initial['C'],input_sha256=digest,
      caveat='Exploratory only. No adjustment for factor/horizon screening. Day sign flips assume independent symmetric differences. Bootstrap holds model and C fixed; overlapping training folds are not independent coefficient replications.'))
    pd.DataFrame(coef,columns=SNAPSHOT).to_csv(OUT/'系数按日重采样.csv',index=False,encoding='utf-8-sig')
    names=['最新 OFI','最新报价位移','末价相对盘口位置']
    lines=['# 三因子显著性诊断','','这是既有选择之后的探索性检验，不能宣称通过独立样本的 5%／1% 显著性验证。原三因子模型没有修改。','',
      '## 系数稳定性（主要目标：未来1秒）','','用第一折前20个有效日期的模型，固定当时选择的 C，以整日为单位重抽样1000次。系数为上涨与下跌的对数概率比之差，每增加原训练样本1个标准差对应的变化；不是上涨概率的直接变化。每次重新标准化，最后换算至同一原始标准差尺度。', '',
      '| 因子 | 系数 | 条件95%区间 | 三系数Bonferroni条件区间 | 重采样正向比例 |','|---|---:|---|---|---:|']
    for n,r in zip(names,coefficient):lines.append(f"| {n} | {r['estimate']:.3f} | {r['conditional_ci95']} | {r['conditional_bonferroni_three_ci']} | {r['positive_bootstrap_share']:.1%} |")
    lines+=['','这不是普通无惩罚回归的系数p值。区间条件于已选因子、模型与正则参数；不包含重新筛选因子或 C 的不确定性，也不消除正则化偏差。四折系数重叠使用训练日期，不算四次独立证据。','',
      '## 组合相对训练比例基准的改善','','18个检验日、882个共同任务。先逐日求准确率／概率损失差，再枚举全部日期符号翻转，给出双侧名义p值。正的改善表示模型更好。另对六项检验作Holm校正，但没有校正整个历史研究中的因子和期限选择。','',
      '| 未来秒数 | 指标 | 平均改善 | 日期名义p | 六项Holm p | 四个检验段整块翻转p |','|---:|---|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['horizon']} | {r['metric']} | {r['improvement']:.5f} | {r['nominal_day_p']:.6f} | {r['holm_six_p']:.6f} | {r['four_block_p']:.3f} |")
    lines+=['','日级符号翻转要求日期差值独立且在零假设下符号可交换（对称）。任务已按日聚合，避免将重叠快照当成独立样本，但跨日依赖和重叠训练仍可能影响结论。四段敏感性检查将每个原检验段内的日期整体翻转；只有四段，双侧p值最小为0.125，不能达到5%。','',
      '因此：日级名义显著与独立样本显著不同，系数方向稳定也不等于该因子不可替代。每个因子的预测增量仍应通过移除该因子的模型对照检验。','',
      '方法参考：[SciPy 配对符号翻转检验说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)。']
    (OUT/'三因子显著性说明.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(performance=results,coefficients=coefficient),ensure_ascii=False,indent=2))

if __name__=='__main__':run()
