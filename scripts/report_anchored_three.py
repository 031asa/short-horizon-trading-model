"""Data tables and readable report; deliberately no new dashboard."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_task_window import BASE,NAMES,INTERACTIONS,CATALOG
from scripts.run_opening_prediction import save_json
from scripts.run_anchored_three import OUT,KEY

STRUCTURE={'baseline':'三因子','add1':'三因子＋1项','add2':'三因子＋2项','composite':'三因子＋复合组'}

def intervals(values):
    x=np.asarray(values,float);rng=np.random.default_rng(20260923)
    starts=rng.integers(0,len(x),size=(5000,int(np.ceil(len(x)/5))))
    ix=((starts[:,:,None]+np.arange(5))%len(x)).reshape(5000,-1)[:,:len(x)]
    ci=np.quantile(x[ix].mean(axis=1),[.025,.975])
    ci[np.abs(ci)<1e-12]=0.
    return ci.tolist()

def run():
    a=pd.read_parquet(OUT/'固定三因子原子表.parquet');p=pd.read_parquet(OUT/'逐任务预测.parquet')
    models=json.loads((OUT/'模型参数.json').read_text(encoding='utf-8'))
    selection=json.loads((OUT/'选择过程.json').read_text(encoding='utf-8'))
    decisions=json.loads((OUT/'窗口选择.json').read_text(encoding='utf-8'))
    registry=[dict(feature=f,name=name,group=group,logical_prior=prior,formula=formula,
        role='fixed_core' if f in BASE else 'candidate_main',required_parents='') for f,name,group,prior,formula in CATALOG]
    registry.extend(dict(feature=f,name=NAMES[f],group='复合表达',logical_prior='uncertain',formula=' * '.join(parents),
        role='candidate_interaction',required_parents='|'.join(parents)) for f,parents in INTERACTIONS.items())
    pd.DataFrame(registry).to_csv(OUT/'候选特征清单.csv',index=False,encoding='utf-8-sig')
    regular=p[p.method.isin(['baseline','selected'])]
    counts=regular.groupby(KEY).size();keys=counts[counts==20].reset_index()[KEY]
    label_cols=[f'{k}_{h}' for k in ['cum','inc'] for h in range(1,11)]
    p=p.merge(a[KEY+['window']+label_cols],on=KEY+['window'],validate='many_to_one')
    p['model']=np.where(p.method.str.startswith('adaptive'),p.method,p.method+'_w'+p.window.astype(str))
    common=p.merge(keys,on=KEY,validate='many_to_one');common.to_parquet(OUT/'共同任务预测.parquet',index=False)
    daily=[]
    for scope,data in [('own',p),('common',common)]:
        for (model,date),g in data.groupby(['model','trade_date']):
            prob=g[['p_down','p_flat','p_up']].to_numpy();signal=prob.argmax(1)-1
            for kind in ['cum','inc']:
                for h in range(1,11):
                    y=np.sign(g[f'{kind}_{h}'].to_numpy());valid=np.isfinite(y);n=int(valid.sum())
                    if not n:continue
                    labels=y[valid].astype(int);pr=prob[valid];hits=signal[valid]==labels
                    recalls=[float(hits[labels==c].mean()) for c in [-1,0,1] if np.any(labels==c)]
                    row=dict(sample=scope,model=model,trade_date=date,fold=str(g.fold.iloc[0]),kind=kind,horizon=h,n=n,
                        accuracy=float(hits.mean()),flat_share=float((labels==0).mean()),balanced_accuracy=float(np.mean(recalls)),
                        logloss3=float(-np.log(np.clip(pr[np.arange(n),labels+1],1e-15,1)).mean()) if kind=='cum' and h==3 else np.nan)
                    daily.append(row)
    d=pd.DataFrame(daily);d.to_csv(OUT/'逐日评价.csv',index=False,encoding='utf-8-sig')
    s=d.groupby(['sample','model','kind','horizon']).agg(days=('trade_date','nunique'),n=('n','sum'),
        accuracy=('accuracy','mean'),flat_share=('flat_share','mean'),balanced_accuracy=('balanced_accuracy','mean'),logloss3=('logloss3','mean')).reset_index()
    s['coverage']=s.n/1080;s.to_csv(OUT/'全部期限结果.csv',index=False,encoding='utf-8-sig')
    pairs=[]
    for scope in ['own','common']:
        for w in list(range(1,11))+['adaptive']:
            base='adaptive_baseline' if w=='adaptive' else f'baseline_w{w}'
            methods=['adaptive_selected'] if w=='adaptive' else [f'{m}_w{w}' for m in ['add1','add2','composite','selected']]
            for kind in ['cum','inc']:
                for h in range(1,11):
                    sub=d[(d['sample']==scope)&(d.kind==kind)&(d.horizon==h)]
                    pivot=sub.pivot(index='trade_date',columns='model',values='accuracy')
                    for model in methods:
                        if model not in pivot:continue
                        diff=(pivot[model]-pivot[base]).dropna();ci=intervals(diff.to_numpy())
                        pairs.append(dict(sample=scope,model=model,baseline=base,window=w,kind=kind,horizon=h,days=len(diff),
                            accuracy_gain=float(diff.mean()),better_days=int((diff>0).sum()),worse_days=int((diff<0).sum()),
                            ci_low=ci[0],ci_high=ci[1]))
    paired=pd.DataFrame(pairs);paired.to_csv(OUT/'相对三因子增益.csv',index=False,encoding='utf-8-sig')
    f=d[(d['sample']=='common')&(d.kind=='cum')].groupby(['model','fold','horizon']).agg(
        accuracy=('accuracy','mean'),days=('trade_date','nunique'),n=('n','sum')).reset_index()
    f.to_csv(OUT/'四折表现.csv',index=False,encoding='utf-8-sig')
    combinations=[]
    for record in selection:
        for method,choice in record['choices'].items():
            if choice is None:continue
            spec=next(c for c in record['candidates'] if c['candidate_id']==choice['candidate_id'])
            added=[x for x in choice['features'] if x not in BASE]
            combinations.append(dict(fold=record['fold'],window=record['window'],method=method,structure=choice['structure'],
                added='|'.join(added),added_names='＋'.join(NAMES[x] for x in added) or '无新增',features='|'.join(choice['features']),
                C=choice['C'],max_vif=spec['refit_stats']['vif'],max_pair_corr=spec['refit_stats']['correlation'],
                validation_accuracy3=choice['accuracy3'],validation_accuracy5to10=choice['accuracy5to10'],
                validation_logloss3=choice['logloss3'],core_r2=json.dumps(spec['core_r2'],ensure_ascii=False),
                interaction_formula='; '.join(x+' = '+' * '.join(INTERACTIONS[x]) for x in added if x in INTERACTIONS)))
    combos=pd.DataFrame(combinations);combos.to_csv(OUT/'各折组合成员.csv',index=False,encoding='utf-8-sig')
    freq=combos[combos.method=='selected'].groupby(['window','added','added_names']).agg(
        selected_folds=('fold','count'),max_vif=('max_vif','max')).reset_index()
    freq.to_csv(OUT/'新增组合选择频次.csv',index=False,encoding='utf-8-sig')
    def metric(model,h=3,kind='cum'):
        return s[(s['sample']=='common')&(s.model==model)&(s.kind==kind)&(s.horizon==h)].iloc[0]
    main=[]
    for w in range(1,11):
        b=metric(f'baseline_w{w}');e=metric(f'selected_w{w}');pr=paired[(paired['sample']=='common')&(paired.model==f'selected_w{w}')&(paired.kind=='cum')&(paired.horizon==3)].iloc[0]
        distribution=combos[(combos.window==w)&(combos.method=='selected')].structure.value_counts().to_dict()
        main.append(dict(window=w,baseline_accuracy=b.accuracy,enhanced_accuracy=e.accuracy,gain=e.accuracy-b.accuracy,
            n=int(e.n),days=int(e.days),structures='；'.join(STRUCTURE[k]+'×'+str(v) for k,v in distribution.items()),
            better_days=int(pr.better_days),ci_low=pr.ci_low,ci_high=pr.ci_high,baseline_loss3=b.logloss3,enhanced_loss3=e.logloss3))
    main=pd.DataFrame(main);main.to_csv(OUT/'主对照表.csv',index=False,encoding='utf-8-sig')
    used_windows=[r['choice']['window'] for r in decisions if r['fold']!='final']
    final=json.loads((OUT/'下一批行情配置.json').read_text(encoding='utf-8'))
    b=metric('adaptive_baseline');e=metric('adaptive_selected')
    paired_adaptive=paired[(paired['sample']=='common')&(paired.model=='adaptive_selected')&(paired.kind=='cum')&(paired.horizon==3)].iloc[0]
    lines=['# 固定三因子增强与回看窗口选择','',
      f'每折固定保留最新OFI、报价位移、末价盘口位置，并重新拟合全部系数。无冷启动，观察＝回看1–10秒，每日开盘0–59秒产生任务。原子表22,800行、38日；向前检验18日，十窗口共同任务{len(keys)}个，占计划1080任务的{len(keys)/1080:.1%}。','',
      f'训练内部选出的窗口依次为{used_windows}秒。该选择流程的未来3秒准确率：增强{e.accuracy:.2%}，同窗口三因子{b.accuracy:.2%}，差{100*(e.accuracy-b.accuracy):+.2f}个百分点。逐日改善{int(paired_adaptive.better_days)}/18日；五日块重采样95%差值区间[{100*paired_adaptive.ci_low:+.2f}, {100*paired_adaptive.ci_high:+.2f}]个百分点。','',
      '结论：这轮发现局部小幅改善，尚未找到稳定更优、值得增加等待时间的窗口。9秒窗相对自身三因子的提升最大，但绝对准确率仍未超过1–2秒三因子；它是本轮窗口比较中事后较好的结果，不能据此认定最优。最终拟合选8秒也只是一项待新数据检验的配置。','',
      '## 所有观察窗口：未来3秒','',
      '| 观察/回看 | 三因子 | 增强 | 提升百分点 | 有效任务 | 各折结构 |','|---:|---:|---:|---:|---:|---|']
    for r in main.to_dict('records'):lines.append(f"| {r['window']}秒 | {r['baseline_accuracy']:.2%} | {r['enhanced_accuracy']:.2%} | {100*r['gain']:+.2f} | {r['n']} | {r['structures']} |")
    lines+=['','## 训练期选窗口流程：一个信号的1–10秒效果','','| 未来期限 | 三因子累计 | 增强累计 | 提升百分点 | 三因子新增 | 增强新增 |','|---:|---:|---:|---:|---:|---:|']
    for h in range(1,11):
        bc=metric('adaptive_baseline',h);ec=metric('adaptive_selected',h);bi=metric('adaptive_baseline',h,'inc');ei=metric('adaptive_selected',h,'inc')
        lines.append(f'| {h}秒 | {bc.accuracy:.2%} | {ec.accuracy:.2%} | {100*(ec.accuracy-bc.accuracy):+.2f} | {bi.accuracy:.2%} | {ei.accuracy:.2%} |')
    lines+=['','## 训练选中窗口及新增成员','','| 检验折 | 窗口 | 新增成员 | 最大VIF |','|---:|---:|---|---:|']
    for decision in decisions:
        if decision['fold']=='final':continue
        r=combos[(combos.fold==decision['fold'])&(combos.window==decision['choice']['window'])&(combos.method=='selected')].iloc[0]
        lines.append(f"| {decision['fold']} | {r.window}秒 | {r.added_names} | {r.max_vif:.3f} |")
    lines+=['','## 下一批新行情的冻结配置','',
        f"同规则使用现有38日完成最终选择，观察窗口为{final['window']}秒，新增成员为"+'＋'.join(NAMES[x] for x in final['models']['selected']['features'] if x not in BASE)+'。配置、标准化和全部系数见下一批行情配置.json。该最终固定配置尚无新日期验证，不能把前述折间流程表现当成它自己的胜率。','',
        '## 样本、筛选与局限','',
        '- 模型统一预测未来3秒，每个任务只产生一个涨／平／跌信号。将此信号原样对照未来1–10秒，不按期限重训、不筛强信号。真实不变保留。',
        '- 主要按3秒日等权准确率选优；距最高不超过0.5个百分点再比较5–10秒平均准确率；后者仍接近时优先较短窗口、较少列、较低3秒Log loss、较强正则。相同窗口先选各类结构，再在四类赢家中选择，最后在十个窗口赢家中选择。',
        '- 相同窗口的所有结构采用相同训练、验证、检验任务；覆盖资格与候选池由内部训练确定，内部验证取十窗口共同任务。共线性先在内部训练检查，并在完整训练重拟合设计上复核，不使用检验日期。汇总主表也取十窗口共同任务。跨窗口的观察结束时点不同，不能将全部窗口差异解释为历史信息的作用。',
        '- 最大VIF≤5、两两|相关|<0.9，且复合项包含主效应。低共线性不等于独立信息；新增项被原三因子解释的R²记录在成员表。',
        '- 短窗原有80%跨度、90%特征覆盖、80%候选池共同覆盖门槛保留；不足列未填补。07-20、07-27仍排除，07-21原缺口仍按缺失处理。',
        '- 1–2秒窗口只有8列合格候选（含三因子），每折检查12种结构；3–10秒有31列，每折检查260种结构。五轮训练含最后全历史配置共检查10,520个折／窗候选配置，不是10,520种独立因子。其中5个配置通过内部训练共线性检查，但完整训练段复核超标，因而拒绝；拟合无数值失败。',
        '- 累计价变为P(t+h)−P(t)，新增价变为P(t+h)−P(t+h−1)。两者均以LastPrice计算；新增价变真实不变的比例往往更高，因此不能只直接比较两条准确率高低。flat_share列保留这些比例。',
        '- 全部38日此前已参与因子探索；1,080个任务有时间重叠，不是独立样本。日期块区间仅为探索性不确定性描述，不校正此前筛选，也不是交易收益。四折表现、逐日增益和原始预测均提供。',
        '- 不依据检验期最高一格倒改模型或规则。若提升、窗口或成员不稳定，应记为暂未找到稳定更优窗口。','',
        '配套文件：主对照表.csv、全部期限结果.csv、相对三因子增益.csv、四折表现.csv、各折组合成员.csv、新增组合选择频次.csv、逐日评价.csv、逐任务预测.parquet、模型参数.json、选择过程.json、窗口选择.json。']
    (OUT/'固定三因子增强报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(OUT/'汇总复核入口.json',dict(common_tasks=len(keys),scheduled_tasks=1080,summary_rows=len(s),daily_rows=len(d),
        windows_chosen=used_windows,final_window=final['window'],models=len(models)))
    print(main.to_string(index=False));print('Adaptive:',e.accuracy,b.accuracy,'windows',used_windows,'final',final['window'])

if __name__=='__main__':run()
