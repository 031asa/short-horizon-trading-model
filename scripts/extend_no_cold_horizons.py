"""Evaluate frozen three-second signals at all future horizons; never refit."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.opening_ic import SessionData,ICConfig
from utils.opening_schedule import ScheduleConfig
from scripts.audit_opening_data import RAW,EXPECTED_SHA256
from scripts.run_opening_prediction import save_json
OUT=ROOT/'result/opening_prediction/no_cold_start'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run():
    inputs={f:sha(OUT/f) for f in ['逐任务预测.parquet','模型参数.json','训练期选择的窗口.json','无冷启动原子表.parquet','看板数据.json']}
    assert sha(RAW)==EXPECTED_SHA256
    config=dict(horizons=list(range(1,31)),model_training_horizon=3,refit=False,
       origin='observation end',price='LastPrice',label_types=['cumulative','incremental'],
       prediction='unchanged argmax of saved three-class probabilities',confidence='unchanged validation-frozen thresholds from the 3s experiment',
       samples=['per_horizon','common_30'],raw_sha256=EXPECTED_SHA256,input_sha256=inputs)
    freeze=OUT/'完整期限口径.json'
    if freeze.exists():assert json.loads(freeze.read_text(encoding='utf-8'))==config
    else:save_json(freeze,config)
    a=pd.read_parquet(OUT/'无冷启动原子表.parquet');pred=pd.read_parquet(OUT/'逐任务预测.parquet')
    cfg=ICConfig(schedule=ScheduleConfig(cold_start_seconds=0,observation_seconds=(1.,2.,3.,4.,5.,8.,10.)),
                 horizons_seconds=tuple(range(1,31)))
    path=OUT/'未来1至30秒标签.parquet'
    label_source=sha(ROOT/'utils/opening_ic.py')
    labelproof=OUT/'完整期限标签复核.json'
    if path.exists():
        proof=json.loads(labelproof.read_text(encoding='utf-8'));assert proof['source_sha256']==label_source and proof['label_sha256']==sha(path)
        labels=pd.read_parquet(path)
    else:
        raw=pd.read_parquet(RAW);rows=[];checks=0
        for date,g in raw.groupby(raw.Date.astype(str),sort=True):
            if date not in set(a.trade_date):continue
            d=SessionData(g,pd.Timestamp(date+' 09:30',tz='Asia/Shanghai'),cfg)
            for end in range(1,70):
                lab=d.labels(end);r=dict(trade_date=date,decision_seconds=end)
                for h in range(1,31):
                    for kind,key in [('cum',f'y_decision_{h}s'),('inc',f'y_decision_incremental_{h}s')]:
                        r[f'{kind}_{h}']=lab[key];r[f'{kind}_{h}_status']=lab[key+'_status']
                    # Independent endpoint lookup plus path-validity check.
                    i=np.searchsorted(d.times,end,side='right')-1;j=np.searchsorted(d.times,end+h,side='left')
                    valid=(i>=0 and j<d.n and d.valid['price'][i] and end-d.times[i]<=.5+1e-10
                      and d.times[j]-(end+h)<=.5+1e-10 and d.valid['price'][i:j+1].all() and d.edge['price'][i+1:j+1].all())
                    expected=d.p[j]-d.p[i] if valid else np.nan
                    np.testing.assert_allclose(r[f'cum_{h}'],expected,atol=0,rtol=0,equal_nan=True);checks+=1
                    if h>1:
                        valid_inc=np.isfinite(r[f'cum_{h}']) and np.isfinite(r[f'cum_{h-1}'])
                        expected=r[f'cum_{h}']-r[f'cum_{h-1}'] if valid_inc else np.nan
                        np.testing.assert_allclose(r[f'inc_{h}'],expected,atol=0,rtol=0,equal_nan=True)
                rows.append(r)
        labels=a[['trade_date','task_elapsed_seconds','window','decision_seconds']].merge(pd.DataFrame(rows),on=['trade_date','decision_seconds'],validate='many_to_one')
        np.testing.assert_allclose(labels.cum_3,a.y3,atol=0,rtol=0,equal_nan=True)
        labels.to_parquet(path,index=False)
        save_json(labelproof,dict(rows=len(labels),source_sha256=label_source,label_sha256=sha(path),independent_endpoint_checks=checks,
                                 cumulative_incremental_relation=True,three_second_labels_unchanged=True))
    keys=['trade_date','task_elapsed_seconds','window'];joined=pred.merge(labels.drop(columns='decision_seconds'),on=keys,validate='many_to_one')
    joined['model_id']=np.where(joined.method.str.startswith('adaptive'),joined.method,joined.method+'_w'+joined.window.astype(str))
    summary=[];daily=[];common_sizes={}
    for kind in ['cum','inc']:
        columns=[f'{kind}_{h}' for h in range(1,31)]
        regular=joined[joined.method.isin(['selected','baseline'])]
        good=regular[np.isfinite(regular[columns]).all(axis=1)]
        counts=good.groupby(['trade_date','task_elapsed_seconds']).size();common=counts[counts==14].reset_index()[['trade_date','task_elapsed_seconds']]
        common_sizes[kind]=len(common)
        for sample,cohort in [('per_horizon',joined),('common_30',joined.merge(common,on=['trade_date','task_elapsed_seconds'],validate='many_to_one'))]:
            for model,g in cohort.groupby('model_id',sort=False):
                g=g.sort_values(['trade_date','task_elapsed_seconds']);scores=g[['p_down','p_flat','p_up']].to_numpy()
                classes=scores.argmax(1)-1;conf=np.maximum(scores[:,0],scores[:,2]);date=g.trade_date.to_numpy()
                outcomes=g[columns].to_numpy();finite=np.isfinite(outcomes);hit=classes[:,None]==np.sign(outcomes)
                for level in ['all','0.2','0.4','0.6']:
                    signal=np.ones(len(g),bool) if level=='all' else conf>=g['threshold_'+level].to_numpy()
                    for segment,mask in [('all',np.ones(len(g),bool)),('opening_0_9',g.task_elapsed_seconds.to_numpy()<10),('later_10_59',g.task_elapsed_seconds.to_numpy()>=10)]:
                        all_acc=[];all_counts=[];all_flat=[]
                        for day in sorted(set(date)):
                            take=(date==day)&signal&mask;valid=finite[take];n=valid.sum(axis=0)
                            acc=np.divide((hit[take]&valid).sum(axis=0),n,out=np.full(30,np.nan),where=n>0)
                            flat=np.divide(((outcomes[take]==0)&valid).sum(axis=0),n,out=np.full(30,np.nan),where=n>0)
                            all_acc.append(acc);all_counts.append(n);all_flat.append(flat)
                            for u in range(30):daily.append(dict(kind=kind,sample=sample,model=model,signal=level,segment=segment,trade_date=day,horizon=u+1,n=int(n[u]),accuracy=acc[u]))
                        aa=np.array(all_acc);nn=np.array(all_counts);ff=np.array(all_flat)
                        nd=np.isfinite(aa).sum(axis=0)
                        mean=np.divide(np.nansum(aa,axis=0),nd,out=np.full(30,np.nan),where=nd>0)
                        fs=np.divide(np.nansum(ff,axis=0),nd,out=np.full(30,np.nan),where=nd>0)
                        count=nn.sum(axis=0);scheduled=18*({'all':60,'opening_0_9':10,'later_10_59':50}[segment])
                        for u in range(30):summary.append(dict(kind=kind,sample=sample,model=model,signal=level,segment=segment,horizon=u+1,
                            accuracy=mean[u],n=int(count[u]),days=int(nd[u]),scheduled_coverage=count[u]/scheduled,flat_share=fs[u]))
    s=pd.DataFrame(summary);pd.DataFrame(daily).to_parquet(OUT/'完整期限逐日明细.parquet',index=False)
    s.to_csv(OUT/'未来1至30秒准确率.csv',index=False,encoding='utf-8-sig')
    original=pd.read_csv(OUT/'模型汇总.csv');checked=0
    for r in original[original['sample']=='own'].to_dict('records'):
        row=s[(s.kind=='cum')&(s['sample']=='per_horizon')&(s.model==r['model'])&(s.signal=='all')&(s.segment==r['segment'])&(s.horizon==3)].iloc[0]
        np.testing.assert_allclose(row.accuracy,r['accuracy'],atol=1e-14);assert row.n==r['n'];checked+=1
    confidence=pd.read_csv(OUT/'命中率与覆盖.csv');threshold_checks=0
    for r in confidence[(confidence['sample']=='own')&(confidence['mode']=='frozen_threshold')].to_dict('records'):
        row=s[(s.kind=='cum')&(s['sample']=='per_horizon')&(s.model==r['model'])&(s.signal==str(r['level']))&(s.segment=='all')&(s.horizon==3)].iloc[0]
        np.testing.assert_allclose(row.accuracy,r['accuracy'],atol=1e-14)
        assert row.n==r['signals'] and row.days==r['days'];threshold_checks+=1
    assert all(sha(OUT/f)==v for f,v in inputs.items())
    save_json(OUT/'期限看板数据.json',dict(config=config,summary=summary,common_tasks=common_sizes))
    save_json(OUT/'完整期限复核.json',dict(model_and_predictions_unchanged=True,original_three_second_summary_checks=checked,
       original_frozen_threshold_checks=threshold_checks,
       summary_rows=len(s),common_tasks=common_sizes,input_sha256=inputs,
       source_sha256=sha(Path(__file__)),labels_rows=len(labels)))
    lines=['# 完整预测期限：冻结3秒模型信号的1–30秒检验','',
      '原表的所有准确率都针对观察结束后的未来3秒；左侧1／2／3／4／5／8／10秒为观察时长。', '',
      '本次保持已选组合、参数、观察期选择和预测信号全部不变，只扩展未来标签，逐秒查看同一信号对应的实际方向。不代表对每个预测期限重新选组合、训练或校准概率。', '',
      '累计响应：LastPrice(t+u)−LastPrice(t)。新增响应：LastPrice(t+u)−LastPrice(t+u−1)。t为观察结束时刻，u为未来期限。所有真实不变都保留并按原三分类预测判断对错。', '',
      '“逐期限有效”针对每个期限单独使用可评价任务；“30期限共同任务”要求七个窗口的三因子和小组合、全部30期限均可评价。后者便于对比曲线但可能减少覆盖。', '',
      '信号可选全量或本轮原有训练期冻结阈值。阈值是3秒模型输出概率的门槛，不代表其他期限有相同概率；没有用各期限未来结果重新挑选信号。准确率按日期等权，逐日明细另存。', '',
      f'旧3秒的{checked}个模型／任务范围汇总逐值复核一致。模型、预测、原子表、原汇总指纹均不变。新增标签15960行，全部端点和累计／新增关系核对。','',
      '详细准确率在未来1至30秒准确率.csv；完整期限逐日明细.parquet 保存每天样本数与命中率。当前数据仍属于已有日期的探索，期限曲线不能当作交易收益曲线。']
    (OUT/'完整期限说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(rows=len(s),three_second_checks=checked,common_tasks=common_sizes),ensure_ascii=False))

if __name__=='__main__':run()
