"""Describe frozen out-of-sample accuracy by actual signal clock; no refitting."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE=ROOT/'result/opening_prediction/three_factor_enhancement/共同任务预测.parquet'
OUT=SOURCE.parent/'time_of_minute'
BLUE='#3575B5';GREEN='#008B77'

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    p=pd.read_parquet(SOURCE);p=p[p.method.isin(['baseline','selected'])].copy()
    p['hit']=p[['p_down','p_flat','p_up']].to_numpy().argmax(1)-1==np.sign(p.cum_3.to_numpy())
    before=len(p);p=p[(p.decision_seconds>=0)&(p.decision_seconds<60)]
    assert p.cum_3.notna().all()
    assert not p.duplicated(['model','trade_date','decision_seconds']).any()
    daily=[];rolling=[];bins=[]
    rng=np.random.default_rng(20260923);starts=rng.integers(0,18,size=(4000,4))
    bootstrap=((starts[:,:,None]+np.arange(5))%18).reshape(4000,-1)[:,:18]
    for w in range(1,11):
        pair=p[p.window==w].pivot(index=['trade_date','decision_seconds'],columns='method',values='hit').reset_index()
        assert pair[['baseline','selected']].notna().all().all()
        dates=sorted(pair.trade_date.unique());assert len(dates)==18
        for e in range(w,60):
            exact=pair[pair.decision_seconds==e];assert len(exact)==18
            for method in ['baseline','selected']:
                daily.append(dict(window=w,second=e,method=method,accuracy=float(exact[method].mean()),n=len(exact),days=18))
            group=pair[pair.decision_seconds.between(max(w,e-9),e)]
            means=group.groupby('trade_date')[['baseline','selected']].mean().reindex(dates).astype(float)
            delta=means.selected.to_numpy()-means.baseline.to_numpy()
            ci=np.quantile(delta[bootstrap].mean(axis=1),[.025,.975])
            rolling.append(dict(window=w,second=e,start_second=max(w,e-9),end_second=e,
                baseline=float(means.baseline.mean()),selected=float(means.selected.mean()),
                gain=float(delta.mean()),gain_low=float(ci[0]),gain_high=float(ci[1]),n=len(group),days=18,
                full_ten_seconds=e-w>=9))
        for left in range(0,60,10):
            group=pair[(pair.decision_seconds>=left)&(pair.decision_seconds<left+10)]
            if group.empty:continue
            means=group.groupby('trade_date')[['baseline','selected']].mean().astype(float)
            bins.append(dict(window=w,start=left,end=left+10,baseline=float(means.baseline.mean()),
                selected=float(means.selected.mean()),gain=float((means.selected-means.baseline).mean()),n=len(group),days=len(means)))
    r=pd.DataFrame(rolling);b=pd.DataFrame(bins);s=pd.DataFrame(daily)
    r.to_csv(OUT/'按信号时刻_十秒滚动准确率.csv',index=False,encoding='utf-8-sig')
    b.to_csv(OUT/'按信号时刻_十秒分段准确率.csv',index=False,encoding='utf-8-sig')
    s.to_csv(OUT/'按信号时刻_逐秒准确率.csv',index=False,encoding='utf-8-sig')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#ADBBC5','text.color':'#263D4B',
        'axes.labelcolor':'#263D4B','xtick.color':'#526A79','ytick.color':'#526A79'})
    fig,(ax,delta_ax)=plt.subplots(2,1,figsize=(13,8),sharex=True,gridspec_kw={'height_ratios':[2.1,1]})
    fig.subplots_adjust(top=.84,bottom=.15,hspace=.2,left=.08,right=.96)
    fig.suptitle('开盘首分钟：准确率是否随信号时刻变化？',x=.08,y=.96,ha='left',fontsize=20,fontweight='bold')
    fig.text(.08,.898,'固定观察／回看 3 秒 · 预测观察结束后未来 3 秒 · LastPrice 涨／平／跌 · 18 个检验日',fontsize=11)
    q=r[r.window==3];complete=q[q.full_ten_seconds];partial=q[~q.full_ten_seconds]
    for method,color,label in [('baseline',BLUE,'原三因子'),('selected',GREEN,'三因子＋新增因子')]:
        raw=s[(s.window==3)&(s.method==method)]
        ax.scatter(raw.second,raw.accuracy*100,color=color,alpha=.16,s=13)
        ax.plot(complete.second,complete[method]*100,color=color,lw=2.5,label=label)
        leading=q[q.second<=12]
        ax.plot(leading.second,leading[method]*100,color=color,lw=2,ls='--')
    ax.axvspan(0,3,color='#DCE3E8',alpha=.55);ax.text(1.5,79,'尚未\n出信号',ha='center',va='top',fontsize=9)
    ax.set_ylim(20,95);ax.set_ylabel('分类准确率（%）');ax.legend(loc='upper right',frameon=False,ncol=2)
    ax.grid(axis='y',alpha=.18)
    ax.set_title('实线：最近 10 秒的信号；虚线：开盘初期不足 10 秒；淡点：逐秒原值',loc='left',fontsize=10,pad=10)
    delta_ax.fill_between(q.second,q.gain_low*100,q.gain_high*100,color=GREEN,alpha=.13,label='日期块重采样 95% 逐点区间')
    delta_ax.plot(q.second,q.gain*100,color=GREEN,lw=1.8);delta_ax.axhline(0,color='#8798A3',lw=1,ls='--')
    delta_ax.set_ylabel('增强 − 三因子\n（百分点）');delta_ax.grid(axis='y',alpha=.18)
    delta_ax.legend(loc='upper right',frameon=False,fontsize=9)
    delta_ax.set_xlim(0,60);delta_ax.set_xticks([0,10,20,30,40,50,60],['09:30:00','09:30:10','09:30:20','09:30:30','09:30:40','09:30:50','09:31:00'])
    delta_ax.set_xlabel('信号发出时刻（观察结束）')
    fig.text(.08,.065,'每个逐秒点 18 个任务；完整滚动段 180 个任务。真实不变保留；没有按时间段重新训练或筛选信号。',fontsize=10)
    fig.text(.08,.035,'相邻曲线点共享任务，阴影不是整条曲线的同时置信带；已有日期的探索结果，不代表实盘胜率。',fontsize=9,color='#607787')
    fig.savefig(OUT/'开盘首分钟_准确率时间曲线.png',dpi=170);fig.savefig(OUT/'开盘首分钟_准确率时间曲线.pdf');plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(14,14),sharex=True,sharey=True)
    for w,ax in zip(range(1,11),axes.flat):
        q=r[r.window==w]
        for method,color,label in [('baseline',BLUE,'三因子'),('selected',GREEN,'增强')]:
            ax.plot(q[q.full_ten_seconds].second,q[q.full_ten_seconds][method]*100,color=color,lw=1.8,label=label)
            leading=q[q.second<=w+9];ax.plot(leading.second,leading[method]*100,color=color,lw=1.5,ls='--')
        ax.axvspan(0,w,color='#DCE3E8',alpha=.4);ax.set_title(f'观察／回看 {w} 秒',loc='left',fontsize=11)
        ax.set_ylim(30,90);ax.set_xlim(0,60);ax.grid(alpha=.15);ax.set_xticks([0,15,30,45,60])
    axes[0,0].legend(frameon=False,ncol=2)
    fig.suptitle('不同观察窗口下的开盘时间曲线',fontsize=20,y=.985)
    fig.supxlabel('信号发出时刻：09:30 后第几秒',y=.035);fig.supylabel('未来 3 秒分类准确率（%）',x=.02)
    fig.text(.08,.012,'均为最近10秒内信号的日等权准确率；虚线表示历史不足10秒。各窗口使用原有模型，不按时段重新选择。',fontsize=10)
    fig.tight_layout(rect=[.035,.05,1,.96]);fig.savefig(OUT/'全部观察窗口_时间曲线.png',dpi=150);plt.close(fig)
    # Exact aggregation checks: the disjoint bins must reproduce the restricted sample.
    for w in range(1,11):
        sub=b[b.window==w]
        for method in ['baseline','selected']:
            actual=p[(p.window==w)&(p.method==method)].hit.mean()
            np.testing.assert_allclose(np.average(sub[method],weights=sub.n),actual,atol=1e-14)
        assert int(sub.n.sum())==18*(60-w)
    assert s[s.window==3].accuracy.between(.2,.95).all()
    assert r[['baseline','selected']].ge(.3).all().all() and r[['baseline','selected']].le(.9).all().all()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==digest
    proof=dict(source_sha256=digest,source_unchanged=True,model_refit=False,training_horizon=3,
        x_axis='decision_seconds',clock_interval='09:30:00 inclusive to 09:31:00 exclusive',
        rolling_seconds=10,partial_windows='dashed',dates=18,main_window=3,main_tasks=18*57,
        excluded_after_minute_prediction_rows=before-len(p),all_window_bin_aggregation_checks=20,
        confidence='4000 circular 5-date block resamples, pointwise, exploratory',
        summary_rows=dict(rolling=len(r),bins=len(b),per_second=len(s)))
    (OUT/'时间曲线复核.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 开盘首分钟准确率随时间变化','',
        '横轴为信号实际发出时刻（观察结束），不是任务开始时刻。主图固定观察＝回看3秒，消除不同等待时长混在一起的影响；全部窗口图为1–10秒敏感性对照。预测目标仍是观察结束后未来3秒LastPrice的涨／平／跌。','',
        '信号时刻限制为09:30:00≤t<09:31:00；未来标签仍允许越过09:31。3秒窗最早09:30:03出信号，共18日、1026个任务，因此与原表包含09:31后信号的1080任务口径不同。','',
        '实线为截至横轴时刻最近10个整秒发出的信号，先逐日计算准确率，再对18日等权；不足10秒画虚线。每个原始秒点18任务，完整滚动段180任务。曲线平滑只用于展示，不重新训练或筛选。','',
        '阴影为5日循环块重采样的逐点95%差值区间，非同时置信带，也未校正看过整条曲线之后的时段挑选。','',
        '| 信号时段 | 原三因子 | 增强 | 任务数 |','|---|---:|---:|---:|']
    for row in b[b.window==3].to_dict('records'):
        end='09:31:00' if row['end']==60 else f"09:30:{row['end']:02}"
        lines.append(f"| [09:30:{row['start']:02}, {end}) | {row['baseline']:.2%} | {row['selected']:.2%} | {row['n']} |")
    lines+=['','曲线只能描述现有预测在这些时段的差异，不能单凭高低确定开盘时间导致准确率变化，或直接据此选择实盘时段。因子的单因子IC与模型分类准确率是不同指标，本次不重新计算时段IC。']
    (OUT/'时间曲线说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(b[b.window==3].to_string(index=False));print(json.dumps(proof,ensure_ascii=False))

if __name__=='__main__':run()
