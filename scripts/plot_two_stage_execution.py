"""Static, exportable figure for the two-stage execution ablation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'result/opening_execution/two_stage_ablation'


def main():
    summary=pd.read_csv(OUT/'策略总表.csv')
    summary=summary[summary.side=='买卖各半'].sort_values('policy')
    clock=pd.read_csv(OUT/'按任务时刻分段节约.csv')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei'],
                         'axes.unicode_minus':False,'font.size':11})
    fig,ax=plt.subplots(1,2,figsize=(13,5.7),gridspec_kw={'width_ratios':[.85,1.35]})
    fig.suptitle('两阶段执行：更早成交，是否更省成本？',fontsize=19,y=.97,fontweight='bold')
    fig.text(.5,.905,'IC2609 · 18个检验日 · 1,062个共同任务 × 买卖双向 · 只在第3秒判断一次',ha='center',color='#536574')
    colors=['#8a98a5','#167d9b','#d87935']
    bars=ax[0].bar(np.arange(3),summary.cost_bp,color=colors,width=.62)
    for b,v in zip(bars,summary.cost_bp):
        ax[0].text(b.get_x()+b.get_width()/2,v+.045,f'{v:.3f}',ha='center',fontsize=13,fontweight='bold')
    ax[0].set_xticks([0,1,2],['A 基准\n限价等10秒','B 先观察\n3秒再下单','C 先挂单\n3秒再判断'])
    ax[0].set_ylabel('平均执行成本（bp，越低越好）')
    ax[0].set_ylim(0,2.3)
    ax[0].set_title('同一起点LastPrice衡量成本',pad=12)
    for name,label,color in [('B_vs_A_saving_bp','B 较基准节约','#167d9b'),
                              ('C_vs_A_saving_bp','C 较基准节约','#d87935')]:
        g=clock[clock.comparison==name].sort_values('clock_bin')
        x=np.arange(6)+(-.07 if name.startswith('B') else .07)
        ax[1].errorbar(x,g.mean_bp,yerr=np.vstack([g.mean_bp-g.low_bp,g.high_bp-g.mean_bp]),
                       marker='o',lw=1.8,capsize=3,color=color,label=label)
    ax[1].axhline(0,color='#526372',lw=1,ls='--')
    ax[1].set_xticks(np.arange(6),['01–09','10–19','20–29','30–39','40–49','50–59'])
    ax[1].set_xlabel('任务产生时刻：09:30之后的秒数')
    ax[1].set_ylabel('较基准节约（bp，正数更好）')
    ax[1].set_title('按任务时刻分段：各日等权',pad=12)
    ax[1].legend(frameon=False,loc='upper right',fontsize=10)
    for a in ax:
        a.spines[['top','right']].set_visible(False)
        a.grid(axis='y',alpha=.18)
        a.set_axisbelow(True)
    b=summary[summary.policy=='B_observe_first'].iloc[0]
    c=summary[summary.policy=='C_limit_first'].iloc[0]
    difference=c.cost_bp-b.cost_bp
    comparison=f'C 比 B 平均{"多花" if difference>=0 else "节约"} {abs(difference):.3f} bp'
    uncertainty='C 较基准的节约区间包含零。' if c.saving_ci_low<=0<=c.saving_ci_high else '区间与样本范围见报告。'
    fig.text(.07,.09,comparison+'；'+uncertainty,fontsize=12,fontweight='bold',color='#334957')
    fig.text(.07,.045,'误差线：5日日期块重采样95%区间。第0秒缺少当时行情，统一排除。L1撮合代理，不含手续费、排队与冲击。',fontsize=9,color='#687986')
    fig.subplots_adjust(left=.07,right=.98,top=.8,bottom=.25,wspace=.3)
    fig.savefig(OUT/'两阶段执行成本对比.png',dpi=180)
    fig.savefig(OUT/'两阶段执行成本对比.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()
