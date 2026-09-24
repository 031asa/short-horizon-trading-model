"""Static fixed T+3 offset cost/time/fallback curves and paired result tables."""
from scripts.run_signal_offset_sweep import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zipfile,hashlib
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'svg.fonttype':'path','font.size':11,'axes.spines.top':False,'axes.spines.right':False})


def main():
    lines=['# 初始19档固定：只扫描第3秒限价档位','',
        '初始买入LastPrice−3.8、卖出LastPrice+3.8。第3秒不利信号仍转市价，其余信号按当时LastPrice偏移k档；10秒提交市价兜底，保留两条快照延迟和在途旧单优先成交。k=0是新C，k=19是旧C。模型不重训。',
        '预先固定扫描0–30整数档及40、60档，主比较首小时逐秒全部任务，日期等权、买卖各半；8组完整结果及分买卖结果另存。各档位及参考策略严格使用共同任务；更远档位引起的路径缺口排除按任务同时作用于全部方案。',
        '18日沿用原四折冻结模型，40日覆盖版沿用第四折模型，首小时39日有效，含训练内回算。最低点仅为已看过日期的描述性结果。5日循环日期块5000次区间未经档位选择校正；不能据最低点认定未来最优。限价按限价结算，无排队、冲击、费用及部分成交模拟。','',
        '|数据|第3秒档位|成本bp|比旧C节约bp|配对95%区间|比立即市价节约bp|完成秒|10秒兜底比例|',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    keyrows=[];coverage=[]
    for scope in SOURCES:
        folder=DEST/scope;s=pd.read_csv(folder/'全部档位汇总.csv')
        a=s[s.minutes.eq(60)&s.grid.eq('second')&s.side.eq('both')].set_index('k')
        g=a[a.index>=0].sort_index();best=int(g.cost_bp.idxmin());x=g.index.to_numpy()
        receipt=json.loads((folder/'验收与最低点.json').read_text(encoding='utf-8'))
        before=sum(r['tasks_before'] for r in receipt['checks']);after=sum(r['tasks_after'] for r in receipt['checks'])
        coverage.append(dict(scope=scope,tasks_before=before,tasks_after=after,lost=before-after,independent_engine_checks=sum(r['event_engine_checks'] for r in receipt['checks'])))
        daily=pd.read_csv(folder/'逐日结果.csv');p=daily[daily.minutes.eq(60)&daily.grid.eq('second')&daily.side.eq('both')].pivot(index='trade_date',columns='k',values='cost_bp')
        wins=(p[19]-p[best]);pd.DataFrame({'saving_vs_old_C':wins}).to_csv(folder/'最低点逐日节约.csv',encoding='utf-8-sig')
        selected=sorted(set([0,14,15,19,best,30,40,60]))
        for k in selected:
            r=a.loc[k];keyrows.append(dict(k=k,**r.to_dict()))
            lines.append(f'|{scope}|{k}|{r.cost_bp:.6f}|{r.saving_vs_old_C:+.6f}|[{r.low_vs_old_C:+.6f},{r.high_vs_old_C:+.6f}]|{r.saving_vs_M:+.6f}|{r.elapsed_seconds:.3f}|{r.deadline_market:.2%}|')
        fig,axes=plt.subplots(2,2,figsize=(15,10))
        ax=axes[0,0];ax.plot(x,g.cost_bp,'o-',ms=3,color='#376fa0',label='初始19档，第3秒k档')
        ax.axhline(a.loc[-1,'cost_bp'],color='#777777',ls='--',label='M：任务开始立即市价')
        for k,color in [(0,'#D38A28'),(19,'#8358AC'),(best,'#198573')]:
            ax.scatter([k],[g.loc[k,'cost_bp']],s=60,color=color,zorder=5)
        ax.annotate(f'本批最低：{best}档\n{g.loc[best,"cost_bp"]:.4f} bp',xy=(best,g.loc[best,'cost_bp']),xytext=(12,35),textcoords='offset points',arrowprops={'arrowstyle':'->','color':'#198573'})
        ax.set_ylabel('平均执行成本（bp，越低越好）');ax.legend(fontsize=9)
        ax=axes[0,1];ax.plot(x,g.saving_vs_old_C,color='#198573',label='相对旧C（第3秒19档）的节约')
        ax.fill_between(x,g.low_vs_old_C,g.high_vs_old_C,color='#198573',alpha=.18,label='逐点95%日期块区间')
        ax.axhline(0,color='#777777',lw=.8);ax.set_ylabel('节约（bp，越高越好）');ax.legend(fontsize=9)
        axes[1,0].plot(x,g.elapsed_seconds,'o-',ms=3,color='#D38A28');axes[1,0].set_ylabel('从任务起点到成交（秒）')
        axes[1,1].plot(x,100*g.deadline_market,'o-',ms=3,color='#8358AC');axes[1,1].set_ylabel('10秒兜底市价成交比例（%）')
        for ax in axes.flat:
            ax.set_xlabel('第3秒限价向被动方向偏移（档）');ax.grid(alpha=.18)
            ax.axvline(19,color='#8358AC',ls=':',alpha=.5)
        label='18日' if scope=='18d' else '40日覆盖（39个有效日）'
        fig.suptitle(f'{label}｜初始固定19档，只调整第3秒限价\n首小时每秒任务 · {after:,} 个共同任务 · 日期等权、买卖各半',fontsize=16)
        fig.text(.04,.02,'k=0：新C；k=19：旧C。只改变限价分支，不利信号仍转市价。全部任务含提前成交；完成时间包含延迟。\n范围0–30、40、60档，远端连线仅连接已计算点。区间未校正选档；40日含训练内回算，最低点不代表样本外最优。',fontsize=10)
        fig.tight_layout(rect=[0,.07,1,.92])
        for ext in ['png','svg']:fig.savefig(folder/f'固定初始19档_第3秒档位扫描.{ext}',dpi=190)
        plt.close(fig)
    pd.DataFrame(keyrows).to_csv(DEST/'重点档位对照.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(coverage).to_csv(DEST/'共同样本与验收.csv',index=False,encoding='utf-8-sig')
    lines+=['','## 覆盖与验收']+[f"- {r['scope']}：原{r['tasks_before']}任务，共同保留{r['tasks_after']}，排除{r['lost']}；完整事件撮合交叉验证{r['independent_engine_checks']}笔。所有保留任务0档/19档逐笔复现旧结果。" for r in coverage]
    lines+=['','独立向量扫描仅用于完整有效行情路径，其余路径调用原事件撮合。每日期按固定间隔抽样全部档位，用原事件撮合及独立最早成交扫描交叉核对；提交价格核对初始19档、3秒k档。保存每日期全部档位逐单价格、时间、成本和成交方式，原行情、信号、模型及基准文件校验值保持不变。']
    (DEST/'结果说明.md').write_text('\n'.join(lines),encoding='utf-8')
    archive=ROOT/'result/初始19档_第3秒固定档位扫描.zip'
    files=[p for p in DEST.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(DEST).parts)]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(archive) as z:
        for p in files:assert hashlib.sha256(z.read(p.relative_to(DEST).as_posix())).hexdigest()==sha(p)
    print(pd.DataFrame(keyrows)[['scope','k','cost_bp','saving_vs_old_C','saving_vs_M','elapsed_seconds','deadline_market']].to_string(index=False),flush=True)
    print('Archive verified',len(files),'files',flush=True)


if __name__=='__main__':main()
