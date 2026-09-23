from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'result/opening_prediction/no_cold_start'

def run():
    d=json.loads((OUT/'看板数据.json').read_text(encoding='utf-8'))
    payload=json.dumps(d,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    (OUT/'无冷启动_3秒小组合.html').write_text((ROOT/'utils/no_cold_dashboard.html').read_text(encoding='utf-8').replace('__DATA__',payload),encoding='utf-8')
    def row(model,segment='all'):return next(r for r in d['summary'] if r['sample']=='common' and r['segment']==segment and r['model']==model)
    s=row('selected_w3');b=row('baseline_w3');early=row('selected_w3','opening_0_9');eb=row('baseline_w3','opening_0_9')
    adaptive=row('adaptive_selected');ab=row('adaptive_baseline')
    lines=['# 无冷启动：未来3秒小组合实验','','## 本轮发现','',
      f"本轮没有找到整分钟稳定胜过三因子的优秀小组合。观察3秒的小组合准确率为{s['accuracy']:.1%}，三因子为{b['accuracy']:.1%}；两者Log loss分别为{s['logloss']:.4f}、{b['logloss']:.4f}。观察8／10秒也没有改善。",'',
      f"开盘0–9秒产生的任务，小组合在观察3秒时准确率为{early['accuracy']:.1%}，三因子为{eb['accuracy']:.1%}，共{early['n']}个检验任务。这是预设分段的描述结果，不可据此宣称整分钟有效或直接据此调整筛选目标。",'',
      f"四个训练折选定的观察期依次为{'／'.join(str(x['window']) for x in d['adaptive'])}秒。该选择流程的检验准确率为{adaptive['accuracy']:.1%}，对应三因子为{ab['accuracy']:.1%}，没有稳定超越。",'',
      '## 新口径','',
      '- 冷启动为0。开盘第0–59秒每秒产生任务，观察＝回看1／2／3／4／5／8／10秒，共38日、15,960行。任务可以在第一分钟内产生、在其后结束观察。',
      '- 每个因子的所有输入必须位于本任务[T,T+W]，不使用任务之前的盘口、价格、成交增量或派生值。OFI和成交增量的前后两个快照均在区间内；首条累计成交量仅作为差分起点。',
      '- 标签固定为观察结束后的未来3秒LastPrice价变符号，真实不变保留。沿用原始tick、as-of、未来端点对齐、陈旧和缺口口径，未来标签可越过开盘首分钟。',
      '- 07-20／07-27整段开盘缺行情仍排除；07-21的既有缺口仍留空。没有恢复缺失行情或借用此前快照。',
      '- 原9500行原子表和原预测看板保留，新实验另存，不能直接把两轮准确率差当成模型提升：任务范围和历史边界已经变化。','',
      '## 组合筛选','',
      '- 25个预定义主特征，涵盖瞬时盘口、持续压力、成交确认、价格路径、状态；6个预定义交互。状态与路径因子不强行赋予正向逻辑。',
      '- 使用L2三分类逻辑回归，C∈{0.1,1,10}。有界beam搜索宽度2、最多5列，另种入弱特征成组及交互组合，避免只从最强单因子开始。不是穷举全空间，也不是Elastic Net或神经网络。',
      '- 每折训练期末5日作内部验证；因子覆盖、常数列、共线性、标准化均由更早训练段决定。组合训练VIF≤5、任意两列相关绝对值<0.9；交互需同时包含两个父因子。',
      '- 训练期候选单列有效覆盖需≥90%，公共候选样本覆盖≥80%；组合在同一验证样本上比较。损失距最优≤0.002时优先少列；再在各窗口共同验证任务上比较，距最优≤0.002时优先短窗口。',
      '- 前20／25／30／35日训练，对后5／5／5／3日检验。窗口、组合、参数及信号阈值不由检验结果决定。全量候选已在过去研究中被看过，因此仍是内部探索，不是独立新样本验证。',
      f"- 折与窗口合计检查{sum(r['evaluated'] for r in d['selected'])}个组合配置（含共线性拒绝），每种方法18个检验日。七窗口共同任务为{d['common_test_tasks']}，与本轮预期1080一致。",'',
      '## 所有观察期对照（日期等权，全任务）','','| 观察／回看 | 小组合准确率 | 三因子准确率 | 小组合Log loss | 三因子Log loss |','|---:|---:|---:|---:|---:|']
    for w in d['rules']['windows']:
        x=row('selected_w'+str(w));y=row('baseline_w'+str(w));lines.append(f"| {w}s | {x['accuracy']:.1%} | {y['accuracy']:.1%} | {x['logloss']:.4f} | {y['logloss']:.4f} |")
    lines+=['','## 观察3秒的实际组合','','这些是各折独立选出的组合，不是事后把四组拼成一个固定模型。','','| 折 | 小组合 | 训练最大VIF |','|---:|---|---:|']
    for r in d['selected']:
        if r['window']==3:lines.append(f"| {r['fold']} | {'＋'.join(d['names'][f] for f in r['features'])} | {r['vif']:.3f} |")
    lines+=['','## 观察3秒的命中率与覆盖','','| 方法 | 目标覆盖 | 实际覆盖 | 命中率（日等权） | 信号数 |','|---|---:|---:|---:|---:|']
    for r in d['confidence']:
        if r['sample']=='common' and r['model']=='selected_w3':
            name='每天固定排名覆盖（离线）' if r['mode']=='fixed_rank' else '训练期冻结阈值'
            lines.append(f"| {name} | {r['level']:.0%} | {r['coverage']:.1%} | {r['accuracy']:.1%} | {r['signals']} |")
    lines+=['','按当天分数取固定比例仅用于同覆盖离线比较，不能直接当作实时门槛。冻结阈值只用内部验证概率确定，检验期实际覆盖允许变化。真实不变算方向信号错误。','',
      '## 覆盖与限制','',
      '严格限制任务内历史后，1秒窗口在部分日期只有两快照、实际跨度0.5秒；2秒窗口可能只有1.5秒跨度，分别低于原80%门槛，不能把它们伪装成完整1／2秒滚动统计。最初训练段滚动特征覆盖约58%，未过90%候选门槛，故1–2秒候选池只有8列瞬时／状态及交互；3秒起一般有31列候选。未放宽门槛或补用任务前历史。','',
      '所有模型的默认不变召回仍为零。VIF限制仅约束训练期线性冗余，不等于因子统计独立或独立有效。各折成员变化较大，进一步固定一个组合仍需新的日期检验。当前不能按本轮检验结果反复换门槛、换因子直到胜率好看。','',
      '首10秒任务的62.8%值得保留为后续研究线索，但不能用180个有重叠的任务宣称已经获得可靠高胜率。下一步应先明确是否只研究开盘最初几次决策；本轮未据此重新训练或筛选。','',
      '## 文件与复核','',
      '无冷启动原子表.parquet 保存新任务、特征和标签；选择过程.json 保存各折候选、剔除原因和最佳候选；模型参数.json 保存可重放系数。命中率与覆盖.csv、逐日评价.csv、训练期选择的窗口.json、独立复核.json 和网页验收.json用于审阅。原始文件与旧原子表指纹保持一致。']
    (OUT/'无冷启动_3秒小组合报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':run()
