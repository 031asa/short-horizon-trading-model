# 开盘首分钟单因子 IC 研究

当前阶段：先检验因子与未来 LastPrice 涨跌的关系，暂停模型训练和执行策略。最新用户调整已落实：观察期为 **1、2、3、4、5 秒**，没有 0 秒版本。当前规则见 `撮合规则与研究说明.md`；历史决策和附件阅读记录见 `项目口径核对.md`。

## 本轮参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| 开盘 | AM 09:30，Asia/Shanghai | 本轮只分析上午 |
| cold_start_seconds | 10，可调 | 开盘起积累历史，此前不产生任务 |
| task_step_seconds | 1 | 每天第 10–59 秒各产生一次任务 |
| observation_seconds | 1、2、3、4、5 | 任务产生后再观察多久，然后计算信号 |
| history_seconds | 5、10 | 窗口型因子回看长度，可含任务前已积累的历史 |
| horizons_seconds | 1–30，逐秒 | 信号形成后，分别检验的未来期限 |
| execution_delay_snapshots | 2 | 延迟对照从观察结束后第二张快照开始；主 IC 从观察结束开始 |
| tick_size | 0.2 | 标签以 LastPrice 的 tick 变化计 |

观察期与因子历史窗口分开；延长观察期改变信号时刻，不自动增加因子列数。本轮完整实现 PDF 的 59 个家族，计算输出、别名与质量字段分别登记。不同任务若在同一时刻结束观察，应得到完全相同的因子。第一分钟限制任务产生；最后任务可以在第 64 秒完成观察，之后继续取得标签。

数据缺口超过 1 秒则断开对应历史片段；历史覆盖要求 80%，当前快照陈旧及未来对齐误差各不超过 0.5 秒。缺失、常数列和有效零值分开。逐日 Pearson IC 与 Spearman Rank IC 至少要求 20 对；逐日算完后对有效日期等权平均。此门槛不是显著性保证。

## 运行与环境

用户已明确本项目无需 WSL，本次实际使用 **Windows 原生 Python 3.12**。`environment.yml` 记录直接依赖，可用于创建等价 Conda 环境；本次没有运行 WSL 或创建 Conda 环境。

当前机器使用 Codex 附带 Python，额外的 PyArrow 和绘图依赖位于项目 `.cache/python-packages`，入口脚本会自动加载。命令在项目根目录执行：

```powershell
$projectPython = 'C:/Users/Hello/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $projectPython -X utf8 -m unittest discover -s tests -p 'test_opening*.py' -v
& $projectPython -X utf8 scripts/run_opening_ic.py --cold-start 10 --observations 1 2 3 4 5 --publish
```

其他机器在安装 `environment.yml` 中的依赖后，可用对应 Python 运行相同入口。移除 `--publish` 可生成候选表和结果而不替换当前原子表。参数来源写入 `experiment_manifest.json`，不要混合不同次运行的文件。

Notebook 入口为 `行情转原子执行表.ipynb`，调用同一套代码，不再维护独立的旧撮合函数。

## 输入、输出与历史边界

- 原始行情：`新窗口交接_20260921/20260720_20260911_IC2609.parquet`，保持原样。SHA256 校验写入代码和结果清单。源文件按交接路径保留，没有另复制到 data。
- 当前原子表：`sample_snapshot_原子执行总表.parquet`。按用户要求复用此文件名，**新表结构为任务＋因子＋未来标签**，不再包含旧 k/H 执行收益。元数据键为 `atomic_ic`，规则为 `opening-task-observation-ic-full-v2`。历史脚本不可直接按旧表结构读取它。
- 结果目录：`result/opening_execution/`。含 `全部因子IC与衰减.html`、`IC研究结果.md`、`feature_registry.csv`、`factor_hypotheses.csv`、`factor_coverage.csv`、`family_coverage.csv`、`daily_ic.parquet`、`summary_ic.csv`、`summary_ic.parquet`、`experiment_manifest.json` 和原始数据审计文件。是否已经替换当前表，以清单的 `published` 为准。
- 当前源码：`utils/opening_schedule.py`、`utils/opening_ic.py`、`utils/opening_features.py`、`utils/factor_catalog.py`、`utils/ic_statistics.py`、`scripts/run_opening_ic.py`、`scripts/render_opening_ic.py`。
- 当前测试：`tests/test_opening*.py`。全目录还有旧研究测试，它们依赖交接包内模块，不属于本轮测试入口；直接发现全部旧测试会因缺少 `rebuild_delay3_bp` 导入而失败。
- 旧研究资料、交接包和原始行情未改写；`新窗口交接_20260921/复现参考/` 留存旧执行逻辑，不能与本轮 IC 混用。

当前取消开发／验证划分，38 个有效上午开盘全部进入研究，汇总统一 `phase=all_sample`。当前原子表为 38 × 50 × 5 = 9,500 行。全部日期已在旧项目探索过，不能称为未见测试集。

IC 不是预测正确率或交易收益。重叠标签、同一时刻的重复观察方案不能当独立试验累加。负 IC 保留原符号；幅度/状态因子同时评价绝对价变。暂不根据本轮排行榜选择模型、执行策略或事后最佳期限。

整段无行情日期单列在 `result/opening_execution/excluded_openings.csv`；部分缺口及处理记录在 `数据异常说明.md`。2026-07-21 首分钟存在 10.5 秒相邻行情间隔，保留该日但按连续性规则使受影响数据缺失。后续发现影响口径或结果的异常，应向用户说明日期、影响和处理，不静默改数或删日。

## 全量 IC 与衰减网页

直接用新版 Edge / Chrome 打开 `result/opening_execution/全部因子IC与衰减.html`，无需部署或网络。默认按家族显示全部输出，支持历史窗口、观察期、时间基准、评价目标、样本口径、累计／新增响应与 Pearson／Spearman 切换。点击因子查看逐日分布摘要、覆盖与缺失原因。网页只包含汇总；完整逐日明细使用 Parquet。

共同样本对每个因子独立要求跨全部 5 个观察期及 30 个未来期限有效。累计与新增响应分别求共同样本，原始负 IC 不翻转。逻辑预期先登记，属于已查看部分期货结果之后的研究假设；本轮不按 IC 调参数、不拟合固定半衰期、不进行显著性或收益判断。全部参数与覆盖原因见结果报告。

`--features-only` 可单独重建并验证候选原子表；`--reuse-features` 会校验原始行情、配置、特征源码与候选表指纹后复用，避免重复计算。未加 `--publish` 的产物统一留在结果目录的 `.pending` 中，当前发布结果保持原样。

验收记录：本轮 40 项单元测试通过；另完成真实数据的前缀截断、独立标签选点、原有 30 列与旧 IC 兼容比较、累计／增量恒等式、R06 窗口恒等式及新增因子的独立逐日 IC 抽查。完整计数见结果清单。浏览器使用离线的本地 file 页面验收，未发起外部请求。

可用 `scripts/verify_opening_artifacts.py result/opening_execution` 复核发布数据；`tests/opening_dashboard.cjs result/opening_execution` 是额外的网页验收入口，需要本机 Edge 与 Playwright。默认研究运行不依赖浏览器。`--reuse-evaluation` 仅在候选目录仍保留完整 IC 时使用，须逐项匹配计算输入、统计方法和结果指纹。

网页“预期与 IC”可筛选逻辑正向且原始平均日 IC > 0，期限可选择任一期、全部 30 期或指定秒数。`全部因子IC与衰减.html#positive` 直接进入计算因子的正向绿色筛选；默认研究口径下为 31 个至少一期为正、4 个全部期限为正。筛选随当前观察期、目标、样本、时间基准、响应及相关系数同步变化；完整曲线保留原符号。`逻辑正向且绿色IC_筛选清单.csv` 为默认口径导出。

看板排列更新：按当前评价目标的逻辑预期“正向 → 负向 → 不确定 → 不适用”分组，组内保持 F／A／B／C／D／M／R 家族编号顺序。筛选区下方显示随选项变化的 IC 公式，包含观察时点、累计／新增与有方向／绝对标签、配对或共同样本、Pearson／Spearman 及日期等权汇总；质量视图说明不计算 IC。
