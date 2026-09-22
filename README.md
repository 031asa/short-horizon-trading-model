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
& $projectPython -X utf8 scripts/test_opening.py result/opening_execution/.review_pending
& $projectPython -X utf8 scripts/run_opening_ic.py --cold-start 10 --observations 1 2 3 4 5 --publish
```

其他机器在安装 `environment.yml` 中的依赖后，可用对应 Python 运行相同入口。移除 `--publish` 可生成候选表和结果而不替换当前原子表。参数来源写入 `experiment_manifest.json`，不要混合不同次运行的文件。

Notebook 入口为 `行情转原子执行表.ipynb`，调用同一套基础实验代码，不再维护独立的旧撮合函数。上面的研究运行入口重建 PDF 基础实验；最新的方向机制审阅还需执行下述审阅步骤。

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

## 2026-09-22 方向机制审阅（当前展示版本）

用户授权更激进地明确方向假设，并在需要时修改表达后重测。原有 275 个计算输出中，96 个“不确定”升级为明确研究假设：正向 104、负向 46、不确定 125；另新增 28 个方向交互（正向 26、负向 2）。当前合计 303 个计算输出、6 个别名、284 个质量字段，仍为 59 家族、38 日、9,500 行。原角色与原表达不变，单侧幅度输出的方向假设附带条件。

看板默认“激进研究假设”，支持切回“原始登记”及筛选“原有表达／新增方向表达”。原始视图隐藏新增表达；新旧预期均可追溯，IC 不随预期切换改变。逐项理由、条件、竞争机制见 `有方向逻辑预期审阅.md` 与 `逻辑预期逐项审阅.csv`。最初 `factor_hypotheses.csv` 保留原样，新审阅在 `directional_hypotheses.csv` 单独冻结。此批数据已经看过，不标为期货事前检验。

新增表达只在当前原子行组合已知父因子。父因子缺失仍为缺失，精确年龄门槛不放宽；门控未激活且所有父因子有效时记零。新增 IC 完整覆盖原有全部评价组合；不根据结果反改假设。默认口径下有 16 个预期正向新增表达在 30 期均为负 IC，两个预期负向表达在多数期限为正 IC，因此不能宣称方向化已经改善预测。

可复现流程如下。初次运行会把原 275 因子实验保存到 `.cache/directional_review_base` 并校验指纹；再次运行复用该不可改写基准。`build` 写入 `.review_pending`，只重算新增表达并逐值核对原表和原 IC，验收后才发布。此脚本是基础实验之后的审阅步骤，不另建一套行情或 IC 口径。

```powershell
& $projectPython -X utf8 scripts/review_directional_ic.py build
& $projectPython -X utf8 scripts/test_opening.py result/opening_execution/.review_pending
& $projectPython -X utf8 scripts/verify_opening_artifacts.py result/opening_execution/.review_pending
& $projectPython -X utf8 scripts/verify_directional_review.py result/opening_execution/.review_pending
& 'C:/Users/Hello/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe' tests/opening_dashboard.cjs result/opening_execution/.review_pending
& $projectPython -X utf8 scripts/review_directional_ic.py publish
```

单元测试入口先加载项目 PyArrow，再加载 pandas，避免混用系统和项目 Arrow 版本造成扩展类型注册冲突。当前 49 项测试；真实数据另核对新因子的父输入有效性交集、跨缺口／未来扰动、所有新增因子的独立逐日相关与汇总，网页校验原／新假设、来源筛选和原始 IC 不变。完整证据保存在结果目录。

原版绿色筛选的 31／4 是“原始登记”口径；本次默认“激进研究假设”下为 64 个至少一期为正、5 个全部 30 期为正。这只是展示条件，不能将数量增加解释为预测能力改善。完整 IC 始终保留红色和缺失期限。

## 可输入的回看窗口（最新展示）

普通回看扩展为 **1–10 秒整数**。默认按表达合并，不再为同一表达的每个窗口重复列一行；上方输入全局默认值，或在行内输入单独的回看秒数，按 Enter／失焦后应用。当前为 **182 种计算表达、1,271 个计算版本**，加上 6 个别名和 1,100 个质量版本。38 日、9,500 个任务行、全部 LastPrice 标签与 IC 统计方法不变。

输入只切换已经真实计算的参数版本，不插值或在线拟合 IC。当前不接受小数、0、负值或超过 10 秒的输入。别名、固定事件参数、短长均线及外层统计分别标注：前后两段仍各 2／5 秒，M04/M05 滞后 2 秒，M06 为 5／10 秒，M07–M09 外层 5 秒，B09 为历史 10 秒／事件 2 秒，D09 滞后版为历史 10 秒／滞后 2 秒。快照／累计表达没有可随意替换的普通回看参数。

原始登记视图只展示之前的 5／10 秒原始表达；新增回看参数记录于 `window_hypotheses.csv`，在计算前冻结并沿用相同表达的机制假设。共同样本仍针对单一回看版本，跨观察期及未来期限求交集；不跨回看窗口。不能把窗口版本数量当成独立因子数量，也没有自动选优窗口。

`scripts/extend_history_windows.py` 接在已完成的方向审阅之后运行。先将当前审阅实验保存到 `.cache/history_window_base`，逐日计算新增窗口，再按家族并行计算新增 IC；原有值、状态、标签和全部 IC 对照保持一致。候选目录为 `.window_pending`，可恢复已完成的日期和家族计算。

```powershell
& $projectPython -X utf8 scripts/extend_history_windows.py features
& $projectPython -X utf8 scripts/extend_history_windows.py evaluate
& $projectPython -X utf8 scripts/extend_history_windows.py render
& $projectPython -X utf8 scripts/test_opening.py result/opening_execution/.window_pending
& $projectPython -X utf8 scripts/verify_opening_artifacts.py result/opening_execution/.window_pending
& $projectPython -X utf8 scripts/verify_history_windows.py result/opening_execution/.window_pending
& 'C:/Users/Hello/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe' tests/opening_dashboard.cjs result/opening_execution/.window_pending
& $projectPython -X utf8 scripts/extend_history_windows.py publish
```

本轮新增 4 项历史窗口测试，合计 53 项。包括真实回看起点、短窗支持门槛、旧窗不变与历史派生因果性。完整数据核查与输入框测试另存结果目录。绿色筛选 CSV 展开全部窗口版本，网页默认合并窗口；需选择“展开全部窗口版本”才能直接对照 CSV 数量。详细说明见 `result/opening_execution/可输入回看窗口说明.md`。
