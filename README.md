# 开盘首分钟预测与执行研究

当前已完成因子IC、三因子预测与增强、开盘时间曲线及两阶段执行消融。最新执行实验见本文末尾“两阶段执行消融”，输出在 `result/opening_execution/two_stage_ablation/`。不同阶段任务、模型与撮合口径独立保存，不混用。历史决策和附件阅读记录见 `项目口径核对.md`。

运行 `python -X utf8 scripts/package_research_results.py` 将报告、图片、表格、模型参数和主要逐任务数据汇总为 `result/研究结果汇总.zip`，附阅读指南、文件清单与SHA256核验。压缩包不包含原始行情、暂存缓存、已取消的日期验证，以及约1GB的完整逐日IC；保留完整IC汇总Parquet及短窗口逐日IC。代码通过Git单独管理，结果文件仍按原目录保存。

## 原单因子 IC 实验参数

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

普通回看扩展为 **1–10 秒整数**。默认按表达合并，不再为同一表达的每个窗口重复列一行；上方输入全局默认值，或在行内输入单独的回看秒数，按 Enter／失焦后应用。当前为 **175 种计算表达、1,271 个计算版本**，加上 6 个别名和 1,100 个质量版本。38 日、9,500 个任务行、全部 LastPrice 标签与 IC 统计方法不变。

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

分段比较表达也合并为单行：每段长度可输入已计算的 2／5 秒，分别比较相邻两段共 4／10 秒。包括 A08、C08、D03、R05 的分段变化及对应别名。普通回看输入独立于每段长度，固定差分滞后不归入此控件；完整展开仍保留全部原始版本。

当前逻辑预期已恢复保守口径：额外主假设与新增交互方向暂归不确定。175 种表达为正向 28、负向 4、不确定 143；所有 IC 原值保留。当前登记以 feature_registry.csv 为准；计算前冻结文件仅记录当时假设。可用 scripts/restore_conservative_priors.py 同步登记、看板与正向筛选导出。

短观察期逐期限比较：result/opening_execution/短观察期因子初筛.html 主实验采用观察＝回看 1–5 秒，未来 1–30 秒分别从各组观察结束起算。附加同一结束时点的历史量诊断；28 种明确预期表达完整展示，4 种特殊参数另列。取消跨期限平均分与延迟筛选门槛；11 种早段／首秒候选，但目前没有通过“短窗接近有效长窗”的研究性区间规则。页面显示实际快照数量、完整曲线及逐期限配对差异；33,600 个原口径汇总已复核。运行 scripts/screen_short_windows.py 计算，再运行 scripts/render_short_windows.py 出报告与网页；原数据不改写。测试共 60 项，结果不代表统计显著或交易收益。

## 短窗口涨跌预测试验

新试验在 `result/opening_prediction/短窗口涨跌预测检验.html`，不修改原有 IC 页面和原子表。六个单因子、最新快照三因子组合、观察＝回看 1–5 秒的六因子组合，分别预测观察结束后 1／2／3 秒的 LastPrice 涨／平／跌。采用 L2 softmax 逻辑回归，前 20 日训练后逐段向前检验，共 18 个检验日；参数及最佳单因子只在每折训练期内部选择。原始 38 日已用于筛选因子，因此这里只是内部诊断。

本机应用控制阻止新 scikit-learn 二进制加载，实际执行使用已有 NumPy 的 Newton 求解器，无新增运行依赖。对称 L2 惩罚所有类别的斜率，不惩罚截距；目标为日期等权平均交叉熵加 `||coef||² / (2*C*n_train)`。有限差分验证梯度与 Hessian，保存每折标准化和系数，可独立复算全部预测。继续使用用户指定的 Windows 本机运行方式。

依次运行 `scripts/run_opening_prediction.py`、`scripts/render_opening_prediction.py`、`scripts/verify_opening_prediction.py`。`scripts/test_opening.py result/opening_prediction` 运行 68 项测试；`tests/opening_prediction_dashboard.cjs` 检验离线网页。实验配置在首次拟合前冻结，复跑需同一输入和参数。网页提供共同任务／各窗有效任务、分类混淆、高概率信号覆盖与逐日期配对差异。原子表校验指纹不变；新试验不引入撮合、盈亏或下单策略。

六因子共线性诊断运行 `scripts/check_factor_collinearity.py`，结果位于 `result/opening_prediction/collinearity/六因子共线性报告.md`。只检查输入矩阵，不重训模型：日期等权／等行权、去除日均值、逐日及原训练段的 Pearson、Spearman、未正则化 VIF、标准化条件数与矩阵秩。主口径观察＝回看 1 秒，1–5 秒作为补充；逐因子辅助回归 VIF 与逆相关矩阵对角线交叉复核。额外三项测试覆盖正交、精确共线、常数列与量纲不变性。

三因子显著性探索运行 `scripts/check_three_significance.py`，输出到 `result/opening_prediction/significance/`。第一折训练20日、固定当时C的1000次整日重采样用于系数方向稳定性；组合预测使用18日配对差的精确双侧符号翻转、六项Holm校正及四检验段整块翻转敏感性。条件系数区间不是普通回归p值；结果不校正此前的因子／期限筛选，不代表独立验证显著性。

## 无冷启动的3秒小组合实验（当前实验口径）

用户取消10秒冷启动：每次任务只使用自身 `[T,T+W]` 观察区间内的信息，观察＝回看，连OFI／成交差分的前一快照也不得越过T。任务从开盘第0–59秒每秒生成；W取1／2／3／4／5／8／10秒，未来3秒从各自观察结束起算。38日共15,960行，原9,500行实验和旧看板不改写。源行情缺口和80%实际历史跨度规则保留，不因窗口短而借用历史。

运行 `scripts/run_no_cold_combinations.py` 生成原子表并进行训练日期内部的小组合搜索，再运行 `scripts/render_no_cold_combinations.py`、`scripts/verify_no_cold_combinations.py`。结果为 `result/opening_prediction/no_cold_start/无冷启动_3秒小组合.html`。25个主效应和6个固定交互，最多5列，VIF≤5且两两相关绝对值<0.9；参数、成员和窗口仅通过内部验证选择，窗口比较使用共同验证任务。每折结果可断点复用，输入和特征源码有指纹校验。采用已有NumPy逻辑回归，不增加运行依赖。

看板显示全部窗口、开盘0–9秒与10–59秒任务、各折实际成员、相同覆盖率诊断以及训练期冻结阈值的命中率。原子表边界、475项公式对照、19个独立任务切片、原9,500个标签一致性、56个模型的15,120条预测和共线性限制均复核；测试合计79项，离线网页966个数据单元格对照通过。新小组合未在整分钟稳定超越三因子，不能据首10秒的局部改善宣称已实现可靠高胜率。

完整期限扩展：运行 `scripts/extend_no_cold_horizons.py` 后重新运行 `scripts/render_no_cold_combinations.py`，同一看板新增观察结束后未来1–30秒准确率、样本详情与曲线。冻结已有3秒模型、组合、窗口选择和训练期阈值，只扩展累计／逐秒新增LastPrice标签，不代表每个期限分别重训。可选逐期限有效样本或七窗口、两类模型、全部30期限的共同任务；真实不变保留，准确率逐日等权。原3秒全量及冻结阈值结果逐项核对，输入文件指纹不变。完整CSV、逐日明细和标签保存在同一结果目录；`tests/no_cold_horizons.cjs` 检查全部23,040个期限单元格、明细与曲线。

## 固定三因子增强与窗口选择

独立实验固定保留最新OFI、报价位移、末价盘口位置，观察＝回看1–10秒，零冷启动，38日共22,800行。枚举原三因子、加1／2个主特征以及6个预定义复合表达与必要主效应；不按单因子IC提前删选。内部训练确定候选覆盖并检查共线性，完整训练重拟合设计再次复核；最大VIF≤5、两两相关绝对值<0.9，所有系数共同重拟合。与三因子同样本比较，统一按未来3秒训练，同一信号评价1–10秒累计和新增价变。

依次运行 `scripts/run_anchored_three.py --workers 2`、`scripts/report_anchored_three.py`、`scripts/verify_anchored_three.py`。复用既有Windows本机NumPy运行时，无新增依赖。输出位于 `result/opening_prediction/three_factor_enhancement/`，仅生成报告、表格与完整数据，不新建看板。原子表、逐任务预测、全部候选与参数、共线性及被三因子解释的R²、模型系数、折间选择和全历史拟合配置均可审阅。旧实验所有文件指纹保持不变。

训练期先按3秒日等权准确率筛选；0.5个百分点容忍范围内再比较5–10秒平均准确率，然后依次偏好短窗口、少列、低3秒Log loss、强正则。每窗口先选结构赢家，再选增强方案，最后跨窗口比较，包含退回原三因子的选项。四折前推检验18日，全部10窗口共同任务1080条。训练选窗口依次8／3／7／3秒，其3秒准确率53.06%，同窗口三因子52.41%；成员和窗口变化较大，尚未找到稳定更优窗口。全历史最终8秒配置只供下一批新数据检验，不把历史流程胜率归到该固定配置上。

验收87项测试通过；684个真实任务切片复算、456,000个标签值对照、200个模型与54,000条检验预测重放、250次结构选择和5次窗口选择复核、1,040条共同样本期限曲线复算。完整训练设计复核后的已选模型最大VIF为3.866；5个候选配置在该复核阶段超标拒绝，所有拟合均收敛。运行 `scripts/test_opening.py result/opening_prediction/three_factor_enhancement` 可保存单元测试凭证。

运行 `scripts/plot_opening_accuracy_time.py` 查看模型准确率随09:30–09:31信号时刻的变化，输出在本实验的 `time_of_minute/`。主图固定观察3秒、预测未来3秒，使用18个检验日的冻结预测；最近10秒滚动准确率配合逐秒原值、分段样本数及配对差值的日期块逐点区间，另附全部1–10秒观察窗口图。不把不同折选择的窗口混进主图，不按时段重新训练。信号必须在首分钟内，未来标签可越界；主图1,026任务，不含09:31后才形成的信号。20项分段聚合对照和原数据指纹检验随结果保存。

## 两阶段执行消融（T～T+3，T+3～T+10）

运行 `scripts/run_two_stage_execution.py`，再运行 `scripts/plot_two_stage_execution.py`。结果在 `result/opening_execution/two_stage_ablation/`，保存表格、报告、静态图片、完整成交路径及冻结模型信号，不新建看板。保留全部原预测实验。只用四折冻结的3秒增强模型，不重训、不按执行收益选参数。

A基准在T按LastPrice挂0档限价；B先观察3秒再下单；C在T先挂0档限价，第3秒仍未成交才按与B相同的信号规则处理。买入遇预测上涨、卖出遇预测下跌则转市价，否则按第3秒LastPrice挂／改限价；不变信号走限价。第6／9秒不再更新，全部策略统一T+10提交市价兜底。下单及原子撤换各延迟决策后第2条快照，原单撤换在途仍可成交；同快照原单成交优先，限价相同不重复下单。市场单按生效快照对手价、限价单按既定对手价触发或正成交量LastPrice严格穿价规则，在限价结算。单位订单，无排队、冲击、手续费或部分成交，属于L1撮合代理。

18个检验日共有1,080个名义任务。每天第0秒尚无开盘报价，不能借未来报价定初始限价，统一排除18个；三策略、买卖双向共有1,062个共同任务、6,372条有效执行路径。按同一起点LastPrice衡量有符号成本，逐日等权、买卖各半。A/B/C成本分别1.771／1.431／1.853bp；B较A节约0.340bp，C较A为−0.082bp。C比B多花0.422bp，未发现本轮先挂单带来成本改善；C平均更快完成，成本与完成速度分别评估。区间用5日日期块重采样，仍为已有日期上的探索结果。

1,080条冻结概率重放、54组观察边界截断／扰动、324条路径未来扰动、6,372笔独立最早成交扫描及原输入指纹验证通过。运行 `scripts/test_opening.py result/opening_execution/two_stage_ablation` 保存113项测试凭证，包含成交触发、严格穿价、延迟、同快照优先级、在途旧单成交、共同截止时刻及异常数据处理。主结果、分方向、四折、逐日、时段与消融贡献都保存在独立实验目录。

只看每天第一笔发单，运行 `scripts/report_first_order_execution.py`，输出到 `result/opening_execution/first_order_ablation/`。09:30:00整缺少报价，改用首条实际开盘快照作为可发单起点T（09:30:00.1～00.5），观察结束和兜底分别是T+3、T+10；不能把该口径误写成09:30:00整。原四折W=3增强模型不重训，在新时点重算观察区间特征。18天买卖各一笔，相对A基准，B严格胜率18/36=50.0%、平均节约1.557bp；C为3/36=8.3%、平均节约−1.055bp，26/36持平。小样本的日期块节约区间均包含零，暂不能确认稳定改善。18组特征截断及108条最早成交独立重放通过，方向预测准确率另表保存。

每分钟扩展运行 `scripts/run_minute_execution.py`，输出在 `result/opening_execution/every_minute/`。上午09:30–11:29、下午13:00–14:59，每分钟整点后首条快照发起任务（偏移≤0.5秒），各自观察3秒、10秒兜底。冻结原开盘模型检验向全天迁移，不重训或选阈值。18日4,320个名义任务中，共同有效4,196个；51个无及时快照、73个路径或信号无效，具体分钟及原因单列。全天B较A节约0.0108bp，95%日期块区间含零；C为−0.0677bp。09:30–09:59的B平均节约0.3075bp，但属于本轮时段探索，不能直接认定为稳定可用时段。保存完整240个分钟点、半小时、上午／下午、分买卖及逐任务明细；25,456笔有效路径独立最早成交复算、179组区间截断、09:30首笔108条对照均通过。曲线用最近10个分钟点平滑，午休两侧分别处理。

立即市价对照运行 `scripts/compare_minute_market.py`，结果在上述目录的 `market_comparison/`。新增M在任务开始即提交市价，仍延迟两条严格未来快照，按到达对手一价成交；不是按LastPrice或零延迟成交。原4,196任务全部保留，无额外排除，8,392笔双向市价均通过路径、价格和队列量检查；买卖平均成本与半价差恒等式逐任务通过。全天M成本0.6316bp，A/B/C分别比M多花0.2364／0.2256／0.3041bp。原来09:30–09:59的B虽优于A，相对M仍多花0.3871bp。市价对照与原A对照分别保留，不把基准变化混入原表。
