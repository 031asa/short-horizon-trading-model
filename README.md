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
| horizons_seconds | 1、2、3、4、5 | 信号形成后，分别检验的未来期限 |
| execution_delay_snapshots | 2 | 延迟对照从观察结束后第二张快照开始；主 IC 从观察结束开始 |
| tick_size | 0.2 | 标签以 LastPrice 的 tick 变化计 |

观察期与因子历史窗口分开；延长观察期改变信号时刻，不自动增加因子列数。本轮共 30 个输出列。不同任务若在同一时刻结束观察，应得到完全相同的因子。第一分钟限制任务产生；最后任务可以在第 64 秒完成观察，之后继续取得标签。

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
- 当前原子表：`sample_snapshot_原子执行总表.parquet`。按用户要求复用此文件名，**新表结构为任务＋因子＋未来标签**，不再包含旧 k/H 执行收益。元数据键为 `atomic_ic`，规则为 `opening-task-observation-ic-v1`。历史脚本不可直接按旧表结构读取它。
- 结果目录：`result/opening_execution/`。含 `IC研究结果.md`、`ic_heatmap.png`、`feature_registry.csv`、`daily_ic.csv`、`summary_ic.csv`、`experiment_manifest.json` 和原始数据审计文件。是否已经替换当前表，以清单的 `published` 为准。
- 当前源码：`utils/opening_schedule.py`、`utils/opening_ic.py`、`scripts/run_opening_ic.py`、`scripts/audit_opening_data.py`。
- 当前测试：`tests/test_opening*.py`。全目录还有旧研究测试，它们依赖交接包内模块，不属于本轮测试入口；直接发现全部旧测试会因缺少 `rebuild_delay3_bp` 导入而失败。
- 旧研究资料、交接包和原始行情未改写；`新窗口交接_20260921/复现参考/` 留存旧执行逻辑，不能与本轮 IC 混用。

固定日期划分为原始 40 个交易日的前 28 日开发、后 12 日后续验证，按日期顺序在查看本轮 IC 前确定。其中开发段有 2 日缺上午开盘，任务保留为空。全部日期已在旧项目探索过，不能称为完全未见的测试集。

IC 不是预测正确率或交易收益。重叠标签、同一时刻的重复观察方案不能当独立试验累加。负 IC 保留原符号；幅度/状态因子同时评价绝对价变。暂不根据本轮排行榜选择模型、执行策略或事后最佳期限。
