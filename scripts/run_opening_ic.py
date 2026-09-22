"""Build, audit and publish the current task-level single-factor IC experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if (ROOT / ".cache/python-packages").is_dir():
    sys.path.insert(0,str(ROOT / ".cache/python-packages"))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.opening_ic import (ICConfig, RULES_VERSION, SessionData, build_session_atomic,
                              correlation_pair, feature_registry)
from utils.opening_schedule import ScheduleConfig
from scripts.audit_opening_data import RAW, EXPECTED_SHA256, OUT

CURRENT_ATOMIC = ROOT / "sample_snapshot_原子执行总表.parquet"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def daily_evaluation(atomic, registry, config):
    records=[]
    observations=list(config.schedule.observation_seconds)
    for day_number,((date,session),day) in enumerate(atomic.groupby(["trade_date","session"],sort=True),1):
        for anchor in ("decision","arrival"):
            labels=[f"y_{anchor}_{u}s" for u in config.horizons_seconds]
            for factor in registry.to_dict("records"):
                name=factor["factor"]
                row_ok=np.isfinite(day[name]) & day[labels].notna().all(axis=1)
                per_task=day.assign(_ok=row_ok).groupby("task_time").agg(n=("_ok","size"),ok=("_ok","sum"))
                common_tasks=per_task.index[(per_task.n==len(observations)) & (per_task.ok==len(observations))]
                for observation,all_rows in day.groupby("observation_seconds",sort=True):
                    for u in config.horizons_seconds:
                        label=f"y_{anchor}_{u}s"
                        targets=("signed","absolute") if factor["role"]=="magnitude_state" else ("signed",)
                        memo={}
                        for mode in ("own","common_observations_horizons"):
                            subset=all_rows if mode=="own" else all_rows.loc[all_rows.task_time.isin(common_tasks)]
                            x=subset[name].to_numpy(float)
                            raw_y=subset[label].to_numpy(float)
                            n_label=int(np.isfinite(raw_y).sum())
                            for target in targets:
                                y=np.abs(raw_y) if target=="absolute" else raw_y
                                key=(target,tuple(subset.index))
                                if key not in memo:
                                    memo[key]=correlation_pair(x,y,config.minimum_pairs)
                                n,ic,rank,status=memo[key]
                                valid=np.isfinite(x) & np.isfinite(y)
                                records.append(dict(trade_date=date,session=session,factor=name,
                                    role=factor["role"],history_seconds=factor["history_seconds"],
                                    observation_seconds=observation,horizon_seconds=u,anchor=anchor,target=target,
                                    pair_set=mode,n_scheduled=len(all_rows),n_candidate=len(subset),n_label=n_label,
                                    n_pairs=n,coverage=n/n_label if n_label else np.nan,
                                    scheduled_coverage=n/len(all_rows) if len(all_rows) else np.nan,
                                    zero_return_share=float((y[valid]==0).mean()) if n else np.nan,
                                    ic=ic,rank_ic=rank,status=status))
        if day_number%10==0:
            print(f"Evaluated daily IC for {day_number} dates",flush=True)
    return pd.DataFrame(records)


GROUPS=["session","factor","role","history_seconds","observation_seconds",
        "horizon_seconds","anchor","target","pair_set"]


def summarize_daily(daily,development_dates,validation_dates):
    parts=[]
    for phase,dates in [("development",development_dates),("validation",validation_dates),
                        ("all_exploratory",development_dates+validation_dates)]:
        part=daily.loc[daily.trade_date.isin(dates)]
        summary=part.groupby(GROUPS,dropna=False,as_index=False).agg(
            scheduled_days=("trade_date","size"),valid_days=("rank_ic","count"),
            mean_ic=("ic","mean"),median_ic=("ic","median"),
            mean_rank_ic=("rank_ic","mean"),median_rank_ic=("rank_ic","median"),
            std_rank_ic=("rank_ic","std"),mean_pairs=("n_pairs","mean"),
            total_pairs=("n_pairs","sum"),mean_scheduled_coverage=("scheduled_coverage","mean"),
            mean_zero_return_share=("zero_return_share","mean"),
            positive_rank_ic_days=("rank_ic",lambda x:int(x.gt(0).sum())),
            negative_rank_ic_days=("rank_ic",lambda x:int(x.lt(0).sum())),
            rank_ic_p10=("rank_ic",lambda x:x.quantile(.1)),
        )
        summary["positive_day_share"]=summary.positive_rank_ic_days/summary.valid_days.replace(0,np.nan)
        summary["negative_day_share"]=summary.negative_rank_ic_days/summary.valid_days.replace(0,np.nan)
        summary["rank_ic_ir"]=summary.mean_rank_ic/summary.std_rank_ic.replace(0,np.nan)
        summary.insert(0,"phase",phase)
        parts.append(summary)
    return pd.concat(parts,ignore_index=True)


def verify_actual_rows(atomic,raw,config,dates):
    checks=0
    factor_checks=0
    for date in dates:
        open_time=pd.Timestamp(f"{date} 09:30",tz="Asia/Shanghai")
        frame=raw.loc[raw.trade_date.eq(date)]
        task_grid=sorted(atomic.task_elapsed_seconds.unique())
        selected_tasks={task_grid[0],task_grid[len(task_grid)//2],task_grid[-1]}
        samples=atomic.loc[atomic.trade_date.eq(date)
                           & atomic.observation_seconds.eq(min(config.schedule.observation_seconds))
                           & atomic.task_elapsed_seconds.isin(selected_tasks)]
        for _,row in samples.iterrows():
            prefix=frame.loc[(frame.Datetime>=open_time) & (frame.Datetime<=row.decision_time)]
            reference=SessionData(prefix,open_time,config).features(row.decision_elapsed_seconds)
            for name in feature_registry(config).factor:
                np.testing.assert_allclose(row[name],reference[name],rtol=1e-12,atol=1e-10,equal_nan=True)
                factor_checks+=1
            if pd.isna(row.decision_lastprice):
                continue
            valid_history=prefix.loc[(prefix.LastPrice>0) & prefix.LastPrice.notna()]
            current=valid_history.iloc[-1]
            # Independent dataframe timestamp selection, with the same explicit gap restriction.
            after=frame.loc[frame.Datetime>row.decision_time]
            for anchor in ("decision","arrival"):
                if anchor=="decision":
                    origin_time,origin_price=row.decision_time,current.LastPrice
                else:
                    if pd.isna(row.arrival_time):continue
                    arrival=after.iloc[config.execution_delay_snapshots-1]
                    assert arrival.Datetime==row.arrival_time
                    origin_time,origin_price=arrival.Datetime,arrival.LastPrice
                for u in config.horizons_seconds:
                    key=f"y_{anchor}_{u}s"
                    future=frame.loc[frame.Datetime>=origin_time+pd.Timedelta(seconds=u)]
                    if future.empty:continue
                    end=future.iloc[0]
                    if row[key+"_status"]=="ok":
                        expected=(round(end.LastPrice/config.tick_size)-round(origin_price/config.tick_size))
                        assert row[key]==expected
                        assert row[key+"_end_time"]==end.Datetime
                        assert end.Datetime>=origin_time+pd.Timedelta(seconds=u)
                        checks+=1
    return dict(causal_prefix_feature_values_checked=factor_checks,
                independently_selected_labels_checked=checks)


def build_report(summary,metadata,output):
    schedule=metadata['config']['schedule']
    observation_grid=schedule['observation_seconds']
    reference_observation=min(observation_grid)
    first_task=np.ceil(schedule['cold_start_seconds']/schedule['task_step_seconds'])*schedule['task_step_seconds']
    last_task=(np.ceil(schedule['task_range_seconds']/schedule['task_step_seconds'])-1)*schedule['task_step_seconds']
    base=summary.loc[(summary.pair_set=="common_observations_horizons") & (summary.target=="signed")
                     & (summary.anchor=="decision") & (summary.observation_seconds==reference_observation)]
    score=base.loc[base.phase.eq("development")].groupby("factor").mean_rank_ic.apply(lambda x:x.abs().mean())
    top=score.sort_values(ascending=False).head(8).index.tolist()
    lines=["# 开盘首分钟单因子 IC：首轮结果", "",
           "当前原子表是任务、因子与未来价格标签的对齐表。没有训练预测模型，也没有模拟执行策略或计算交易收益。", "",
           "## 本轮固定配置", "",
           f"- 上午 09:30 开盘；冷启动 {schedule['cold_start_seconds']:g} 秒，每 {schedule['task_step_seconds']:g} 秒产生一个任务，任务时刻为开盘后第 {first_task:g}–{last_task:g} 秒。",
           "- 观察期为 "+"、".join(f"{v:g}" for v in observation_grid)+" 秒，观察结束立即形成因子快照。冷启动和观察期间持续积累历史。",
           f"- 首分钟限制任务产生；最晚观察结束为第 {last_task+max(observation_grid):g} 秒，后续 1–5 秒评价允许继续到分钟之后。",
           "- 30 个实际因子输出列：6 个当前/最新变化项，以及 12 个窗口表达各取 5、10 秒。B09、D09 和复杂残差暂缓。",
           "- 历史覆盖至少达到名义窗口的 80%；记录真实覆盖。最大相邻间隔 1 秒，超过则重置历史片段。",
           "- 当前取不晚于决策时刻的最后一张快照，最多陈旧 0.5 秒；未来取不早于目标时刻的第一张，最多晚 0.5 秒。",
           "- 未来变化以 LastPrice 的 tick 变化计；零变化保留，缺失不填零，不按成交条件筛选 IC 样本。",
           "- decision：因子与观察结束后价格变化；arrival：同一因子与其后第二张快照开始的价格变化。后者只作执行延迟对照。",
           "- 日内信号时点等权，先逐日计算 Pearson/Spearman，再按有效交易日等权汇总；每组至少 20 对，常数列相关系数缺失。",
           f"- 普通样本和共同任务样本同时报告；共同样本要求该因子在全部 {len(observation_grid)} 种观察期、五个未来期限均可评价。",
           "", "## 数据与验证", "",
           f"原始行情 {metadata['raw_rows']:,} 行、40 个交易日；上午开盘有行情的 38 日，另 2 日的任务保留并标记缺失。原子表 {metadata['atomic_rows']:,} 行。",
           f"开发期：{metadata['development_dates'][0]}—{metadata['development_dates'][-1]}；后续日期验证：{metadata['validation_dates'][0]}—{metadata['validation_dates'][-1]}。日期分界在查看本轮 IC 前固定。",
           "现有 40 日都曾参与旧项目探索，后续日期验证不能称为完全未见的新测试集；本轮没有因子优化或模型调参。",
           "", "## 因子与各期限的关系", "",
           f"以下因子按开发期、观察 {reference_observation:g} 秒、共同任务样本的五个期限平均绝对 Rank IC 排列；表中始终保留原始正负号。该排序用于展示，不代表已选出最终策略。", "",
           "| 因子 | 阶段 | 1s | 2s | 3s | 4s | 5s |", "|---|---|---:|---:|---:|---:|---:|"]
    for factor in top:
        for phase,title in [("development","开发"),("validation","后续验证")]:
            rows=base.loc[base.factor.eq(factor)&base.phase.eq(phase)].set_index("horizon_seconds")
            numbers=[f"{rows.loc[u,'mean_rank_ic']:+.3f}" for u in metadata['config']['horizons_seconds']]
            lines.append(f"| {factor} | {title} | "+" | ".join(numbers)+" |")
    lines.extend(["", "## 解释边界", "",
        "- IC 不是正确率、上涨概率或策略收益。持续为负的 IC 可能是反向关系，不逐日翻号或取绝对值后冒充稳定正 IC。",
        "- 相邻任务的未来区间重叠，同一决策时刻也可能属于不同观察方案，不能把这些行当独立试验累加。IR 未年化，也不是策略 Sharpe。",
        "- 观察期改变信号产生的时点。本轮滚动特征使用信号时刻之前的合法历史；同一时刻来自不同任务的特征应完全一致。观察期之间的 IC 差异本身不能证明等待创造了增量信息。",
        "- 波动、活动等状态因子另报告对绝对价格变化的 IC，不能只按有方向 IC 淘汰。",
        "- arrival 的标签基准移动到理论生效时刻，测量延迟后尚未发生的价格变化，不是从信号时刻累计到更晚终点的变化。",
        "- 后续执行口径已记录：观察结束的 LastPrice 为限价锚点，保留执行延迟；有效对手价触发或新增成交 LastPrice 严格穿价，先可见者决定成交时点，两类都按限价结算，同张标记同时可见。本轮不设挂单等待与转市价策略。",
        "", "## 文件", "",
        "- 原子表：写入新表前保存在 `.atomic_candidate.parquet`；验证通过并发布后替换项目根目录 `sample_snapshot_原子执行总表.parquet`。是否已替换以 `experiment_manifest.json` 的 `published` 为准；新表元数据类型为 task_feature_label。",
        "- `feature_registry.csv`：30 个具体输出列、家族、角色、窗口和单位。",
        "- `daily_ic.csv`：逐日、逐观察期、逐期限、两种时间基准及样本口径的 IC 和覆盖。",
        "- `summary_ic.csv`：开发期、后续验证、全样本探索性汇总。",
        "- `experiment_manifest.json`：参数、原始数据指纹、源码指纹和验证记录。",
        "- `data_audit.csv` / `data_audit.json`：原始开盘数据覆盖与异常，不含 IC 筛选。",
        f"- `ic_heatmap.png`：观察 {reference_observation:g} 秒时，开发期与后续验证的有方向 Rank IC。",
    ])
    (output/"IC研究结果.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    factors=feature_registry(ICConfig()).factor.tolist()
    matrices=[]
    for phase in ("development","validation"):
        matrices.append(base.loc[base.phase.eq(phase)].pivot(index="factor",columns="horizon_seconds",values="mean_rank_ic").reindex(factors).to_numpy())
    bound=max(.1,float(np.nanmax(np.abs(matrices))))
    fig,axes=plt.subplots(1,2,figsize=(13,12),sharey=True,layout="constrained")
    for ax,phase,matrix in zip(axes,["Development","Later-date validation"],matrices):
        im=ax.imshow(matrix,aspect="auto",cmap="RdBu_r",vmin=-bound,vmax=bound)
        ax.set_title(phase);ax.set_xticks(range(5),["1s","2s","3s","4s","5s"])
        ax.set_yticks(range(len(factors)),factors,fontsize=8)
        ax.set_xlabel("Forward LastPrice horizon")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value=matrix[row,col]
                ax.text(col,row,f"{value:+.2f}" if np.isfinite(value) else "NA",ha="center",va="center",fontsize=7,
                        color="white" if abs(value)>bound*.6 else "black")
    fig.colorbar(im,ax=axes,shrink=.65,label="Mean daily Spearman IC")
    fig.suptitle(f"Opening tasks | observation = {reference_observation:g}s | decision anchor | common tasks\nHistorical dates: exploratory research, not a fresh holdout",fontsize=12)
    fig.savefig(output/"ic_heatmap.png",dpi=160)
    plt.close(fig)
    return top


def run(cold_start=10.,observations=(1.,2.,3.,4.,5.),publish=False):
    config=ICConfig(schedule=ScheduleConfig(cold_start_seconds=cold_start,observation_seconds=tuple(observations)))
    if sha(RAW)!=EXPECTED_SHA256:raise ValueError("Raw data fingerprint mismatch")
    raw=pd.read_parquet(RAW)
    raw["Datetime"]=pd.to_datetime(raw.Datetime).dt.tz_convert("Asia/Shanghai")
    raw["source_row"]=np.arange(len(raw))
    raw["trade_date"]=raw.Datetime.dt.strftime("%Y-%m-%d")
    dates=sorted(raw.trade_date.unique())
    # Fixed before evaluating any IC; chronological, not shuffled snapshots.
    development_dates,validation_dates=dates[:28],dates[28:]
    contract=raw.Contract.dropna().unique()
    if len(contract)!=1:raise ValueError("Expected a single contract")
    parts=[]
    for number,date in enumerate(dates,1):
        open_time=pd.Timestamp(f"{date} 09:30",tz="Asia/Shanghai")
        parts.append(build_session_atomic(raw.loc[raw.trade_date.eq(date)],open_time,"AM",contract[0],config))
        if number%10==0:print(f"Built {number}/{len(dates)} dates",flush=True)
    atomic=pd.concat(parts,ignore_index=True)
    keys=["trade_date","session","task_time","observation_seconds"]
    assert not atomic.duplicated(keys).any()
    assert atomic.task_elapsed_seconds.ge(cold_start).all() and atomic.task_elapsed_seconds.lt(60).all()
    registry=feature_registry(config)
    assert len(registry)==30
    assert np.isfinite(atomic[registry.factor].to_numpy(float)).sum()>0
    # All versions of one decision instant must have identical factor values.
    assert atomic.groupby(["trade_date","session","decision_time"])[registry.factor.tolist()].nunique(dropna=False).le(1).all().all()
    checks=verify_actual_rows(atomic,raw,config,dates)
    print("Verified causal prefixes and independent future-price selections",checks,flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    daily=daily_evaluation(atomic,registry,config)
    summary=summarize_daily(daily,development_dates,validation_dates)
    print(f"Computed {len(daily):,} daily IC records",flush=True)
    registry.to_csv(OUT/"feature_registry.csv",index=False,encoding="utf-8-sig")
    daily.to_csv(OUT/"daily_ic.csv",index=False,encoding="utf-8-sig")
    summary.to_csv(OUT/"summary_ic.csv",index=False,encoding="utf-8-sig")
    source_files=["utils/opening_ic.py","utils/opening_schedule.py","scripts/run_opening_ic.py","scripts/audit_opening_data.py"]
    metadata=dict(rules_version=RULES_VERSION,table_role="task_feature_label",config=config.to_dict(),
        raw_sha256=EXPECTED_SHA256,raw_rows=len(raw),atomic_rows=len(atomic),
        development_dates=development_dates,validation_dates=validation_dates,
        feature_columns=registry.factor.tolist(),source_sha256={name:sha(ROOT/name) for name in source_files},
        verification=checks,ic_daily_rows=len(daily),time_range_policy="first_minute_tasks_followup_allowed",
        sessions={"AM":"09:30"},minimum_pairs=config.minimum_pairs,
        old_current_sha256=sha(CURRENT_ATOMIC) if CURRENT_ATOMIC.exists() else None,
        future_execution_conventions=dict(limit_anchor="decision_LastPrice",delay_snapshots=2,
            quote_condition="buy limit>=ask; sell limit<=bid",trade_condition="strict LastPrice cross and deltaVolume>0",
            limit_fill_price="limit for both triggers",same_snapshot="both_visible; no inferred intrainterval order",
            execution_strategy="not implemented; not used in IC"),
        scientific_status="descriptive single-factor research; overlapping samples; no fresh holdout")
    candidate=OUT/".atomic_candidate.parquet"
    table=pa.Table.from_pandas(atomic,preserve_index=False)
    table=table.replace_schema_metadata({**(table.schema.metadata or {}),b"atomic_ic":json.dumps(metadata,ensure_ascii=False).encode("utf-8")})
    pq.write_table(table,candidate,compression="zstd")
    reread=pd.read_parquet(candidate)
    pd.testing.assert_frame_equal(atomic,reread)
    assert json.loads(pq.read_schema(candidate).metadata[b"atomic_ic"])["rules_version"]==RULES_VERSION
    metadata["atomic_sha256"]=sha(candidate)
    metadata["published"]=False
    top=build_report(summary,metadata,OUT)
    metadata["display_ranking_development_only"]=top
    if publish:
        os.replace(candidate,CURRENT_ATOMIC)
        assert sha(CURRENT_ATOMIC)==metadata["atomic_sha256"]
        metadata["published"]=True
    (OUT/"experiment_manifest.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"rows":len(atomic),"factor_columns":len(registry),"published":metadata['published'],
                      "output":str(OUT),"top_display_factors":top,"checks":checks},ensure_ascii=False,indent=2),flush=True)
    return metadata


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cold-start",type=float,default=10.)
    parser.add_argument("--observations",type=float,nargs="+",default=[1.,2.,3.,4.,5.])
    parser.add_argument("--publish",action="store_true",help="Replace the active legacy atomic file after verification")
    args=parser.parse_args()
    run(args.cold_start,args.observations,args.publish)
