"""Read-only market-data audit; never repairs or overwrites source observations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT / ".cache" / "python-packages"
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW = ROOT / "新窗口交接_20260921" / "20260720_20260911_IC2609.parquet"
EXPECTED_SHA256 = "3245abaf6da37d159c0179177178b253bc8e09fd82f59861098aac185dfcd44c"
OUT = ROOT / "result" / "opening_execution"


def audit() -> dict:
    with RAW.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Raw-file digest differs from the verified handoff")
    frame = pd.read_parquet(RAW)
    stamps = pd.to_datetime(frame["Datetime"])
    if stamps.isna().any() or stamps.dt.tz is None:
        raise ValueError("Expected nonmissing timezone-aware Datetime")
    stamps = stamps.dt.tz_convert("Asia/Shanghai")
    frame = frame.assign(Datetime=stamps, source_row=np.arange(len(frame)))
    dates = stamps.dt.strftime("%Y-%m-%d")
    records = []
    for date in sorted(dates.unique()):
        day = frame.loc[dates == date]
        for session, clock in [("AM", "09:30"), ("PM", "13:00")]:
            start = pd.Timestamp(f"{date} {clock}", tz="Asia/Shanghai")
            elapsed = (day.Datetime - start).dt.total_seconds()
            view = day.loc[elapsed.ge(0) & elapsed.lt(60)]
            dt = view.Datetime.diff().dt.total_seconds()
            last = view.LastPrice.to_numpy(float)
            bid = view.BidPrice1.to_numpy(float)
            ask = view.AskPrice1.to_numpy(float)
            q_bid = view.BidVolume1.to_numpy(float)
            q_ask = view.AskVolume1.to_numpy(float)
            invalid_last = (~np.isfinite(last)) | (last <= 0)
            invalid_quote = ((~np.isfinite(bid)) | (~np.isfinite(ask))
                             | (bid <= 0) | (ask <= bid)
                             | (~np.isfinite(q_bid)) | (~np.isfinite(q_ask))
                             | (q_bid <= 0) | (q_ask <= 0))
            off_grid = sum(int((np.abs(x / .2 - np.rint(x / .2)) > 1e-6).sum())
                           for x in [last, bid, ask])
            dv = view.Volume.diff()
            record = {
                "trade_date": date, "session": session, "rows_first_minute": len(view),
                "first_elapsed_seconds": (float((view.Datetime.iloc[0]-start).total_seconds())
                                          if len(view) else None),
                "last_elapsed_seconds": (float((view.Datetime.iloc[-1]-start).total_seconds())
                                         if len(view) else None),
                "duplicate_timestamps": int(view.Datetime.duplicated(keep=False).sum()),
                "backward_intervals": int(dt.lt(0).sum()),
                "max_interval_seconds": float(dt.max()) if dt.notna().any() else None,
                "intervals_gt_0_5s": int(dt.gt(.5).sum()),
                "intervals_gt_1s": int(dt.gt(1).sum()),
                "invalid_last_rows": int(invalid_last.sum()),
                "invalid_quote_rows": int(invalid_quote.sum()),
                "off_grid_price_cells": off_grid,
                "negative_volume_increments": int(dv.lt(0).sum()),
                "unchanged_volume_with_changed_last": int((dv.eq(0) & view.LastPrice.diff().ne(0)).sum()),
                "rows_next_minute": int((elapsed.ge(60) & elapsed.lt(120)).sum()),
            }
            records.append(record)
    opening = pd.DataFrame(records)
    old_path = ROOT / "sample_snapshot_原子执行总表.parquet"
    old_meta = {}
    if old_path.exists():
        parquet = pq.ParquetFile(old_path)
        schema_meta = parquet.schema_arrow.metadata or {}
        metadata = json.loads(schema_meta.get(b"atomic_ic", schema_meta.get(b"atomic_execution", b"{}")))
        with old_path.open("rb") as stream:
            old_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        old_meta = {"path": str(old_path.relative_to(ROOT)), "rows": parquet.metadata.num_rows,
                    "rules_version": metadata.get("rules_version"), "dates": metadata.get("dates"),
                    "sha256": old_digest, "table_role": metadata.get("table_role", "legacy_execution")}
    summary = {
        "raw_path": str(RAW.relative_to(ROOT)), "raw_sha256": digest,
        "raw_rows": len(frame), "date_count": int(dates.nunique()),
        "contracts": sorted(frame.Contract.dropna().unique().tolist()),
        "global_duplicate_timestamps": int(frame.Datetime.duplicated(keep=False).sum()),
        "global_backward_intervals": int(frame.Datetime.diff().dt.total_seconds().lt(0).sum()),
        "opening_rows": int(opening.rows_first_minute.sum()),
        "openings_with_data": opening.loc[opening.rows_first_minute.gt(0)].groupby("session").size().to_dict(),
        "missing_openings": opening.loc[opening.rows_first_minute.eq(0), ["trade_date", "session"]].to_dict("records"),
        "opening_invalid_quote_rows": int(opening.invalid_quote_rows.sum()),
        "opening_invalid_last_rows": int(opening.invalid_last_rows.sum()),
        "opening_negative_volume_increments": int(opening.negative_volume_increments.sum()),
        "opening_duplicates": int(opening.duplicate_timestamps.sum()),
        "current_atomic_at_audit": old_meta,
        "scope": "Data audit only; no execution or IC outcomes generated.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    opening.to_csv(OUT / "data_audit.csv", index=False, encoding="utf-8-sig")
    (OUT / "data_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Incomplete/irregular openings:")
    print(opening.loc[opening.rows_first_minute.ne(120) | opening.intervals_gt_0_5s.gt(0)
                      | opening.invalid_quote_rows.gt(0)].to_string(index=False))
    return summary


if __name__ == "__main__":
    audit()
