"""Causal opening features and separately generated future-price labels.

This is a task/feature/label atomic table, not an execution-return table.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
import numpy as np
import pandas as pd

from .opening_schedule import ScheduleConfig, build_task_schedule

RULES_VERSION = "opening-task-observation-ic-v1"


@dataclass(frozen=True)
class ICConfig:
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    history_seconds: tuple[int, ...] = (5, 10)
    horizons_seconds: tuple[int, ...] = (1, 2, 3, 4, 5)
    tick_size: float = .2
    minimum_coverage: float = .8
    maximum_gap_seconds: float = 1.0
    maximum_source_age_seconds: float = .5
    maximum_label_error_seconds: float = .5
    execution_delay_snapshots: int = 2
    minimum_pairs: int = 20

    def __post_init__(self):
        if not math.isfinite(self.tick_size) or self.tick_size <= 0:
            raise ValueError("tick_size must be finite and positive")
        for values in (self.history_seconds, self.horizons_seconds):
            if (not values or len(values) != len(set(values))
                or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values)):
                raise ValueError("History and horizon grids must contain unique positive integers")
        if not 0 < self.minimum_coverage <= 1:
            raise ValueError("minimum_coverage must lie in (0,1]")
        for name in ("maximum_gap_seconds", "maximum_source_age_seconds", "maximum_label_error_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.execution_delay_snapshots < 1 or self.minimum_pairs < 3:
            raise ValueError("Invalid delay or minimum_pairs")

    def to_dict(self):
        return asdict(self)


def feature_registry(config: ICConfig) -> pd.DataFrame:
    rows = []
    current = [
        ("F01_QI", "QueueImbalance", "direction", "ratio", "queue"),
        ("F02_QIChange", "QueueImbalanceChange", "direction", "ratio", "queue"),
        ("F03_NOFI_latest", "SnapshotOFI", "direction", "ratio", "ofi"),
        ("F04_QuoteShift", "QuoteShift", "direction", "tick", "book"),
        ("F05_LastChange", "LastPriceMomentum", "direction", "tick", "price"),
        ("F07_QuotePosition", "LastPriceQuotePosition", "direction", "ratio", "position"),
    ]
    for name, family, role, unit, dependency in current:
        rows.append(dict(factor=name, family=family, role=role, unit=unit,
                         history_seconds=0, dependency=dependency))
    windowed = [
        ("F03_NOFI", "SnapshotOFI", "direction", "ratio", "ofi"),
        ("F05_Momentum", "LastPriceMomentum", "direction", "tick", "price"),
        ("F06_PathEfficiency", "SignedPathEfficiency", "direction", "ratio", "price"),
        ("F08_SignedVolume", "LastPriceSignedVolume", "direction", "ratio", "price_volume"),
        ("A03_QIMean", "ImbalanceLevelAndVolatility", "direction", "ratio", "queue"),
        ("A03_QIStd", "ImbalanceLevelAndVolatility", "magnitude_state", "ratio", "queue"),
        ("A07_OFIDirection", "OFIDirectionAndActivity", "direction", "ratio", "ofi"),
        ("A07_OFIActivity", "OFIDirectionAndActivity", "magnitude_state", "per_second", "ofi"),
        ("B01_MeanQueueImbalance", "TimeMeanQueueImbalance", "direction", "ratio", "queue"),
        ("C01_PriceRV", "PricePathVariation", "magnitude_state", "tick/sqrt(second)", "price"),
        ("D01_VolumeRate", "VolumeActivityRate", "magnitude_state", "volume/second", "volume"),
        ("M01_MADeviation", "PriceToRollingMean", "direction", "tick", "price"),
    ]
    for h in config.history_seconds:
        for name, family, role, unit, dependency in windowed:
            rows.append(dict(factor=f"{name}_h{h}s", family=family, role=role,
                             unit=unit, history_seconds=h, dependency=dependency))
    return pd.DataFrame(rows)


def limit_fill_evidence(direction: int, limit: float, last: float,
                        bid: float, ask: float, delta_volume: float) -> tuple[bool, str]:
    """User-confirmed evidence rule; both triggers settle at limit, not quote.

    Caller must supply valid quotes and eligible post-delay observations. The IC
    pipeline does not call this function or select observations by fills.
    """
    if direction not in (-1, 1) or not all(math.isfinite(x) for x in (limit, last, bid, ask)):
        raise ValueError("Invalid direction or price")
    if min(limit, last, bid, ask) <= 0 or bid >= ask:
        raise ValueError("Prices must be positive with a valid spread")
    quote = ask <= limit if direction == 1 else bid >= limit
    trade = (delta_volume > 0 and (last < limit if direction == 1 else last > limit))
    source = "both_visible" if quote and trade else "quote" if quote else "lastprice" if trade else "none"
    return quote or trade, source


class SessionData:
    def __init__(self, frame: pd.DataFrame, open_time: pd.Timestamp, config: ICConfig):
        self.config = config
        self.open_time = pd.Timestamp(open_time)
        frame = frame.copy()
        if "source_row" not in frame:
            frame["source_row"] = np.arange(len(frame))
        frame["Datetime"] = pd.to_datetime(frame["Datetime"]).dt.tz_convert("Asia/Shanghai")
        # Includes a bounded continuation for the last task's observation and labels.
        tail = (config.schedule.task_range_seconds + max(config.schedule.observation_seconds)
                + max(config.horizons_seconds) + 10)
        frame = frame.loc[(frame.Datetime >= open_time)
                          & (frame.Datetime <= open_time + pd.Timedelta(seconds=tail))].reset_index(drop=True)
        self.frame = frame
        self.times = (frame.Datetime - open_time).dt.total_seconds().to_numpy(float)
        n = len(frame)
        self.n = n
        if n and np.any(np.diff(self.times) < 0):
            raise ValueError("Out-of-order opening data; source was not silently sorted")
        unique_time = ~frame.Datetime.duplicated(keep=False).to_numpy()
        self.dt = np.r_[np.nan, np.diff(self.times)] if n else np.array([])
        adjacent = np.r_[False, np.diff(frame.source_row.to_numpy()) == 1] if n else np.array([], dtype=bool)
        self.time_edge = adjacent & (self.dt > 0) & (self.dt <= config.maximum_gap_seconds)

        def price_ticks(column):
            x = frame[column].to_numpy(float)
            tick = np.rint(x / config.tick_size)
            valid = (np.isfinite(x) & (x > 0) & (np.abs(tick) <= 2**52)
                     & (np.abs(x - tick * config.tick_size) <= config.tick_size * 1e-6))
            return np.where(valid, tick, np.nan), valid

        self.p, valid_p = price_ticks("LastPrice")
        self.b, valid_b = price_ticks("BidPrice1")
        self.a, valid_a = price_ticks("AskPrice1")
        self.qb = frame.BidVolume1.to_numpy(float)
        self.qa = frame.AskVolume1.to_numpy(float)
        self.volume = frame.Volume.to_numpy(float)
        valid_q = (np.isfinite(self.qb) & np.isfinite(self.qa) & (self.qb > 0) & (self.qa > 0)
                   & (self.qb == np.floor(self.qb)) & (self.qa == np.floor(self.qa)))
        valid_v = (np.isfinite(self.volume) & (self.volume >= 0)
                   & (self.volume == np.floor(self.volume)))
        valid_book = valid_b & valid_a & (self.a > self.b)
        self.valid = {
            "price": valid_p & unique_time, "book": valid_book & unique_time,
            "queue": valid_q & unique_time, "ofi": valid_q & valid_book & unique_time,
            "volume": valid_v & unique_time,
            "price_volume": valid_v & valid_p & unique_time,
            "position": valid_p & valid_book & unique_time,
        }
        self.dv = np.r_[np.nan, np.diff(self.volume)] if n else np.array([])
        self.edge, self.segment = {}, {}
        for group, valid in self.valid.items():
            previous = np.r_[False, valid[:-1]] if n else np.array([], dtype=bool)
            edge = self.time_edge & valid & previous
            if group in ("volume", "price_volume"):
                edge &= self.dv >= 0
            segment = np.full(n, -1, dtype=int)
            for i in range(n):
                if valid[i]:
                    segment[i] = segment[i - 1] if edge[i] else i
            self.edge[group], self.segment[group] = edge, segment
        self.depth = self.qb + self.qa
        self.qi = np.full(n, np.nan)
        self.qi[self.valid["queue"]] = ((self.qb - self.qa) / self.depth)[self.valid["queue"]]
        self.ofi = np.full(n, np.nan)
        for i in np.flatnonzero(self.edge["ofi"]):
            bid_flow = (self.qb[i] if self.b[i] > self.b[i-1]
                        else -self.qb[i-1] if self.b[i] < self.b[i-1]
                        else self.qb[i] - self.qb[i-1])
            ask_flow = (-self.qa[i-1] if self.a[i] > self.a[i-1]
                        else self.qa[i] if self.a[i] < self.a[i-1]
                        else self.qa[i] - self.qa[i-1])
            self.ofi[i] = bid_flow - ask_flow

    def source_index(self, decision_seconds: float) -> int:
        return int(np.searchsorted(self.times, decision_seconds, side="right")) - 1

    def window(self, index: int, h: int, group: str):
        if index < 0 or not self.valid[group][index]:
            return None
        start = max(self.segment[group][index],
                    int(np.searchsorted(self.times, max(0, self.times[index] - h), side="left")))
        coverage = self.times[index] - self.times[start]
        return start, coverage, np.diff(self.times[start:index+1])

    def features(self, decision_seconds: float) -> dict:
        config = self.config
        result = {name: np.nan for name in feature_registry(config).factor}
        index = self.source_index(decision_seconds)
        result.update(feature_time=pd.NaT, feature_source_age_seconds=np.nan,
                      feature_status="NO_SOURCE", decision_lastprice=np.nan)
        if index < 0:
            return result
        age = decision_seconds - self.times[index]
        result.update(feature_time=self.frame.Datetime.iloc[index], feature_source_age_seconds=age)
        if age > config.maximum_source_age_seconds + 1e-10:
            result["feature_status"] = "STALE_SOURCE"
            return result
        result["feature_status"] = "ok"
        if self.valid["price"][index]:
            result["decision_lastprice"] = self.p[index] * config.tick_size
        if self.valid["queue"][index]:
            result["F01_QI"] = self.qi[index]
        if self.edge["queue"][index]:
            result["F02_QIChange"] = self.qi[index] - self.qi[index-1]
        if self.edge["ofi"][index]:
            result["F03_NOFI_latest"] = self.ofi[index] / ((self.depth[index] + self.depth[index-1]) / 2)
        if self.edge["book"][index]:
            result["F04_QuoteShift"] = (self.a[index] - self.a[index-1] + self.b[index] - self.b[index-1]) / 2
        if self.edge["price"][index]:
            result["F05_LastChange"] = self.p[index] - self.p[index-1]
        if self.valid["position"][index]:
            result["F07_QuotePosition"] = (self.a[index] + self.b[index] - 2*self.p[index]) / (self.a[index] - self.b[index])
        for h in config.history_seconds:
            suffix = f"_h{h}s"
            windows = {}
            for group in ("price", "queue", "ofi", "volume", "price_volume"):
                window = self.window(index, h, group)
                result[f"coverage_{group}{suffix}"] = window[1] if window else np.nan
                result[f"count_{group}{suffix}"] = index-window[0]+1 if window else 0
                if window and window[1] > 0 and window[1] / h >= config.minimum_coverage - 1e-10:
                    windows[group] = window
            if "price" in windows:
                start, coverage, weight = windows["price"]
                moves = np.diff(self.p[start:index+1])
                mean = np.dot(weight, self.p[start:index]) / coverage
                net = self.p[index] - self.p[start]
                path = np.abs(moves).sum()
                result["F05_Momentum"+suffix] = net
                result["F06_PathEfficiency"+suffix] = net/path if path > 0 else 0.0
                result["C01_PriceRV"+suffix] = math.sqrt(np.dot(moves, moves)/coverage)
                result["M01_MADeviation"+suffix] = self.p[index] - mean
            if "queue" in windows:
                start, coverage, weight = windows["queue"]
                mean_qi = np.dot(weight, self.qi[start:index]) / coverage
                result["A03_QIMean"+suffix] = mean_qi
                result["A03_QIStd"+suffix] = math.sqrt(np.dot(weight, (self.qi[start:index]-mean_qi)**2)/coverage)
                mean_bid = np.dot(weight, self.qb[start:index])/coverage
                mean_ask = np.dot(weight, self.qa[start:index])/coverage
                result["B01_MeanQueueImbalance"+suffix] = (mean_bid-mean_ask)/(mean_bid+mean_ask)
            if "ofi" in windows:
                start, coverage, weight = windows["ofi"]
                mean_depth = np.dot(weight, self.depth[start:index])/coverage
                values = self.ofi[start+1:index+1]
                total, absolute = values.sum(), np.abs(values).sum()
                result["F03_NOFI"+suffix] = total/mean_depth
                result["A07_OFIDirection"+suffix] = total/absolute if absolute > 0 else np.nan
                result["A07_OFIActivity"+suffix] = absolute/(mean_depth*coverage)
            if "volume" in windows:
                start, coverage, _ = windows["volume"]
                result["D01_VolumeRate"+suffix] = self.dv[start+1:index+1].sum()/coverage
            if "price_volume" in windows:
                start, _, _ = windows["price_volume"]
                volume = self.dv[start+1:index+1]
                total = volume.sum()
                if total > 0:
                    result["F08_SignedVolume"+suffix] = np.dot(np.sign(np.diff(self.p[start:index+1])), volume)/total
        return result

    def labels(self, decision_seconds: float) -> dict:
        config = self.config
        current = self.source_index(decision_seconds)
        result = {"arrival_time": pd.NaT, "arrival_lastprice": np.nan,
                  "arrival_delay_seconds": np.nan, "arrival_status": "NO_SOURCE"}
        first_after = int(np.searchsorted(self.times, decision_seconds, side="right"))
        arrival = first_after + config.execution_delay_snapshots - 1
        source_ok = (current >= 0 and self.valid["price"][current]
                     and decision_seconds-self.times[current] <= config.maximum_source_age_seconds + 1e-10)
        arrival_ok = (source_ok and arrival < self.n and self.valid["price"][arrival]
                      and self.segment["price"][arrival] <= current)
        if source_ok:
            result["arrival_status"] = "ok" if arrival_ok else "MISSING_OR_GAPPED_ARRIVAL"
        if arrival_ok:
            result.update(arrival_time=self.frame.Datetime.iloc[arrival],
                          arrival_lastprice=self.p[arrival]*config.tick_size,
                          arrival_delay_seconds=self.times[arrival]-decision_seconds)
        for anchor in ("decision", "arrival"):
            origin = current if anchor == "decision" else arrival
            valid_origin = source_ok if anchor == "decision" else arrival_ok
            start = decision_seconds if anchor == "decision" else (self.times[arrival] if arrival_ok else np.nan)
            for u in config.horizons_seconds:
                key = f"y_{anchor}_{u}s"
                result.update({key: np.nan, key+"_status": "INVALID_ORIGIN",
                               key+"_end_time": pd.NaT, key+"_actual_seconds": np.nan,
                               key+"_alignment_error_seconds": np.nan})
                if not valid_origin:
                    continue
                end = int(np.searchsorted(self.times, start+u, side="left"))
                if end >= self.n:
                    result[key+"_status"] = "NO_FUTURE"
                    continue
                error = self.times[end] - start - u
                result[key+"_alignment_error_seconds"] = error
                if error > config.maximum_label_error_seconds + 1e-10:
                    result[key+"_status"] = "LATE_FUTURE"
                elif not self.valid["price"][end] or self.segment["price"][end] > origin:
                    result[key+"_status"] = "GAPPED_OR_INVALID_FUTURE"
                else:
                    result.update({key: self.p[end]-self.p[origin], key+"_status": "ok",
                                   key+"_end_time": self.frame.Datetime.iloc[end],
                                   key+"_actual_seconds": self.times[end]-start})
        return result


def build_session_atomic(frame: pd.DataFrame, open_time: pd.Timestamp, session: str,
                         contract: str, config: ICConfig) -> pd.DataFrame:
    data = SessionData(frame, open_time, config)
    schedule = build_task_schedule(open_time, config.schedule)
    # Identical decision times reuse exactly the same causal factor snapshot.
    cache = {}
    rows = []
    for task in schedule.to_dict("records"):
        decision = task["decision_elapsed_seconds"]
        if decision not in cache:
            cache[decision] = {**data.features(decision), **data.labels(decision)}
        rows.append({"trade_date": open_time.strftime("%Y-%m-%d"), "session": session,
                     "contract": contract, **task, **cache[decision]})
    result = pd.DataFrame(rows)
    # Missing openings must have the same timezone-aware schema as observed days.
    for name in result.columns:
        if name.endswith("_time"):
            result[name] = pd.to_datetime(result[name], utc=True).dt.tz_convert("Asia/Shanghai").dt.as_unit("us")
    return result


def correlation_pair(x, y, minimum_pairs: int):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    a, b = x[mask], y[mask]
    n = len(a)
    if n < minimum_pairs:
        return n, np.nan, np.nan, "NOT_ENOUGH_PAIRS"
    if np.unique(a).size < 2:
        return n, np.nan, np.nan, "CONSTANT_FACTOR"
    if np.unique(b).size < 2:
        return n, np.nan, np.nan, "CONSTANT_LABEL"

    def corr(v, w):
        v, w = v-v.mean(), w-w.mean()
        return float(np.clip(np.dot(v,w)/(np.linalg.norm(v)*np.linalg.norm(w)), -1, 1))

    pearson = corr(a,b)
    ranked_a = pd.Series(a).rank(method="average").to_numpy()
    ranked_b = pd.Series(b).rank(method="average").to_numpy()
    return n, pearson, corr(ranked_a,ranked_b), "ok"
