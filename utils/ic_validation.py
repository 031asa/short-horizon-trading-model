"""Causal-prefix and independently selected future-label checks."""
import numpy as np
import pandas as pd
from .opening_ic import SessionData, feature_registry

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


