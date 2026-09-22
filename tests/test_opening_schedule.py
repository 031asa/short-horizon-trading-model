"""Boundary tests for the user-requested opening task clock."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from utils.opening_schedule import ScheduleConfig, build_task_schedule


class OpeningScheduleTests(unittest.TestCase):
    start = pd.Timestamp("2026-09-01 09:30:00", tz="Asia/Shanghai")

    def test_one_second_tasks_and_same_tasks_across_observation_periods(self):
        schedule = build_task_schedule(self.start, ScheduleConfig())
        self.assertEqual(len(schedule), 250)
        self.assertEqual(sorted(schedule.observation_seconds.unique()), [1, 2, 3, 4, 5])
        for _, group in schedule.groupby("observation_seconds"):
            self.assertEqual(group.task_elapsed_seconds.tolist(), list(range(10, 60)))

    def test_cold_start_keeps_clock_anchored_to_open(self):
        schedule = build_task_schedule(self.start, ScheduleConfig(cold_start_seconds=10.2))
        self.assertEqual(schedule.task_elapsed_seconds.min(), 11)
        self.assertEqual(schedule.task_elapsed_seconds.max(), 59)

    def test_observation_does_not_shift_task_or_restart_cold_start(self):
        schedule = build_task_schedule(self.start, ScheduleConfig(cold_start_seconds=0))
        first = schedule.loc[schedule.task_elapsed_seconds.eq(0)]
        self.assertEqual(first.decision_elapsed_seconds.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(first.task_time.nunique(), 1)

    def test_last_task_observation_records_boundary_without_silent_shortening(self):
        schedule = build_task_schedule(self.start, ScheduleConfig())
        row = schedule.loc[schedule.task_elapsed_seconds.eq(59) & schedule.observation_seconds.eq(5)].iloc[0]
        self.assertEqual(row.decision_elapsed_seconds, 64)
        self.assertFalse(row.decision_within_first_minute)
        self.assertNotIn(60, schedule.task_elapsed_seconds.tolist())

    def test_invalid_config_and_naive_opening(self):
        for kwargs in [{"cold_start_seconds": -1}, {"cold_start_seconds": 60},
                       {"observation_seconds": (1, 1)}, {"observation_seconds": (-1,)},
                       {"task_step_seconds": 0}]:
            with self.assertRaises(ValueError):
                ScheduleConfig(**kwargs)
        with self.assertRaises(ValueError):
            build_task_schedule(pd.Timestamp("2026-09-01 09:30"), ScheduleConfig())


if __name__ == "__main__":
    unittest.main()
