"""Direct multi-horizon features shared by training and inference."""

from datetime import timedelta
from typing import cast

import numpy as np
import pandas as pd

from demon_plus.errors import DataError

FEATURES = [
    "category",
    "step",
    "target_weekday",
    "target_month",
    "target_day",
    "target_weekend",
    "lag_0",
    "lag_6",
    "lag_13",
    "lag_27",
    "mean_7",
    "mean_14",
    "mean_28",
    "std_28",
    "zero_share_28",
    "weekday_mean_28",
]


def validate_panel(panel: pd.DataFrame) -> None:
    if not isinstance(panel.index, pd.DatetimeIndex) or len(panel) < 28 or panel.shape[1] == 0:
        raise DataError("Demand features need at least 28 daily observations and one category.")
    if not panel.index.equals(pd.date_range(panel.index[0], periods=len(panel), freq="D")):
        raise DataError("Demand history must have consecutive, unique, sorted daily dates.")
    values = panel.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not panel.columns.is_unique:
        raise DataError(
            "Demand history must contain finite non-negative values and unique categories."
        )


def feature_rows(panel: pd.DataFrame, origin: pd.Timestamp, horizon: int = 30) -> pd.DataFrame:
    """Slice first: supplied future values cannot affect features or their validation."""
    if origin not in panel.index or not 1 <= horizon <= 30:
        raise DataError("Invalid forecast origin or horizon (expected 1..30).")
    history = panel.loc[:origin]
    validate_panel(history)
    tail = history.tail(28)
    records: list[dict[str, object]] = []
    for step in range(1, horizon + 1):
        target_date = origin + timedelta(days=step)
        for category in history.columns:
            values = tail[category].to_numpy(dtype=float)
            weekdays = pd.DatetimeIndex(tail.index).dayofweek == target_date.dayofweek
            records.append(
                {
                    "origin": origin,
                    "date": target_date,
                    "category": str(category),
                    "step": step,
                    "target_weekday": target_date.dayofweek,
                    "target_month": target_date.month,
                    "target_day": target_date.day,
                    "target_weekend": int(target_date.dayofweek >= 5),
                    "lag_0": values[-1],
                    "lag_6": values[-7],
                    "lag_13": values[-14],
                    "lag_27": values[-28],
                    "mean_7": float(values[-7:].mean()),
                    "mean_14": float(values[-14:].mean()),
                    "mean_28": float(values.mean()),
                    "std_28": float(values.std()),
                    "zero_share_28": float((values == 0).mean()),
                    "weekday_mean_28": float(values[weekdays].mean()),
                }
            )
    return pd.DataFrame(records)


def training_rows(panel: pd.DataFrame, cutoff: pd.Timestamp, stride: int = 7) -> pd.DataFrame:
    """Only complete 30-day labels known by cutoff enter model training."""
    if stride < 1 or cutoff not in panel.index:
        raise DataError("Invalid training stride or cutoff.")
    history = panel.loc[:cutoff]
    validate_panel(history)
    if len(history) < 58:
        raise DataError("Training needs 28 feature days followed by 30 known target days.")
    frames: list[pd.DataFrame] = []
    for origin in history.index[27:-30:stride]:
        frame = feature_rows(history, origin)
        frame["actual"] = [
            float(cast(float, history.at[target, category]))
            for target, category in zip(frame["date"], frame["category"], strict=True)
        ]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
