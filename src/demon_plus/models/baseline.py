"""Simple, inspectable baselines using only history supplied by the caller."""

from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd

from demon_plus.data.olist import UNKNOWN_CATEGORY
from demon_plus.errors import DataError

BaselineName = Literal["seasonal_naive_7", "mean_28"]
BASELINES: tuple[BaselineName, ...] = ("seasonal_naive_7", "mean_28")


def select_categories(
    panel: pd.DataFrame, cutoff: pd.Timestamp, top_k: int, min_active_days: int
) -> list[str]:
    """Freeze the cohort using the first training window only."""
    history = panel.loc[:cutoff].drop(columns=[UNKNOWN_CATEGORY], errors="ignore")
    eligible = history.columns[(history > 0).sum() >= min_active_days]
    volumes = history[eligible].sum().sort_index().sort_values(ascending=False, kind="stable")
    if top_k < 1 or min_active_days < 1 or len(volumes) < top_k:
        raise DataError(
            "Not enough eligible categories; review training coverage and cohort settings."
        )
    return [str(category) for category in volumes.head(top_k).index]


def forecast(history: pd.DataFrame, model: BaselineName, horizon: int = 30) -> pd.DataFrame:
    """Forecast all days at once; never update lags from future actual sales."""
    if horizon < 1 or horizon > 30 or len(history) < 28 or history.shape[1] == 0:
        raise DataError("Baseline needs at least 28 days, one category and a horizon of 1..30.")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise DataError("History must have a daily DatetimeIndex.")
    expected = pd.date_range(history.index[0], periods=len(history), freq="D")
    if not history.index.equals(expected):
        raise DataError("History dates must be unique, sorted and consecutive.")
    values = history.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise DataError("History must contain finite, non-negative sales.")
    if model == "seasonal_naive_7":
        predictions = np.tile(values[-7:], (5, 1))[:horizon]
    elif model == "mean_28":
        predictions = np.tile(values[-28:].mean(axis=0), (horizon, 1))
    else:
        raise DataError("Unknown baseline model.")
    dates = pd.date_range(history.index[-1] + timedelta(days=1), periods=horizon)
    return pd.DataFrame(predictions, index=dates, columns=history.columns).rename_axis("date")
