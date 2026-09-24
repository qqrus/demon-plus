"""Report WAPE as a percentage; all-zero actuals have undefined WAPE."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from demon_plus.errors import DataError


class RegressionMetrics(BaseModel):
    mae: float
    rmse: float
    wape_pct: float | None


def regression_metrics(
    actual: Sequence[float] | NDArray[np.float64],
    prediction: Sequence[float] | NDArray[np.float64],
) -> RegressionMetrics:
    truth = np.asarray(actual, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if truth.ndim != 1 or truth.size == 0 or truth.shape != predicted.shape:
        raise DataError("Metrics need non-empty, matching one-dimensional arrays.")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise DataError("Metrics do not accept missing or infinite values.")
    if (truth < 0).any() or (predicted < 0).any():
        raise DataError("Demand metrics require non-negative quantities.")
    denominator = float(truth.sum())
    wape = float(np.abs(truth - predicted).sum() / denominator * 100) if denominator else None
    return RegressionMetrics(
        mae=float(mean_absolute_error(truth, predicted)),
        rmse=float(root_mean_squared_error(truth, predicted)),
        wape_pct=wape,
    )
