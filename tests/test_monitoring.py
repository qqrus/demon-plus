from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demon_plus.api.contracts import ForecastResponse, Point
from demon_plus.monitoring.report import data_drift, quality_drift


def saved(origin: date, prediction: float, model: str = "v1") -> ForecastResponse:
    return ForecastResponse(
        forecast_id=f"{origin}-{model}",
        category="books",
        horizon=7,
        origin=origin,
        dataset_id="old",
        model_id=model,
        model_name="test",
        history=[],
        forecast=[Point(date=origin + timedelta(days=i), units=prediction) for i in range(1, 8)],
        total=prediction * 7,
    )


def test_quality_waits_for_complete_truth_and_deduplicates() -> None:
    panel = pd.DataFrame({"books": [10.0] * 6}, index=pd.date_range("2020-01-02", periods=6))
    forecast = saved(date(2020, 1, 1), 9)
    result = quality_drift([forecast, forecast], panel, 0.25, 5)
    assert result[0].status == "waiting"
    assert result[0].pending == 1
    assert result[0].current is None


def test_quality_detects_degradation_without_mixing_models_or_overlaps() -> None:
    panel = pd.DataFrame({"books": [10.0] * 21}, index=pd.date_range("2020-01-02", periods=21))
    forecasts = [
        saved(date(2020, 1, 1), 9),
        saved(date(2020, 1, 3), 0),
        saved(date(2020, 1, 8), 5),
        saved(date(2020, 1, 8), 10, "v2"),
    ]
    rows = quality_drift(forecasts, panel, 0.25, 5)
    assert rows[0].status == "warning"
    assert rows[0].threshold_wape == pytest.approx(15)
    assert rows[0].current is not None and rows[0].current.wape_pct == pytest.approx(50)
    assert rows[0].current_origin == date(2020, 1, 8)
    assert rows[1].status == "reference_only"
    assert quality_drift(forecasts[:2], panel, 0.25, 5)[0].status == "reference_only"


def test_quality_stable_and_zero_actuals() -> None:
    panel = pd.DataFrame({"books": [10.0] * 14}, index=pd.date_range("2020-01-02", periods=14))
    forecasts = [saved(date(2020, 1, 1), 9), saved(date(2020, 1, 8), 9)]
    assert quality_drift(forecasts, panel, 0.25, 5)[0].status == "stable"
    assert quality_drift(forecasts, panel * 0, 0.25, 5)[0].status == "undefined"


def test_evidently_shift_and_historical_thresholds(tmp_path: Path) -> None:
    values = np.tile(np.arange(7, dtype=float) + 10, 20)
    panel = pd.DataFrame({"books": values}, index=pd.date_range("2020-01-01", periods=140))
    cutoff = panel.index[111].date()
    assert data_drift(panel.iloc[:112], cutoff, tmp_path / "missing.html").status == "waiting"
    stable = data_drift(panel, cutoff, tmp_path / "stable.html")
    shifted = panel.copy()
    shifted.iloc[112:] += 100
    drifted = data_drift(shifted, cutoff, tmp_path / "shifted.html")
    assert stable.status == "stable"
    assert stable.columns[0].score == pytest.approx(0)
    assert drifted.status == "warning"
    assert drifted.columns[0].score == pytest.approx(50)
    assert drifted.columns[0].threshold == stable.columns[0].threshold
    assert (tmp_path / "shifted.html").stat().st_size > 1000


def test_constant_reference_is_finite(tmp_path: Path) -> None:
    panel = pd.DataFrame({"books": np.zeros(112)}, index=pd.date_range("2020-01-01", periods=112))
    result = data_drift(panel, panel.index[83].date(), tmp_path / "constant.html")
    assert result.status == "stable"
    assert result.columns[0].score == 0
