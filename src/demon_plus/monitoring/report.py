"""Compare complete windows without treating missing truth as zero."""

import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from demon_plus.api.contracts import Contract, ForecastResponse
from demon_plus.evaluation.metrics import RegressionMetrics, regression_metrics


class DriftColumn(Contract):
    category: str
    score: float
    threshold: float
    drift: bool
    calibration_windows: int


class DataDrift(Contract):
    status: Literal["waiting", "stable", "warning"]
    message: str
    reference_start: date | None = None
    reference_end: date | None = None
    current_start: date | None = None
    current_end: date | None = None
    columns: list[DriftColumn] = []


class Quality(Contract):
    model_id: str
    category: str
    horizon: int
    status: Literal["waiting", "reference_only", "undefined", "stable", "warning"]
    pending: int
    matured: int
    reference_origin: date | None = None
    current_origin: date | None = None
    reference: RegressionMetrics | None = None
    current: RegressionMetrics | None = None
    threshold_wape: float | None = None


class MonitoringReport(Contract):
    report_id: str
    dataset_id: str
    model_id: str
    as_of: date
    data_drift: DataDrift
    quality: list[Quality]
    html_available: bool


def distance(reference: NDArray[np.float64], current: NDArray[np.float64]) -> float:
    """Equal-size empirical Wasserstein, normalized exactly as Evidently does."""
    return float(np.abs(np.sort(reference) - np.sort(current)).mean()) / max(
        float(np.std(reference)), 0.001
    )


def data_drift(panel: pd.DataFrame, cutoff: date, destination: Path) -> DataDrift:
    historical = panel.loc[: pd.Timestamp(cutoff)]
    current = panel.loc[panel.index > pd.Timestamp(cutoff)].tail(28)
    if len(historical) < 84 or len(current) < 28:
        return DataDrift(
            status="waiting",
            message="Нужны 84 дня истории до обучения и 28 новых дней после него.",
        )
    reference = historical.tail(28)
    thresholds: dict[str, float] = {}
    counts: dict[str, int] = {}
    for category in panel.columns:
        values = historical[category].to_numpy(dtype=float)
        # Only training history, never the newly uploaded window, sets thresholds.
        calibration = [
            distance(values[end - 56 : end - 28], values[end - 28 : end])
            for end in range(56, len(values) + 1, 28)
        ]
        thresholds[category] = max(0.1, float(np.quantile(calibration, 0.95)) * 1.1)
        counts[category] = len(calibration)

    # Evidently 0.7 imports a deprecated NumPy namespace in its legacy test utilities.
    # Suppress only that external warning during import, keeping our warnings visible.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="numpy.core is deprecated.*",
            category=DeprecationWarning,
            module=r"evidently\..*",
        )
        warnings.filterwarnings(
            "ignore",
            message="invalid escape sequence",
            category=SyntaxWarning,
        )
        from evidently import DataDefinition, Dataset, Report
        from evidently.metrics import ValueDrift
        from evidently.telemetry import event_logger

    event_logger.enabled = False  # Reports stay local; no telemetry events.
    definition = DataDefinition(numerical_columns=list(panel.columns))
    report = Report(
        [
            ValueDrift(column=category, method="wasserstein", threshold=thresholds[category])
            for category in panel.columns
        ]
    ).run(
        Dataset.from_pandas(current.reset_index(drop=True), data_definition=definition),
        Dataset.from_pandas(reference.reset_index(drop=True), data_definition=definition),
    )
    scores = {item["config"]["column"]: float(item["value"]) for item in report.dict()["metrics"]}
    columns = [
        DriftColumn(
            category=category,
            score=scores[category],
            threshold=thresholds[category],
            drift=scores[category] >= thresholds[category],
            calibration_windows=counts[category],
        )
        for category in panel.columns
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(destination))
    return DataDrift(
        status="warning" if any(column.drift for column in columns) else "stable",
        message="Изменение распределений продаж — сигнал, не доказательство падения качества.",
        reference_start=reference.index[0].date(),
        reference_end=reference.index[-1].date(),
        current_start=current.index[0].date(),
        current_end=current.index[-1].date(),
        columns=columns,
    )


def quality_drift(
    forecasts: list[ForecastResponse],
    panel: pd.DataFrame,
    relative: float,
    absolute: float,
) -> list[Quality]:
    groups: dict[tuple[str, str, int], dict[date, ForecastResponse]] = defaultdict(dict)
    for forecast in forecasts:
        # Repeated clicks at the same origin must not inflate the sample size.
        groups[(forecast.model_id, forecast.category, forecast.horizon)].setdefault(
            forecast.origin, forecast
        )
    result = []
    for (model_id, category, horizon), by_origin in sorted(groups.items()):
        ordered = sorted(by_origin.values(), key=lambda item: item.origin)
        matured = [
            item
            for item in ordered
            if all(
                pd.Timestamp(point.date) in panel.index and category in panel.columns
                for point in item.forecast
            )
        ]
        row = Quality(
            model_id=model_id,
            category=category,
            horizon=horizon,
            status="waiting",
            pending=len(ordered) - len(matured),
            matured=len(matured),
        )
        if not matured:
            result.append(row)
            continue

        def score(item: ForecastResponse, category: str = category) -> RegressionMetrics:
            return regression_metrics(
                [
                    float(cast(float, panel.at[pd.Timestamp(p.date), category]))
                    for p in item.forecast
                ],
                [point.units for point in item.forecast],
            )

        first = matured[0]
        row.reference_origin = first.origin
        row.reference = score(first)
        row.status = "reference_only"
        eligible = [item for item in matured if item.origin >= first.forecast[-1].date]
        if eligible:
            latest = eligible[-1]
            row.current_origin = latest.origin
            row.current = score(latest)
            if row.reference.wape_pct is None or row.current.wape_pct is None:
                row.status = "undefined"
            else:
                row.threshold_wape = max(
                    row.reference.wape_pct * (1 + relative),
                    row.reference.wape_pct + absolute,
                )
                row.status = "warning" if row.current.wape_pct > row.threshold_wape else "stable"
        result.append(row)
    return result
