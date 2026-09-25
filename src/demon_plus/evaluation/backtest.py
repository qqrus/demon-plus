"""Rolling origins with a fixed cohort and no random split."""

from datetime import timedelta

import pandas as pd
from pydantic import BaseModel

from demon_plus.errors import DataError
from demon_plus.evaluation.metrics import RegressionMetrics, regression_metrics
from demon_plus.models.baseline import BASELINES, forecast

HORIZONS = (7, 14, 30)


class AggregateScore(BaseModel):
    model: str
    horizon: int
    n_daily: int
    daily: RegressionMetrics
    horizon_totals: RegressionMetrics


class EvaluationReport(BaseModel):
    aggregate: list[AggregateScore]
    per_category: list[dict[str, object]]
    per_origin: list[dict[str, object]]


def backtest(
    panel: pd.DataFrame, cutoffs: list[pd.Timestamp], categories: list[str]
) -> pd.DataFrame:
    if not cutoffs or not categories:
        raise DataError("Backtest requires origins and categories.")
    frames: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        if cutoff not in panel.index:
            raise DataError("Forecast origin is missing from the daily panel.")
        history = panel.loc[:cutoff, categories]
        dates = pd.date_range(cutoff + timedelta(days=1), periods=30)
        truth = panel.reindex(dates)[categories].rename_axis("date")
        if truth.isna().any().any():
            raise DataError("Validation needs actual sales for every forecast day and category.")
        actual = truth.reset_index().melt("date", var_name="category", value_name="actual")
        for model in BASELINES:
            predicted = (
                forecast(history, model)
                .reset_index()
                .melt("date", var_name="category", value_name="prediction")
            )
            frame = predicted.merge(actual, on=["date", "category"], validate="one_to_one")
            frame["origin"] = cutoff
            frame["step"] = (frame["date"] - cutoff).dt.days
            frame["model"] = model
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def score_predictions(predictions: pd.DataFrame) -> EvaluationReport:
    """Pool daily errors; score horizon totals separately for inventory use."""
    aggregate: list[AggregateScore] = []
    per_category: list[dict[str, object]] = []
    per_origin: list[dict[str, object]] = []
    for model in sorted(predictions["model"].unique()):
        for horizon in HORIZONS:
            frame = predictions.loc[
                (predictions["model"] == model) & (predictions["step"] <= horizon)
            ]
            daily = regression_metrics(
                frame["actual"].to_numpy(dtype=float), frame["prediction"].to_numpy(dtype=float)
            )
            totals = frame.groupby(["origin", "category"])[["actual", "prediction"]].sum()
            total_metrics = regression_metrics(
                totals["actual"].to_numpy(dtype=float), totals["prediction"].to_numpy(dtype=float)
            )
            aggregate.append(
                AggregateScore(
                    model=model,
                    horizon=horizon,
                    n_daily=len(frame),
                    daily=daily,
                    horizon_totals=total_metrics,
                )
            )
            for dimension, output in [("category", per_category), ("origin", per_origin)]:
                for key, group in frame.groupby(dimension):
                    metrics = regression_metrics(
                        group["actual"].to_numpy(dtype=float),
                        group["prediction"].to_numpy(dtype=float),
                    )
                    output.append(
                        {
                            "model": model,
                            "horizon": horizon,
                            dimension: str(key),
                            **metrics.model_dump(),
                        }
                    )
    return EvaluationReport(aggregate=aggregate, per_category=per_category, per_origin=per_origin)
