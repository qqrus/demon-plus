from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from demon_plus.errors import DataError
from demon_plus.evaluation.backtest import backtest, score_predictions
from demon_plus.evaluation.metrics import regression_metrics
from demon_plus.models.baseline import BASELINES, forecast, select_categories


def panel(days: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {"books": np.arange(1, days + 1, dtype=float), "toys": np.ones(days)},
        index=pd.date_range("2018-01-01", periods=days, name="date"),
    )


@pytest.mark.parametrize("horizon", [7, 14, 30])
def test_baselines_forecast_whole_horizon_from_history(horizon: int) -> None:
    history = panel(28)
    repeated = forecast(history, "seasonal_naive_7", horizon)
    mean = forecast(history, "mean_28", horizon)
    assert len(repeated) == horizon
    assert repeated.index[0] == pd.Timestamp("2018-01-29")
    assert repeated.index[-1] == history.index[-1] + timedelta(days=horizon)
    np.testing.assert_array_equal(repeated["books"], np.tile(np.arange(22, 29), 5)[:horizon])
    np.testing.assert_array_equal(mean["books"], np.full(horizon, 14.5))


def test_future_actuals_do_not_change_current_origin_forecasts_or_cohort() -> None:
    original = panel()
    cutoff = pd.Timestamp("2018-02-01")
    changed = original.copy()
    changed.loc[changed.index > cutoff, "toys"] = 1_000_000
    categories = select_categories(original, cutoff, 1, 20)
    assert categories == ["books"]
    assert select_categories(changed, cutoff, 1, 20) == categories
    before = backtest(original, [cutoff], ["books", "toys"])
    after = backtest(changed, [cutoff], ["books", "toys"])
    pd.testing.assert_series_equal(before["prediction"], after["prediction"])
    assert not before["actual"].equals(after["actual"])


def test_category_eligibility_ignores_future_only_category() -> None:
    data = panel()
    data["new"] = 0.0
    cutoff = pd.Timestamp("2018-02-01")
    data.loc[data.index > cutoff, "new"] = 10_000
    assert select_categories(data, cutoff, 2, 20) == ["books", "toys"]


def test_missing_validation_actuals_fail_instead_of_becoming_zero() -> None:
    with pytest.raises(DataError, match="actual sales"):
        backtest(panel(40), [pd.Timestamp("2018-01-28")], ["books"])


@pytest.mark.parametrize("problem", ["gap", "nan", "negative", "short"])
def test_invalid_history_rejected(problem: str) -> None:
    history = panel(35)
    if problem == "gap":
        history = history.drop(history.index[10])
    elif problem == "nan":
        history.iloc[0, 0] = np.nan
    elif problem == "negative":
        history.iloc[0, 0] = -1
    else:
        history = history.head(27)
    with pytest.raises(DataError):
        forecast(history, "mean_28")


def test_metric_values_and_zero_denominator() -> None:
    scores = regression_metrics([0.0, 4.0], [1.0, 1.0])
    assert scores.mae == 2.0
    assert scores.rmse == pytest.approx(np.sqrt(5))
    assert scores.wape_pct == 100.0
    zero = regression_metrics([0.0, 0.0], [1.0, 0.0])
    assert zero.wape_pct is None
    assert zero.mae == 0.5


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [([], []), ([1.0], [1.0, 2.0]), ([np.nan], [1.0]), ([1.0], [-1.0])],
)
def test_metrics_reject_invalid_arrays(actual: list[float], predicted: list[float]) -> None:
    with pytest.raises(DataError):
        regression_metrics(actual, predicted)


def test_scoring_uses_expected_windows_and_inventory_totals() -> None:
    data = panel()
    data[:] = 2.0
    predictions = backtest(data, [pd.Timestamp("2018-01-28")], ["books", "toys"])
    assert set(predictions["model"]) == set(BASELINES)
    assert len(predictions) == 2 * 2 * 30
    scores = score_predictions(predictions)
    assert len(scores.aggregate) == 6
    for score in scores.aggregate:
        assert score.n_daily == 2 * score.horizon
        assert score.daily.mae == 0.0
        assert score.horizon_totals.mae == 0.0
    assert len(scores.per_category) == 12


def test_inventory_totals_are_not_daily_errors() -> None:
    predictions = backtest(panel(), [pd.Timestamp("2018-01-28")], ["books"])
    scores = score_predictions(predictions)
    mean_30 = next(s for s in scores.aggregate if s.model == "mean_28" and s.horizon == 30)
    assert mean_30.daily.mae == 29.0
    assert mean_30.horizon_totals.mae == 870.0
