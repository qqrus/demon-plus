import numpy as np
import pandas as pd

from demon_plus.features.demand import FEATURES, feature_rows, training_rows


def series(days: int = 120) -> pd.DataFrame:
    return pd.DataFrame(
        {"books": np.arange(days, dtype=float), "toys": np.ones(days)},
        index=pd.date_range("2018-01-01", periods=days),
    )


def test_feature_values_and_calendar() -> None:
    data = series()
    rows = feature_rows(data, data.index[27])
    row = rows.iloc[0]
    assert row["lag_0"] == 27
    assert row["lag_27"] == 0
    assert row["mean_28"] == 13.5
    assert row["mean_7"] == 24
    assert row["date"] == pd.Timestamp("2018-01-29")
    assert row["target_weekday"] == 0
    assert len(rows) == 60
    assert not {"actual", "date", "origin"}.intersection(FEATURES)


def test_features_and_training_labels_ignore_data_after_cutoff() -> None:
    data = series()
    cutoff = data.index[80]
    corrupted = data.copy()
    corrupted.loc[corrupted.index > cutoff] = np.nan
    pd.testing.assert_frame_equal(feature_rows(data, cutoff), feature_rows(corrupted, cutoff))
    before = training_rows(data, cutoff)
    after = training_rows(corrupted, cutoff)
    pd.testing.assert_frame_equal(before, after)
    assert before["date"].max() <= cutoff
    assert (before["date"] > before["origin"]).all()
    assert before.groupby(["origin", "category"]).size().eq(30).all()


def test_training_features_match_inference_for_same_origin() -> None:
    data = series()
    rows = training_rows(data, data.index[-1])
    origin = rows["origin"].iloc[0]
    historical = rows.loc[rows["origin"] == origin, FEATURES].reset_index(drop=True)
    inference = feature_rows(data.loc[:origin], origin)[FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(historical, inference)
