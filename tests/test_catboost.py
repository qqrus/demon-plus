from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demon_plus.errors import DataError
from demon_plus.features.demand import feature_rows, training_rows
from demon_plus.models.catboost import fit_model, predict_rows, predict_saved, save_model


def test_model_roundtrip_and_origin_guard(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        {"books": np.arange(80, dtype=float) % 7 + 1}, index=pd.date_range("2018-01-01", periods=80)
    )
    cutoff = panel.index[-1]
    rows = training_rows(panel, cutoff)
    model = fit_model(rows, depth=2, iterations=5, seed=42)
    save_model(model, tmp_path, ["books"], cutoff.date(), 2, 5, 42)
    before = predict_rows(model, feature_rows(panel, cutoff))
    after = predict_saved(tmp_path, panel)
    np.testing.assert_allclose(before["prediction"], after["prediction"], atol=1e-12, rtol=0)
    assert after["prediction"].ge(0).all()
    with pytest.raises(DataError, match="trained after"):
        predict_saved(tmp_path, panel.iloc[:-1])
    with pytest.raises(DataError, match="category order"):
        predict_saved(tmp_path, panel.rename(columns={"books": "unknown"}))
    with (tmp_path / "model.cbm").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(DataError, match="checksum"):
        predict_saved(tmp_path, panel)
