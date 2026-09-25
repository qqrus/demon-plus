import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from demon_plus.config import Settings
from demon_plus.data.download import FILES
from demon_plus.errors import DataError
from demon_plus.evaluation.train import evaluate_holdout, train_candidates


def test_tracking_selection_and_frozen_holdout(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    dates = pd.date_range("2018-01-01", periods=150)
    # Varying units avoid a constant-target CatBoost training failure.
    units = [1 + number % 3 for number in range(len(dates))]
    pd.DataFrame(
        {
            "order_id": [str(i) for i in range(len(dates))],
            "order_purchase_timestamp": dates.strftime("%Y-%m-%d 12:00:00"),
            "order_status": "delivered",
        }
    ).to_csv(raw / FILES[0], index=False)
    pd.DataFrame(
        [
            {"order_id": str(i), "order_item_id": item + 1, "product_id": "p"}
            for i, count in enumerate(units)
            for item in range(count)
        ]
    ).to_csv(raw / FILES[1], index=False)
    pd.DataFrame({"product_id": ["p"], "product_category_name": ["books"]}).to_csv(
        raw / FILES[2], index=False
    )
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        mlflow_db=tmp_path / "tracking.db",
        analysis_start=date(2018, 1, 1),
        validation_cutoffs=(date(2018, 3, 1),),
        holdout_cutoff=date(2018, 3, 31),
        top_categories=1,
        min_active_days=10,
        catboost_depths=(2,),
        catboost_iterations=5,
    )
    selected_path = train_candidates(settings)
    selected = json.loads(selected_path.read_text())
    client = MlflowClient()
    run = client.get_run(selected["run_id"])
    assert run.info.status == "FINISHED"
    assert "catboost_d2.h30.daily.mae" in run.data.metrics
    assert client.list_artifacts(selected["run_id"], "training/model")
    predictions = pd.read_csv(selected_path.parent / "validation_predictions.csv")
    assert predictions["date"].max().startswith("2018-03-31")
    assert not (selected_path.parent / "holdout_metrics.json").exists()
    frozen = selected_path.read_bytes()
    report = evaluate_holdout(settings)
    assert selected_path.read_bytes() == frozen
    saved_result = report.read_bytes()
    assert evaluate_holdout(settings).read_bytes() == saved_result
    holdout = pd.read_csv(report.parent / "holdout_predictions.csv")
    assert holdout["date"].min().startswith("2018-04-01")
    assert holdout["date"].max().startswith("2018-04-30")
    with (raw / FILES[0]).open("a") as stream:
        stream.write("\n")
    with pytest.raises(DataError, match="Data changed"):
        evaluate_holdout(settings)
