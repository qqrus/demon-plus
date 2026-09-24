import json
from datetime import date
from pathlib import Path

import pandas as pd

from demon_plus.config import Settings
from demon_plus.data.download import FILES
from demon_plus.evaluation.run import run_baselines


def test_pipeline_writes_reproducible_outputs_without_scoring_holdout(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    dates = pd.date_range("2018-01-01", periods=130)
    order_ids = [f"o{number}" for number in range(len(dates))]
    pd.DataFrame(
        {
            "order_id": order_ids,
            "order_purchase_timestamp": dates.strftime("%Y-%m-%d 12:00:00"),
            "order_status": "delivered",
        }
    ).to_csv(raw / FILES[0], index=False)
    pd.DataFrame({"order_id": order_ids, "order_item_id": 1, "product_id": "p"}).to_csv(
        raw / FILES[1], index=False
    )
    pd.DataFrame({"product_id": ["p"], "product_category_name": ["books"]}).to_csv(
        raw / FILES[2], index=False
    )
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        analysis_start=date(2018, 1, 1),
        validation_cutoffs=(date(2018, 1, 28),),
        holdout_cutoff=date(2018, 2, 27),
        top_categories=1,
        min_active_days=20,
    )
    report = run_baselines(settings)
    output = report.parent
    assert report.exists()
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["holdout_scored"] is False
    predictions = pd.read_csv(output / "predictions.csv")
    assert predictions["date"].max() == "2018-02-27"
    assert len(predictions) == 60
    metrics = (output / "metrics.json").read_bytes()
    assert (output / "eda.png").stat().st_size > 1000
    assert (output / "baseline_forecast.png").stat().st_size > 1000
    run_baselines(settings)
    assert (output / "metrics.json").read_bytes() == metrics
