"""Export the historical holdout as a CSV upload demonstration, never as a new test."""

import json
from pathlib import Path

import pandas as pd

from demon_plus.config import Settings
from demon_plus.data.olist import daily_panel, load_olist


def export_demo(settings: Settings) -> Path:
    root = settings.artifacts_dir / "training"
    run_id = json.loads((root / "latest.json").read_text("utf-8"))["run_id"]
    selected = json.loads((root / run_id / "selection.json").read_text("utf-8"))
    start = pd.Timestamp(selected["holdout_cutoff"]) + pd.Timedelta(days=1)
    sales = load_olist(settings.data_dir / "raw").sales
    panel = daily_panel(sales, start, start + pd.Timedelta(days=29))[selected["categories"]]
    rows = (
        panel.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="category", value_name="units")
    )
    path = settings.data_dir / "demo" / "sales-update.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


if __name__ == "__main__":
    print(export_demo(Settings()))
