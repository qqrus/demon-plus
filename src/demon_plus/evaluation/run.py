"""Reproducible local EDA and baseline evaluation artifacts."""

import hashlib
import json
import platform
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from demon_plus.config import Settings
from demon_plus.data.olist import daily_panel, load_olist
from demon_plus.evaluation.backtest import EvaluationReport, backtest, score_predictions
from demon_plus.models.baseline import BASELINES, select_categories


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _plots(panel: pd.DataFrame, predictions: pd.DataFrame, directory: Path) -> None:
    first_origin = predictions["origin"].min()
    history = panel.loc[:first_origin]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    axes[0].plot(history.index, history.sum(axis=1).rolling(7).mean(), color="#2364AA")
    axes[0].set(title="Selected categories: 7-day mean (training only)", ylabel="Units / day")
    zero_share = (history == 0).mean().sort_values() * 100
    axes[1].barh(zero_share.index, zero_share.values, color="#299D8F")
    axes[1].set(title="Days with zero sales (first training window)", xlabel="% of observed days")
    axes[0].tick_params(axis="x", rotation=30)
    fig.savefig(directory / "eda.png", dpi=160)
    plt.close(fig)

    category = str(panel.columns[0])  # Largest category at the first origin, not cherry-picked.
    origin = predictions["origin"].max()
    frame = predictions.loc[
        (predictions["category"] == category) & (predictions["origin"] == origin)
    ]
    fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
    recent = panel.loc[origin - timedelta(days=55) : origin, category]
    ax.plot(recent.index, recent.values, label="Observed history", color="#334155")
    actual = frame.loc[frame["model"] == BASELINES[0]]
    ax.plot(actual["date"], actual["actual"], label="Validation actual", color="#111827")
    for model, color in zip(BASELINES, ["#DA6A31", "#237DAD"], strict=True):
        model_frame = frame.loc[frame["model"] == model]
        ax.plot(model_frame["date"], model_frame["prediction"], "--", label=model, color=color)
    ax.axvline(origin, color="#64748B", linestyle=":", label="Forecast origin")
    ax.set(title=f"{category}: fixed 30-day forecast from {origin.date()}", ylabel="Units / day")
    ax.legend(ncol=2)
    fig.savefig(directory / "baseline_forecast.png", dpi=160)
    plt.close(fig)


def _markdown(report: EvaluationReport) -> str:
    lines = [
        "# Baseline validation results",
        "",
        "Pooled daily errors across fixed categories and validation origins. WAPE is a percentage.",
        "The final holdout has not been scored.",
        "",
        "| Model | Horizon | Daily MAE | Daily RMSE | Daily WAPE % | Total WAPE % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for score in report.aggregate:
        daily = score.daily
        wape = f"{daily.wape_pct:.2f}" if daily.wape_pct is not None else "N/A"
        total_wape = score.horizon_totals.wape_pct
        total = f"{total_wape:.2f}" if total_wape is not None else "N/A"
        lines.append(
            f"| {score.model} | {score.horizon} | {daily.mae:.3f} | {daily.rmse:.3f} | "
            f"{wape} | {total} |"
        )
    lines += ["", "![Training EDA](eda.png)", "", "![Forecast example](baseline_forecast.png)", ""]
    return "\n".join(lines)


def run_baselines(settings: Settings) -> Path:
    """Do not score or create a panel containing the reserved holdout targets."""
    dataset = load_olist(settings.data_dir / "raw")
    cutoffs = [pd.Timestamp(value) for value in settings.validation_cutoffs]
    validation_end = cutoffs[-1] + timedelta(days=30)
    panel = daily_panel(dataset.sales, pd.Timestamp(settings.analysis_start), validation_end)
    categories = select_categories(
        panel, cutoffs[0], settings.top_categories, settings.min_active_days
    )
    selected = panel[categories]
    predictions = backtest(selected, cutoffs, categories)
    scores = score_predictions(predictions)
    output = settings.artifacts_dir / "baseline"
    output.mkdir(parents=True, exist_ok=True)
    processed = settings.data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    selected.to_csv(processed / "category_daily.csv", date_format="%Y-%m-%d")
    predictions.to_csv(output / "predictions.csv", index=False, date_format="%Y-%m-%d")
    (output / "metrics.json").write_text(scores.model_dump_json(indent=2) + "\n", encoding="utf-8")

    first_train = selected.loc[: cutoffs[0]]
    audit = dict(dataset.audit)
    audit["selected_categories"] = categories
    audit["selected_training_units"] = float(first_train.to_numpy().sum())
    audit["selected_training_share_pct"] = float(
        first_train.to_numpy().sum() / panel.loc[: cutoffs[0]].to_numpy().sum() * 100
    )
    audit["training_zero_day_pct"] = {
        str(key): float(value) for key, value in ((first_train == 0).mean() * 100).items()
    }
    _write_json(output / "eda.json", audit)
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode())
        digest.update(path.read_bytes())
    protocol: dict[str, object] = {
        "analysis_start": settings.analysis_start.isoformat(),
        "validation_cutoffs": [value.isoformat() for value in settings.validation_cutoffs],
        "holdout_cutoff": settings.holdout_cutoff.isoformat(),
        "holdout_end": (settings.holdout_cutoff + timedelta(days=30)).isoformat(),
        "holdout_scored": False,
        "forecast_horizons": [7, 14, 30],
        "top_categories": settings.top_categories,
        "min_active_days": settings.min_active_days,
        "category_selection_cutoff": str(cutoffs[0].date()),
        "categories": categories,
        "model_selection_metric": "pooled daily WAPE at horizon 30 on validation only",
        "source_code_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ["demon-plus", "pandas", "numpy", "scikit-learn"]
        },
        "input_sha256": dataset.audit["input_sha256"],
        "zero_fill_assumption": "Missing category-day means zero within the accepted store window.",
    }
    _write_json(output / "protocol.json", protocol)
    _plots(selected, predictions, output)
    report_path = output / "report.md"
    report_path.write_text(_markdown(scores), encoding="utf-8")
    return report_path
