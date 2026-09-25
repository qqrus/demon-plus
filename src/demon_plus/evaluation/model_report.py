"""Static artifacts for reviewing validation and the frozen holdout result."""

from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from demon_plus.evaluation.backtest import EvaluationReport


def write_report(output: Path, panel: pd.DataFrame, predictions: pd.DataFrame) -> None:
    lines = [
        "# CatBoost comparison",
        "",
        "Selection used validation only; holdout was evaluated afterwards.",
        "",
    ]
    for split in ("validation", "holdout"):
        report = EvaluationReport.model_validate_json(
            (output / f"{split}_metrics.json").read_text("utf-8")
        )
        lines += [
            f"## {split}",
            "",
            "| Model | Days | MAE | RMSE | WAPE % | Horizon-total WAPE % |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for score in report.aggregate:
            wape = score.daily.wape_pct
            total = score.horizon_totals.wape_pct
            daily_text = f"{wape:.2f}" if wape is not None else "N/A"
            total_text = f"{total:.2f}" if total is not None else "N/A"
            lines.append(
                f"| {score.model} | {score.horizon} | {score.daily.mae:.3f} | "
                f"{score.daily.rmse:.3f} | {daily_text} | {total_text} |"
            )
        lines.append("")
    category = str(panel.columns[0])
    origin = predictions["origin"].iloc[0]
    frame = predictions.loc[predictions["category"] == category]
    fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
    history = panel.loc[origin - timedelta(days=55) : origin, category]
    ax.plot(history.index, history.values, label="Observed history", color="#64748B")
    truth = frame.loc[frame["model"] == "mean_28"]
    ax.plot(truth["date"], truth["actual"], label="Holdout actual", color="#172033")
    for name, group in frame.groupby("model"):
        if name == "seasonal_naive_7":
            continue
        ax.plot(group["date"], group["prediction"], "--", label=str(name))
    ax.axvline(origin, color="#64748B", linestyle=":")
    ax.set(
        title=f"{category}: untouched July holdout, origin {origin.date()}", ylabel="Units / day"
    )
    ax.legend(ncol=2)
    fig.savefig(output / "holdout_forecast.png", dpi=160)
    plt.close(fig)
    lines += ["![Holdout forecast](holdout_forecast.png)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
