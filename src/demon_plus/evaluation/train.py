"""Select on validation, freeze artifacts, then evaluate holdout separately."""

import json
from datetime import date, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import cast

import mlflow
import numpy as np
import pandas as pd
import structlog
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

from demon_plus.config import Settings
from demon_plus.data.download import FILES, sha256
from demon_plus.data.olist import daily_panel, load_olist
from demon_plus.errors import DataError
from demon_plus.evaluation.backtest import backtest, score_predictions
from demon_plus.evaluation.model_report import write_report
from demon_plus.features.demand import FEATURES, feature_rows, training_rows
from demon_plus.models.baseline import select_categories
from demon_plus.models.catboost import fit_model, predict_rows, predict_saved, save_model


class Selection(BaseModel):
    run_id: str
    categories: list[str]
    analysis_start: date
    holdout_cutoff: date
    best_catboost: str
    selected_model: str
    selection_metric: str = "validation pooled daily WAPE, horizon=30"
    input_hashes: dict[str, str]
    settings: dict[str, object]
    package_versions: dict[str, str]
    source_hashes: dict[str, str]


def setup_tracking(settings: Settings) -> None:
    settings.mlflow_db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{settings.mlflow_db.resolve().as_posix()}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("demon-plus")
    if experiment is None:
        directory = settings.artifacts_dir.resolve() / "mlflow-artifacts"
        directory.mkdir(parents=True, exist_ok=True)
        client.create_experiment("demon-plus", artifact_location=directory.as_uri())
    mlflow.set_experiment("demon-plus")


def _with_actual(predicted: pd.DataFrame, panel: pd.DataFrame, name: str) -> pd.DataFrame:
    result = predicted.copy()
    result["actual"] = [
        float(cast(float, panel.at[target, category]))
        for target, category in zip(result["date"], result["category"], strict=True)
    ]
    result["model"] = name
    return result


def log_scores(predictions: pd.DataFrame) -> None:
    for score in score_predictions(predictions).aggregate:
        for scope, metrics in [("daily", score.daily), ("totals", score.horizon_totals)]:
            for metric, value in metrics.model_dump().items():
                if value is not None:
                    mlflow.log_metric(f"{score.model}.h{score.horizon}.{scope}.{metric}", value)


def train_candidates(settings: Settings) -> Path:
    dataset = load_olist(settings.data_dir / "raw")
    cutoffs = [pd.Timestamp(value) for value in settings.validation_cutoffs]
    panel = daily_panel(
        dataset.sales, pd.Timestamp(settings.analysis_start), pd.Timestamp(settings.holdout_cutoff)
    )
    categories = select_categories(
        panel, cutoffs[0], settings.top_categories, settings.min_active_days
    )
    panel = panel[categories]
    setup_tracking(settings)
    logger = structlog.get_logger()
    with mlflow.start_run(run_name="validation-selection") as run:
        output = settings.artifacts_dir / "training" / str(run.info.run_id)
        output.mkdir(parents=True, exist_ok=False)
        mlflow.log_params(
            {
                "iterations": settings.catboost_iterations,
                "depths": str(settings.catboost_depths),
                "seed": settings.random_seed,
                "training_stride": settings.training_stride,
                "loss": "MAE",
                "learning_rate": 0.05,
                "one_hot_max_size": 255,
                "validation_cutoffs": str(settings.validation_cutoffs),
                "holdout_cutoff": str(settings.holdout_cutoff),
                "catboost_version": version("catboost"),
                "mlflow_version": version("mlflow"),
            }
        )
        baseline = backtest(panel, cutoffs, categories)
        frames = [baseline]
        with mlflow.start_run(run_name="baselines-validation", nested=True):
            log_scores(baseline)
        for cutoff in cutoffs:
            rows = training_rows(panel, cutoff, settings.training_stride)
            features = feature_rows(panel, cutoff)
            for depth in settings.catboost_depths:
                name = f"catboost_d{depth}"
                logger.info("training_fold", model=name, cutoff=str(cutoff.date()), rows=len(rows))
                with mlflow.start_run(run_name=f"{name}-{cutoff.date()}", nested=True):
                    mlflow.log_params(
                        {"depth": depth, "cutoff": str(cutoff.date()), "rows": len(rows)}
                    )
                    model = fit_model(
                        rows, depth, settings.catboost_iterations, settings.random_seed
                    )
                    prediction = _with_actual(predict_rows(model, features), panel, name)
                    log_scores(prediction)
                    frames.append(prediction)
        predictions = pd.concat(frames, ignore_index=True)
        report = score_predictions(predictions)
        ranking = sorted(
            [score for score in report.aggregate if score.horizon == 30],
            key=lambda score: (
                score.daily.wape_pct if score.daily.wape_pct is not None else float("inf")
            ),
        )
        best_catboost = next(
            score.model for score in ranking if score.model.startswith("catboost_")
        )
        depth = int(best_catboost.removeprefix("catboost_d"))
        # Selection is finished before the final model is trained, without accessing July targets.
        selection = Selection(
            run_id=run.info.run_id,
            categories=categories,
            analysis_start=settings.analysis_start,
            holdout_cutoff=settings.holdout_cutoff,
            best_catboost=best_catboost,
            selected_model=ranking[0].model,
            input_hashes={name: sha256(settings.data_dir / "raw" / name) for name in FILES[:3]},
            settings=settings.model_dump(mode="json"),
            package_versions={
                name: version(name)
                for name in ["catboost", "mlflow", "numpy", "pandas", "scikit-learn"]
            },
            source_hashes={
                str(path.relative_to(Path(__file__).resolve().parents[1])): sha256(path)
                for path in Path(__file__).resolve().parents[1].rglob("*.py")
            },
        )
        (output / "selection.json").write_text(
            selection.model_dump_json(indent=2), encoding="utf-8"
        )
        (output / "validation_metrics.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        predictions.to_csv(output / "validation_predictions.csv", index=False)
        cutoff = pd.Timestamp(settings.holdout_cutoff)
        logger.info("training_final", model=best_catboost, cutoff=str(cutoff.date()))
        rows = training_rows(panel, cutoff, settings.training_stride)
        with mlflow.start_run(run_name="final-model", nested=True):
            mlflow.log_params(
                {"depth": depth, "rows": len(rows), "training_cutoff": str(cutoff.date())}
            )
            model = fit_model(rows, depth, settings.catboost_iterations, settings.random_seed)
            save_model(
                model,
                output / "model",
                categories,
                settings.holdout_cutoff,
                depth,
                settings.catboost_iterations,
                settings.random_seed,
            )
            expected = predict_rows(model, feature_rows(panel, cutoff))
            loaded = predict_saved(output / "model", panel)
            np.testing.assert_allclose(
                expected["prediction"], loaded["prediction"], rtol=0, atol=1e-12
            )
            importance = pd.DataFrame(
                {"feature": FEATURES, "importance": model.feature_importances_}
            )
            importance.sort_values("importance", ascending=False).to_csv(
                output / "feature_importance.csv", index=False
            )
            mlflow.log_artifacts(str(output / "model"), artifact_path="model")
        mlflow.set_tags({"selected_model": selection.selected_model, "holdout_used": "false"})
        log_scores(predictions)
        mlflow.log_artifacts(str(output), artifact_path="training")
    (settings.artifacts_dir / "training" / "latest.json").write_text(
        json.dumps({"run_id": selection.run_id}), encoding="utf-8"
    )
    return output / "selection.json"


def evaluate_holdout(settings: Settings) -> Path:
    root = settings.artifacts_dir / "training"
    if not (root / "latest.json").exists():
        raise DataError("No trained model found. Run demon-plus train first.")
    pointer = json.loads((root / "latest.json").read_text("utf-8"))
    run_id = str(pointer["run_id"])
    if len(run_id) != 32 or any(char not in "0123456789abcdef" for char in run_id):
        raise DataError("Invalid training run ID.")
    output = root / run_id
    selection = Selection.model_validate_json((output / "selection.json").read_text("utf-8"))
    hashes = {name: sha256(settings.data_dir / "raw" / name) for name in FILES[:3]}
    if hashes != selection.input_hashes:
        raise DataError(
            "Data changed since model selection; holdout comparison is not reproducible."
        )
    report_path = output / "holdout_metrics.json"
    if report_path.exists():
        return report_path  # Explicitly reuse the recorded evaluation; never tune here.
    dataset = load_olist(settings.data_dir / "raw")
    cutoff = pd.Timestamp(selection.holdout_cutoff)
    panel = daily_panel(
        dataset.sales, pd.Timestamp(selection.analysis_start), cutoff + timedelta(days=30)
    )
    panel = panel[selection.categories]
    baseline = backtest(panel, [cutoff], selection.categories)
    predicted = predict_saved(output / "model", panel.loc[:cutoff])
    catboost = _with_actual(predicted, panel, selection.best_catboost)
    predictions = pd.concat([baseline, catboost], ignore_index=True)
    report = score_predictions(predictions)
    setup_tracking(settings)
    with mlflow.start_run(run_name="holdout-evaluation"):
        mlflow.set_tags(
            {
                "selection_run_id": run_id,
                "selected_model": selection.selected_model,
                "split": "holdout",
                "tuning_allowed": "false",
            }
        )
        log_scores(predictions)
        predictions.to_csv(output / "holdout_predictions.csv", index=False)
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        write_report(output, panel, predictions)
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(output / "holdout_predictions.csv"))
        mlflow.log_artifact(str(output / "report.md"))
        mlflow.log_artifact(str(output / "holdout_forecast.png"))
    return report_path
