"""Native CatBoost artifacts with explicit feature and cohort metadata."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pydantic import BaseModel

from demon_plus.data.download import sha256
from demon_plus.errors import DataError
from demon_plus.features.demand import FEATURES, feature_rows


class ModelMetadata(BaseModel):
    schema_version: int = 1
    feature_names: list[str] = FEATURES
    categories: list[str]
    training_cutoff: date
    model_sha256: str
    depth: int
    iterations: int
    seed: int


def fit_model(rows: pd.DataFrame, depth: int, iterations: int, seed: int) -> CatBoostRegressor:
    model = CatBoostRegressor(
        depth=depth,
        iterations=iterations,
        learning_rate=0.05,
        loss_function="MAE",
        random_seed=seed,
        thread_count=2,
        cat_features=["category"],
        one_hot_max_size=255,
        has_time=True,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(rows[FEATURES], rows["actual"])
    return model


def predict_rows(model: CatBoostRegressor, rows: pd.DataFrame) -> pd.DataFrame:
    result = rows[["origin", "date", "step", "category"]].copy()
    result["prediction"] = np.maximum(np.asarray(model.predict(rows[FEATURES]), dtype=float), 0)
    return result


def save_model(
    model: CatBoostRegressor,
    directory: Path,
    categories: list[str],
    cutoff: date,
    depth: int,
    iterations: int,
    seed: int,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.cbm"
    model.save_model(str(path))
    metadata = ModelMetadata(
        categories=categories,
        training_cutoff=cutoff,
        model_sha256=sha256(path),
        depth=depth,
        iterations=iterations,
        seed=seed,
    )
    (directory / "metadata.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_model(directory: Path) -> tuple[CatBoostRegressor, ModelMetadata]:
    metadata = ModelMetadata.model_validate_json((directory / "metadata.json").read_text("utf-8"))
    if metadata.schema_version != 1 or metadata.feature_names != FEATURES:
        raise DataError("Saved model uses an unsupported feature schema.")
    if sha256(directory / "model.cbm") != metadata.model_sha256:
        raise DataError("Model checksum mismatch.")
    model = CatBoostRegressor()
    model.load_model(str(directory / "model.cbm"))
    if list(model.feature_names_) != FEATURES:
        raise DataError("Model feature names do not match metadata.")
    return model, metadata


def predict_saved(directory: Path, history: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    model, metadata = load_model(directory)
    if history.empty or list(history.columns) != metadata.categories:
        raise DataError("Prediction history must match the saved category order.")
    origin = pd.Timestamp(history.index[-1])
    if origin.date() < metadata.training_cutoff:
        raise DataError("Cannot use a model trained after the requested forecast origin.")
    return predict_rows(model, feature_rows(history, origin, horizon))
