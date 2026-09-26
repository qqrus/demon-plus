"""One-process local application service: immutable snapshots and explicit model promotion."""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import cast
from uuid import uuid4

import pandas as pd
import structlog
from pydantic import BaseModel

from demon_plus.api.contracts import (
    ActiveModel,
    ForecastRequest,
    ForecastResponse,
    ImportResponse,
    InventoryRequest,
    InventoryResponse,
    Job,
    Point,
    StatusResponse,
)
from demon_plus.config import Settings
from demon_plus.data.download import FILES, sha256
from demon_plus.data.olist import daily_panel, load_olist
from demon_plus.data.uploads import merge_daily_csv
from demon_plus.errors import DataError
from demon_plus.evaluation.metrics import regression_metrics
from demon_plus.features.demand import feature_rows, training_rows
from demon_plus.inventory.risk import calculate_risk
from demon_plus.models.baseline import BaselineName, forecast
from demon_plus.models.catboost import (
    fit_model,
    load_model,
    predict_rows,
    predict_saved,
    save_model,
)
from demon_plus.storage.store import Store


class Snapshot(BaseModel):
    dataset_id: str
    model: ActiveModel
    initial_run: str


class Service:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = Store(settings.app_db)
        self.lock = threading.RLock()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demon-retrain")
        self.logger = structlog.get_logger()
        for job in self.jobs():
            if job.status in ("queued", "running"):
                job.status = "failed"
                job.message = "Задача прервана перезапуском сервиса. Можно запустить снова."
                self.store.save_job(job.job_id, job.model_dump_json())

    def close(self) -> None:
        self.worker.shutdown(wait=True)
        self.store.engine.dispose()

    def bootstrap(self) -> None:
        if self.store.get_state("snapshot"):
            return
        root = self.settings.artifacts_dir / "training"
        if not (root / "latest.json").exists():
            raise DataError(
                "Модель не найдена. Сначала выполните fetch-data и train, затем перезапустите API."
            )
        run_id = str(json.loads((root / "latest.json").read_text("utf-8"))["run_id"])
        if len(run_id) != 32 or any(char not in "0123456789abcdef" for char in run_id):
            raise DataError("Некорректный идентификатор обученной модели.")
        directory = root / run_id
        selected = json.loads((directory / "selection.json").read_text("utf-8"))
        if {name: sha256(self.settings.data_dir / "raw" / name) for name in FILES[:3]} != selected[
            "input_hashes"
        ]:
            raise DataError(
                "Данные Olist изменились после обучения; нужна согласованная версия модели."
            )
        data = load_olist(self.settings.data_dir / "raw")
        panel = daily_panel(
            data.sales,
            pd.Timestamp(selected["analysis_start"]),
            pd.Timestamp(selected["holdout_cutoff"]),
        )
        panel = panel[selected["categories"]]
        load_model(
            directory / "model"
        )  # Fail before publishing a snapshot if the artifact is corrupt.
        model = ActiveModel(
            model_id=run_id,
            name=selected["selected_model"],
            directory=str((directory / "model").resolve()),
            training_cutoff=selected["holdout_cutoff"],
        )
        self.initialize(panel, model, directory.resolve())

    def initialize(self, panel: pd.DataFrame, model: ActiveModel, initial_run: Path) -> None:
        dataset_id = self._save_dataset(panel)
        snapshot = Snapshot(dataset_id=dataset_id, model=model, initial_run=str(initial_run))
        self.store.set_state("snapshot", snapshot.model_dump_json())

    def snapshot(self) -> Snapshot:
        raw = self.store.get_state("snapshot")
        if raw is None:
            raise DataError(
                "Сервис ещё не подготовлен: выполните fetch-data и train, затем перезапустите API."
            )
        return Snapshot.model_validate_json(raw)

    def _save_dataset(self, panel: pd.DataFrame) -> str:
        payload = (
            panel.rename_axis("date")
            .to_csv(date_format="%Y-%m-%d", lineterminator="\n")
            .encode("utf-8")
        )
        dataset_id = hashlib.sha256(payload).hexdigest()
        directory = self.settings.data_dir / "versions"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{dataset_id}.csv"
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(path)
        self.store.add_dataset(dataset_id, path.resolve())
        return dataset_id

    def panel(self, dataset_id: str) -> pd.DataFrame:
        path = self.store.dataset_path(dataset_id)
        if sha256(path) != dataset_id:
            raise DataError("Контрольная сумма версии продаж не совпадает.")
        return pd.read_csv(path, index_col="date", parse_dates=["date"]).astype(float)

    def status(self) -> StatusResponse:
        with self.lock:
            snapshot = self.snapshot()
            panel = self.panel(snapshot.dataset_id)
            return StatusResponse(
                dataset_id=snapshot.dataset_id,
                categories=list(panel.columns),
                first_date=panel.index[0].date(),
                last_date=panel.index[-1].date(),
                model_id=snapshot.model.model_id,
                model_name=snapshot.model.name,
                training_cutoff=snapshot.model.training_cutoff,
            )

    def _predict(self, snapshot: Snapshot, panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if snapshot.model.name in ("mean_28", "seasonal_naive_7"):
            result = forecast(panel, cast(BaselineName, snapshot.model.name), horizon)
            return result.reset_index().melt("date", var_name="category", value_name="prediction")
        return predict_saved(Path(snapshot.model.directory), panel, horizon)

    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        with self.lock:
            snapshot = self.snapshot()
            panel = self.panel(snapshot.dataset_id)
            if request.category not in panel.columns:
                raise DataError("Неизвестная категория.")
            prediction = self._predict(snapshot, panel, request.horizon)
            selected = prediction.loc[prediction["category"] == request.category]
            points = [
                Point(date=day.date(), units=float(value))
                for day, value in zip(selected["date"], selected["prediction"], strict=True)
            ]
            history = panel[request.category].tail(60)
            response = ForecastResponse(
                forecast_id=uuid4().hex,
                category=request.category,
                horizon=request.horizon,
                origin=panel.index[-1].date(),
                dataset_id=snapshot.dataset_id,
                model_id=snapshot.model.model_id,
                model_name=snapshot.model.name,
                history=[
                    Point(date=pd.Timestamp(day).date(), units=float(value))
                    for day, value in zip(history.index, history.to_numpy(), strict=True)
                ],
                forecast=points,
                total=sum(point.units for point in points),
            )
            self.store.save_forecast(
                response.forecast_id,
                snapshot.dataset_id,
                snapshot.model.model_id,
                response.model_dump_json(),
            )
            return response

    def inventory(self, request: InventoryRequest) -> InventoryResponse:
        with self.lock:
            snapshot = self.snapshot()
            panel = self.panel(snapshot.dataset_id)
            categories = [item.category for item in request.items]
            if len(set(categories)) != len(categories) or not set(categories).issubset(
                panel.columns
            ):
                raise DataError("Нужны известные категории без повторений.")
            prediction = self._predict(snapshot, panel, request.horizon)
            origin = panel.index[-1].date()
            risks = []
            for item in request.items:
                rows = prediction.loc[prediction["category"] == item.category]
                points = [
                    Point(date=day.date(), units=float(value))
                    for day, value in zip(rows["date"], rows["prediction"], strict=True)
                ]
                risks.append(calculate_risk(item, points, origin))
            return InventoryResponse(
                origin=origin,
                dataset_id=snapshot.dataset_id,
                model_id=snapshot.model.model_id,
                synthetic=request.synthetic,
                risks=risks,
            )

    def import_sales(self, content: bytes) -> ImportResponse:
        with self.lock:
            snapshot = self.snapshot()
            old = self.panel(snapshot.dataset_id)
            new = merge_daily_csv(content, old, self.settings.max_upload_bytes)
            dataset_id = self._save_dataset(new)
            snapshot.dataset_id = dataset_id
            self.store.set_state("snapshot", snapshot.model_dump_json())
            self.logger.info(
                "sales_imported", dataset_id=dataset_id, added_days=len(new) - len(old)
            )
            return ImportResponse(
                dataset_id=dataset_id,
                added_days=len(new) - len(old),
                unchanged=len(new) == len(old),
                last_date=new.index[-1].date(),
            )

    def template(self) -> bytes:
        status = self.status()
        rows = ["date,category,units"] + [
            f"{status.last_date + timedelta(days=1)},{category},0" for category in status.categories
        ]
        return ("\n".join(rows) + "\n").encode("utf-8")

    def metrics(self) -> dict[str, object]:
        snapshot = self.snapshot()
        result: dict[str, object] = {
            "initial_model_id": Path(snapshot.initial_run).name,
            "active_model_id": snapshot.model.model_id,
            "note": "Исходная историческая оценка; не текущий quality drift.",
        }
        for split in ("validation", "holdout"):
            path = Path(snapshot.initial_run) / f"{split}_metrics.json"
            result[split] = json.loads(path.read_text("utf-8")) if path.exists() else None
        return result

    def jobs(self) -> list[Job]:
        return [Job.model_validate_json(payload) for payload in self.store.job_payloads()]

    def job(self, job_id: str) -> Job:
        found = next((job for job in self.jobs() if job.job_id == job_id), None)
        if found is None:
            raise DataError("Задача не найдена.")
        return found

    def start_retraining(self) -> Job:
        with self.lock:
            if any(job.status in ("queued", "running") for job in self.jobs()):
                raise DataError("Переобучение уже выполняется. Дождитесь завершения.")
            snapshot = self.snapshot()
            panel = self.panel(snapshot.dataset_id)
            cutoff = panel.index[-1] - timedelta(days=30)
            if cutoff.date() < snapshot.model.training_cutoff or len(panel.loc[:cutoff]) < 58:
                raise DataError("Нужно минимум 30 новых дней после обучения текущей модели.")
            job = Job(
                job_id=uuid4().hex,
                status="queued",
                dataset_id=snapshot.dataset_id,
                parent_model_id=snapshot.model.model_id,
                message="Задача поставлена в очередь.",
            )
            self.store.save_job(job.job_id, job.model_dump_json())
            self.worker.submit(self._retrain, job, snapshot, panel)
            return job

    def _retrain(self, job: Job, snapshot: Snapshot, panel: pd.DataFrame) -> None:
        try:
            # Imports keep the forecast API independent of MLflow startup latency.
            import mlflow

            from demon_plus.evaluation.train import setup_tracking

            job.status = "running"
            job.message = "Сравниваем кандидата на последних 30 днях этой версии данных."
            self.store.save_job(job.job_id, job.model_dump_json())
            _, metadata = load_model(Path(snapshot.model.directory))
            cutoff = panel.index[-1] - timedelta(days=30)
            rows = training_rows(panel, cutoff, self.settings.training_stride)
            candidate = fit_model(rows, metadata.depth, metadata.iterations, metadata.seed)
            candidate_predictions = predict_rows(candidate, feature_rows(panel, cutoff))
            current_predictions = self._predict(snapshot, panel.loc[:cutoff], 30)
            baseline_predictions = (
                forecast(panel.loc[:cutoff], "mean_28")
                .reset_index()
                .melt("date", var_name="category", value_name="prediction")
            )

            def score(predictions: pd.DataFrame) -> float:
                truth = [
                    float(cast(float, panel.at[day, category]))
                    for day, category in zip(
                        predictions["date"], predictions["category"], strict=True
                    )
                ]
                value = regression_metrics(
                    truth, predictions["prediction"].to_numpy(dtype=float)
                ).wape_pct
                if value is None:
                    raise DataError(
                        "В окне оценки нет продаж: WAPE не определён, продвижение недоступно."
                    )
                return value

            job.candidate_wape = score(candidate_predictions)
            job.current_wape = score(current_predictions)
            job.baseline_wape = score(baseline_predictions)
            job.eligible = job.candidate_wape <= min(job.current_wape, job.baseline_wape)
            setup_tracking(self.settings)
            with mlflow.start_run(run_name="app-retraining") as run:
                mlflow.set_tags(
                    {
                        "job_id": job.job_id,
                        "dataset_id": job.dataset_id,
                        "parent_model": job.parent_model_id,
                        "evaluation": "rolling operational window, not independent final test",
                    }
                )
                mlflow.log_params(
                    {
                        "depth": metadata.depth,
                        "iterations": metadata.iterations,
                        "seed": metadata.seed,
                    }
                )
                mlflow.log_metrics(
                    {
                        "candidate_wape": job.candidate_wape,
                        "current_wape": job.current_wape,
                        "baseline_wape": job.baseline_wape,
                    }
                )
                final = fit_model(
                    training_rows(panel, panel.index[-1], self.settings.training_stride),
                    metadata.depth,
                    metadata.iterations,
                    metadata.seed,
                )
                directory = self.settings.artifacts_dir / "candidates" / job.job_id
                save_model(
                    final,
                    directory,
                    list(panel.columns),
                    panel.index[-1].date(),
                    metadata.depth,
                    metadata.iterations,
                    metadata.seed,
                )
                mlflow.log_artifacts(str(directory), artifact_path="model")
                job.model = ActiveModel(
                    model_id=str(run.info.run_id),
                    name=f"catboost_d{metadata.depth}",
                    directory=str(directory.resolve()),
                    training_cutoff=panel.index[-1].date(),
                )
            job.status = "completed"
            job.message = (
                "Кандидат готов к применению."
                if job.eligible
                else "Кандидат не прошёл сравнение; текущая модель сохранена."
            )
        except Exception as exc:
            # Background failures are persisted; never expose input rows or credentials in errors.
            job.status = "failed"
            job.message = (
                "Не удалось завершить обучение. "
                "Проверьте данные и доступность локальных артефактов."
            )
            if isinstance(exc, DataError):
                job.message = str(exc)
            self.logger.error("retraining_failed", job_id=job.job_id, error_type=type(exc).__name__)
        self.store.save_job(job.job_id, job.model_dump_json())

    def activate(self, job_id: str) -> ActiveModel:
        with self.lock:
            job = self.job(job_id)
            snapshot = self.snapshot()
            if job.status != "completed" or not job.eligible or job.model is None:
                raise DataError(
                    "Эту модель нельзя применить: кандидат не готов или не прошёл сравнение."
                )
            if (
                snapshot.dataset_id != job.dataset_id
                or snapshot.model.model_id != job.parent_model_id
            ):
                raise DataError("Данные или активная модель изменились. Запустите новое сравнение.")
            load_model(Path(job.model.directory))
            snapshot.model = job.model
            job.status = "activated"
            job.message = "Модель применена."
            self.store.activate(snapshot.model_dump_json(), job.job_id, job.model_dump_json())
            return snapshot.model
