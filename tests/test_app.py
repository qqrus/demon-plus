from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import demon_plus.ui.app as ui_module
from demon_plus.api.app import create_app
from demon_plus.api.contracts import ActiveModel, ForecastRequest, Job
from demon_plus.config import Settings
from demon_plus.errors import DataError
from demon_plus.features.demand import training_rows
from demon_plus.models.catboost import fit_model, save_model
from demon_plus.service import Service


@pytest.fixture
def service(tmp_path: Path) -> Iterator[Service]:
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        app_db=tmp_path / "app.db",
        mlflow_db=tmp_path / "mlflow.db",
    )
    instance = Service(settings)
    panel = pd.DataFrame(
        {"books": np.arange(90) % 7 + 1.0, "toys": np.arange(90) % 3 + 2.0},
        index=pd.date_range("2018-01-01", periods=90, name="date"),
    )
    model = fit_model(training_rows(panel, panel.index[-1]), 2, 5, 42)
    directory = tmp_path / "model"
    save_model(model, directory, list(panel.columns), panel.index[-1].date(), 2, 5, 42)
    instance.initialize(
        panel,
        ActiveModel(
            model_id="initial",
            name="catboost_d2",
            directory=str(directory),
            training_cutoff=panel.index[-1].date(),
        ),
        tmp_path / "initial",
    )
    yield instance
    instance.close()


def new_sales(service: Service, days: int = 30) -> bytes:
    status = service.status()
    dates = pd.date_range(status.last_date + timedelta(days=1), periods=days)
    rows = ["date,category,units"]
    for number, day in enumerate(dates):
        rows += [f"{day.date()},books,{number % 7 + 1}", f"{day.date()},toys,{number % 3 + 2}"]
    return ("\n".join(rows) + "\n").encode()


def test_forecast_api_contract_and_saved_response(service: Service) -> None:
    with TestClient(create_app(service.settings, service)) as client:
        assert client.get("/health").json()["ready"] is True
        for horizon in (7, 14, 30):
            response = client.post(
                "/api/v1/forecasts", json={"category": "books", "horizon": horizon}
            )
            assert response.status_code == 200
            body = response.json()
            assert len(body["forecast"]) == horizon
            assert body["forecast"][0]["date"] == "2018-04-01"
            assert body["total"] == pytest.approx(sum(point["units"] for point in body["forecast"]))
            assert client.get(f"/api/v1/forecasts/{body['forecast_id']}").json() == body
            assert "x-request-id" in response.headers
        assert (
            client.post("/api/v1/forecasts", json={"category": "books", "horizon": 5}).status_code
            == 422
        )
        assert client.post("/api/v1/forecasts", json={"category": "unknown"}).status_code == 422
        assert client.get("/api/v1/forecasts/missing").status_code == 404
        assert client.post("/api/v1/jobs/retrain").status_code == 409


def test_upload_versions_and_failed_import_are_atomic(service: Service) -> None:
    original = service.status()
    content = new_sales(service)
    with TestClient(create_app(service.settings, service)) as client:
        first = client.post("/api/v1/sales", files={"file": ("sales.csv", content)})
        assert first.status_code == 200
        assert first.json()["added_days"] == 30
        repeated = client.post("/api/v1/sales", files={"file": ("sales.csv", content)})
        assert repeated.json()["unchanged"] is True
        assert repeated.json()["dataset_id"] == first.json()["dataset_id"]
        before = service.status()
        failed = client.post(
            "/api/v1/sales",
            files={"file": ("bad.csv", b"date,category,units\n2018-04-01,books,999")},
        )
        assert failed.status_code == 422
        assert service.status() == before
        assert service.panel(original.dataset_id).index[-1].date() == original.last_date
        assert client.get("/api/v1/sales/template").content.startswith(b"date,category,units")


def test_retraining_records_result_without_automatic_activation(service: Service) -> None:
    service.import_sales(new_sales(service))
    original = service.status().model_id
    job = service.start_retraining()
    service.worker.shutdown(wait=True)
    finished = service.job(job.job_id)
    assert finished.status == "completed", finished.message
    assert finished.candidate_wape is not None
    assert finished.current_wape is not None
    assert service.status().model_id == original
    if finished.eligible:
        service.activate(job.job_id)
        assert service.status().model_id != original
        assert (
            service.forecast(ForecastRequest(category="books")).origin == service.status().last_date
        )
    else:
        with pytest.raises(DataError):
            service.activate(job.job_id)


def test_stale_candidate_cannot_replace_current_state(service: Service) -> None:
    snapshot = service.snapshot()
    job = Job(
        job_id="candidate",
        status="completed",
        dataset_id=snapshot.dataset_id,
        parent_model_id=snapshot.model.model_id,
        message="test candidate",
        eligible=True,
        model=snapshot.model.model_copy(update={"model_id": "candidate-model"}),
    )
    service.store.save_job(job.job_id, job.model_dump_json())
    service.import_sales(new_sales(service, 1))
    with pytest.raises(DataError, match="изменились"):
        service.activate(job.job_id)


def test_busy_and_restart_recovery(service: Service) -> None:
    snapshot = service.snapshot()
    job = Job(
        job_id="interrupted",
        status="running",
        dataset_id=snapshot.dataset_id,
        parent_model_id=snapshot.model.model_id,
        message="running",
    )
    service.store.save_job(job.job_id, job.model_dump_json())
    with pytest.raises(DataError, match="уже выполняется"):
        service.start_retraining()
    restarted = Service(service.settings)
    try:
        assert restarted.job(job.job_id).status == "failed"
        assert restarted.status() == service.status()
    finally:
        restarted.close()


def test_not_ready_is_actionable(tmp_path: Path) -> None:
    settings = Settings(
        app_db=tmp_path / "app.db", data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts"
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["ready"] is False
        assert client.get("/api/v1/status").status_code == 503


def test_monitoring_api_cache_and_truth_arrival(service: Service) -> None:
    with TestClient(create_app(service.settings, service)) as client:
        first = client.get("/api/v1/monitoring").json()
        assert first["data_drift"]["status"] == "waiting"
        assert client.get("/api/v1/monitoring/html").status_code == 404
        client.post("/api/v1/forecasts", json={"category": "books", "horizon": 7})
        pending = client.get("/api/v1/monitoring").json()
        assert pending["report_id"] != first["report_id"]
        assert pending["quality"][0]["status"] == "waiting"
        service.import_sales(new_sales(service, 30))
        measured = client.get("/api/v1/monitoring")
        assert measured.status_code == 200
        body = measured.json()
        assert body["quality"][0]["status"] == "reference_only"
        assert body["quality"][0]["reference"]["mae"] >= 0
        assert body["html_available"]
        assert client.get("/api/v1/monitoring").json() == body
        assert client.get("/api/v1/monitoring/html").status_code == 200
        service.settings.quality_absolute_increase = 10
        assert client.get("/api/v1/monitoring").json()["report_id"] != body["report_id"]


def test_ui_forecast_inventory_and_quality(
    service: Service, monkeypatch: pytest.MonkeyPatch
) -> None:
    with TestClient(create_app(service.settings, service)) as client:

        def request(method: str, url: str, **kwargs: Any) -> Any:
            kwargs.pop("timeout", None)
            return client.request(method, url, **kwargs)

        monkeypatch.setattr(httpx, "request", request)
        app = AppTest.from_file(str(Path(ui_module.__file__)), default_timeout=30).run()
        assert not app.exception
        app.button[0].click().run()
        assert not app.exception
        assert len(app.metric) == 3
        app.sidebar.radio[0].set_value("Запасы").run()
        app.button[0].click().run()
        assert not app.exception
        app.sidebar.radio[0].set_value("Данные").run()
        assert not app.exception
        app.sidebar.radio[0].set_value("Качество").run()
        assert not app.exception
        service.import_sales(new_sales(service))
        app.run()
        assert not app.exception
        assert any("Evidently" in button.label for button in app.button)
