"""Small SQLAlchemy repository; each write uses a transaction."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Column, MetaData, String, Table, Text, create_engine, insert, select, update


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{path.resolve().as_posix()}")
        metadata = MetaData()
        self.state = Table(
            "state",
            metadata,
            Column("key", String, primary_key=True),
            Column("value", Text, nullable=False),
        )
        self.datasets = Table(
            "datasets",
            metadata,
            Column("id", String, primary_key=True),
            Column("path", Text, nullable=False),
            Column("created_at", String, nullable=False),
        )
        self.forecasts = Table(
            "forecasts",
            metadata,
            Column("id", String, primary_key=True),
            Column("dataset_id", String, nullable=False),
            Column("model_id", String, nullable=False),
            Column("payload", Text, nullable=False),
            Column("created_at", String, nullable=False),
        )
        self.jobs = Table(
            "jobs",
            metadata,
            Column("id", String, primary_key=True),
            Column("payload", Text, nullable=False),
        )
        metadata.create_all(self.engine)

    def get_state(self, key: str) -> str | None:
        with self.engine.connect() as connection:
            value = connection.execute(
                select(self.state.c.value).where(self.state.c.key == key)
            ).scalar_one_or_none()
        return str(value) if value is not None else None

    def set_state(self, key: str, value: str) -> None:
        with self.engine.begin() as connection:
            exists = connection.execute(
                select(self.state.c.key).where(self.state.c.key == key)
            ).first()
            if exists:
                connection.execute(
                    update(self.state).where(self.state.c.key == key).values(value=value)
                )
            else:
                connection.execute(insert(self.state).values(key=key, value=value))

    def add_dataset(self, dataset_id: str, path: Path) -> None:
        with self.engine.begin() as connection:
            exists = connection.execute(
                select(self.datasets.c.id).where(self.datasets.c.id == dataset_id)
            ).first()
            if not exists:
                connection.execute(
                    insert(self.datasets).values(
                        id=dataset_id, path=str(path), created_at=datetime.now(UTC).isoformat()
                    )
                )

    def dataset_path(self, dataset_id: str) -> Path:
        with self.engine.connect() as connection:
            path = connection.execute(
                select(self.datasets.c.path).where(self.datasets.c.id == dataset_id)
            ).scalar_one()
        return Path(str(path))

    def save_forecast(self, forecast_id: str, dataset_id: str, model_id: str, payload: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                insert(self.forecasts).values(
                    id=forecast_id,
                    dataset_id=dataset_id,
                    model_id=model_id,
                    payload=payload,
                    created_at=datetime.now(UTC).isoformat(),
                )
            )

    def save_job(self, job_id: str, payload: str) -> None:
        with self.engine.begin() as connection:
            exists = connection.execute(
                select(self.jobs.c.id).where(self.jobs.c.id == job_id)
            ).first()
            if exists:
                connection.execute(
                    update(self.jobs).where(self.jobs.c.id == job_id).values(payload=payload)
                )
            else:
                connection.execute(insert(self.jobs).values(id=job_id, payload=payload))

    def job_payloads(self) -> list[str]:
        with self.engine.connect() as connection:
            return [
                str(value) for value in connection.execute(select(self.jobs.c.payload)).scalars()
            ]

    def activate(self, snapshot: str, job_id: str, payload: str) -> None:
        """Publish the model and its job status in one transaction."""
        with self.engine.begin() as connection:
            connection.execute(
                update(self.state).where(self.state.c.key == "snapshot").values(value=snapshot)
            )
            connection.execute(
                update(self.jobs).where(self.jobs.c.id == job_id).values(payload=payload)
            )

    def forecast_payload(self, forecast_id: str) -> str | None:
        with self.engine.connect() as connection:
            value: Any = connection.execute(
                select(self.forecasts.c.payload).where(self.forecasts.c.id == forecast_id)
            ).scalar_one_or_none()
        return str(value) if value is not None else None
