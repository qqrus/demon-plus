"""Typed HTTP contracts shared with the local application."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Horizon = Literal[7, 14, 30]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Point(Contract):
    date: date
    units: float


class ForecastRequest(Contract):
    category: str
    horizon: Horizon = 14


class ForecastResponse(Contract):
    forecast_id: str
    category: str
    horizon: Horizon
    origin: date
    dataset_id: str
    model_id: str
    model_name: str
    history: list[Point]
    forecast: list[Point]
    total: float


class ActiveModel(Contract):
    model_id: str
    name: str
    directory: str
    training_cutoff: date


class StatusResponse(Contract):
    dataset_id: str
    categories: list[str]
    first_date: date
    last_date: date
    model_id: str
    model_name: str
    training_cutoff: date
    historical_demo: bool = True


class ImportResponse(Contract):
    dataset_id: str
    added_days: int
    unchanged: bool
    last_date: date


class Delivery(Contract):
    date: date
    units: float = Field(ge=0)


class StockItem(Contract):
    category: str
    on_hand: float = Field(ge=0)
    target_remaining: float = Field(default=0, ge=0)
    deliveries: list[Delivery] = Field(default_factory=list)


class InventoryRequest(Contract):
    horizon: Horizon = 14
    items: list[StockItem] = Field(min_length=1, max_length=100)
    synthetic: bool = False


class StockRisk(Contract):
    category: str
    forecast_units: float
    end_balance: float
    shortage_date: date | None
    excess_units: float
    status: Literal["shortage", "excess", "balanced"]
    balances: list[Point]


class InventoryResponse(Contract):
    origin: date
    model_id: str
    dataset_id: str
    synthetic: bool
    risks: list[StockRisk]


class Job(Contract):
    job_id: str
    status: Literal["queued", "running", "completed", "failed", "activated"]
    dataset_id: str
    parent_model_id: str
    message: str
    eligible: bool = False
    candidate_wape: float | None = None
    current_wape: float | None = None
    baseline_wape: float | None = None
    model: ActiveModel | None = None
