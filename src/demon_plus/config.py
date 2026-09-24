"""Validated runtime settings; no filesystem writes happen during loading."""

from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Read .env relative to the working directory, then environment overrides."""

    model_config = SettingsConfigDict(
        env_prefix="DEMON_PLUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        hide_input_in_errors=True,
    )

    log_level: LogLevel = "INFO"
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    analysis_start: date = date(2017, 1, 5)
    validation_cutoffs: tuple[date, ...] = (date(2018, 3, 31), date(2018, 4, 30), date(2018, 5, 31))
    holdout_cutoff: date = date(2018, 6, 30)
    top_categories: int = Field(default=10, ge=1)
    min_active_days: int = Field(default=60, ge=1)

    @model_validator(mode="after")
    def validate_evaluation_dates(self) -> "Settings":
        cutoffs = self.validation_cutoffs
        if not cutoffs or list(cutoffs) != sorted(set(cutoffs)):
            raise ValueError("Validation cutoffs must be non-empty, unique and increasing")
        if cutoffs[0] < self.analysis_start + timedelta(days=27):
            raise ValueError("At least 28 observed days are required before validation")
        if any(right < left + timedelta(days=30) for left, right in pairwise(cutoffs)):
            raise ValueError("Validation windows must not overlap")
        if self.holdout_cutoff < cutoffs[-1] + timedelta(days=30):
            raise ValueError("Holdout must follow all validation targets")
        return self

    @field_validator("data_dir", "artifacts_dir", mode="before")
    @classmethod
    def reject_blank_paths(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Directory path must not be blank")
        return value
