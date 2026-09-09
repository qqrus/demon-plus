"""Validated runtime settings; no filesystem writes happen during loading."""

from pathlib import Path
from typing import Literal

from pydantic import field_validator
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

    @field_validator("data_dir", "artifacts_dir", mode="before")
    @classmethod
    def reject_blank_paths(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Directory path must not be blank")
        return value
