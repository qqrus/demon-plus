"""Tests never read the developer's local .env or application environment."""

import os
from pathlib import Path

import pytest
import structlog


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # CLI capture streams are closed after each test; do not retain their logger factory.
    structlog.reset_defaults()
    monkeypatch.chdir(tmp_path)
    for name in os.environ:
        if name.upper().startswith("DEMON_PLUS_"):
            monkeypatch.delenv(name)
