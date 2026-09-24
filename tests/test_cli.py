import json
from pathlib import Path

import pytest

from demon_plus.cli import main


def test_check_separates_result_from_json_logs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check"]) == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    event = json.loads(captured.err)
    assert result["status"] == "ok"
    assert result["scope"] == "configuration"
    assert result["package_version"]
    assert event["event"] == "configuration_checked"
    assert event["level"] == "info"
    assert event["timestamp"]


def test_invalid_config_does_not_leak_values(capsys: pytest.CaptureFixture[str]) -> None:
    secret_marker = "private-value-must-not-be-logged"
    Path(".env").write_text(f"DEMON_PLUS_LOG_LEVEL={secret_marker}\n", encoding="utf-8")

    assert main(["check"]) == 2

    captured = capsys.readouterr()
    assert not captured.out
    assert secret_marker not in captured.err
    event = json.loads(captured.err)
    assert event["event"] == "configuration_invalid"
    assert ".env.example" in event["message"]


def test_warning_level_suppresses_info(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DEMON_PLUS_LOG_LEVEL", "WARNING")

    assert main(["check"]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "ok"
    assert captured.err == ""


def test_malformed_cutoff_configuration_is_handled(capsys: pytest.CaptureFixture[str]) -> None:
    Path(".env").write_text("DEMON_PLUS_VALIDATION_CUTOFFS=not-json\n", encoding="utf-8")
    assert main(["check"]) == 2
    captured = capsys.readouterr()
    assert "not-json" not in captured.err
    assert json.loads(captured.err)["event"] == "configuration_invalid"


def test_missing_dataset_is_an_actionable_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["baseline"]) == 2
    captured = capsys.readouterr()
    event = json.loads(captured.err)
    assert event["event"] == "data_invalid"
    assert "fetch-data" in event["message"]
