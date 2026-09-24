from pathlib import Path

import pytest
from pydantic import ValidationError

from demon_plus.config import Settings


def test_dotenv_loading_and_environment_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    Path(".env").write_text(
        "DEMON_PLUS_LOG_LEVEL=DEBUG\nDEMON_PLUS_DATA_DIR=данные\n", encoding="utf-8"
    )
    monkeypatch.setenv("DEMON_PLUS_LOG_LEVEL", "WARNING")

    settings = Settings()

    assert settings.log_level == "WARNING"
    assert settings.data_dir == Path("данные")
    assert not settings.data_dir.exists()


@pytest.mark.parametrize(
    "contents",
    [
        "DEMON_PLUS_LOG_LEVEL=VERBOSE\n",
        "DEMON_PLUS_DATA_DIR=\n",
        "DEMON_PLUS_ARTIFACTS_DIR=\n",
        "DEMON_PLUS_LOG_LEVL=INFO\n",
        "DEMON_PLUS_VALIDATION_CUTOFFS=[]\n",
        'DEMON_PLUS_VALIDATION_CUTOFFS=["2018-03-31","2018-04-01"]\n',
        'DEMON_PLUS_VALIDATION_CUTOFFS=["2018-04-30","2018-03-31"]\n',
        "DEMON_PLUS_HOLDOUT_CUTOFF=2018-06-01\n",
        "DEMON_PLUS_TOP_CATEGORIES=0\n",
    ],
)
def test_invalid_dotenv_is_rejected(contents: str) -> None:
    Path(".env").write_text(contents, encoding="utf-8")

    with pytest.raises(ValidationError):
        Settings()
