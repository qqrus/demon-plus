"""Idempotent preparation for fresh Docker named volumes."""

from demon_plus.config import Settings
from demon_plus.data.download import fetch_olist
from demon_plus.evaluation.train import evaluate_holdout, train_candidates
from demon_plus.logging import configure_logging


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    fetch_olist(settings.data_dir)
    if not (settings.artifacts_dir / "training" / "latest.json").exists():
        train_candidates(settings)
    evaluate_holdout(settings)


if __name__ == "__main__":
    main()
