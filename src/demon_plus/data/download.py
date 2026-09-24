"""Download the public, version-pinned Olist archive without credentials."""

import hashlib
import json
import shutil
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

from demon_plus.errors import DataError

SOURCE_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
    "?datasetVersionNumber=2"
)
ARCHIVE_SHA256 = "967e41e04fc306fe604e2a693f488995a8b41e5047418f8a5c8e4abd6deca784"
FILES = (
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "product_category_name_translation.csv",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_selected(archive: Path, destination: Path) -> dict[str, str]:
    """Read only allowlisted flat members; never extract arbitrary zip paths."""
    with ZipFile(archive) as bundle:
        if any(bundle.namelist().count(name) != 1 for name in FILES):
            raise DataError("Olist archive must contain each required CSV exactly once.")
        if any(bundle.getinfo(name).file_size > 100_000_000 for name in FILES):
            raise DataError("A CSV exceeds the supported Olist file size.")
        contents = {name: bundle.read(name) for name in FILES}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}
    # Check all conflicts before writing any files. Existing custom data is never overwritten.
    for name, digest in hashes.items():
        target = destination / name
        if target.exists() and sha256(target) != digest:
            raise DataError("Existing raw CSV differs from Olist v2; use a new data directory.")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        target = destination / name
        if not target.exists():
            target.write_bytes(content)
    return hashes


def fetch_olist(data_dir: Path) -> Path:
    """Reuse a verified archive or download it; record source and file hashes."""
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "olist-v2.zip"
    partial = raw / "olist-v2.zip.part"
    try:
        if not archive.exists():
            with urlopen(DOWNLOAD_URL, timeout=120) as response, partial.open("wb") as target:
                shutil.copyfileobj(response, target)
            if sha256(partial) != ARCHIVE_SHA256:
                raise DataError(
                    "Downloaded archive differs from pinned Olist v2; check the source."
                )
            partial.replace(archive)
        if sha256(archive) != ARCHIVE_SHA256:
            raise DataError("Archive checksum mismatch. Obtain the official Olist version 2 ZIP.")
        hashes = extract_selected(archive, raw)
    except (URLError, TimeoutError, BadZipFile) as exc:
        raise DataError(
            "Cannot obtain Olist. Download version 2 manually to data/raw/olist-v2.zip "
            "(or the configured data directory), then retry fetch-data."
        ) from exc
    finally:
        partial.unlink(missing_ok=True)
    manifest = {
        "source": SOURCE_URL,
        "dataset_version": 2,
        "license": "CC-BY-NC-SA-4.0",
        "archive_sha256": ARCHIVE_SHA256,
        "files": hashes,
    }
    path = raw / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
