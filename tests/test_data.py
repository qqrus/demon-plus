from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from demon_plus.data.download import FILES, extract_selected, fetch_olist
from demon_plus.data.olist import UNKNOWN_CATEGORY, daily_panel, load_olist
from demon_plus.errors import DataError


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    pd.DataFrame(
        {
            "order_id": ["a", "b", "c"],
            "order_purchase_timestamp": [
                "2018-01-01 12:00:00",
                "2018-01-02 01:00:00",
                "2018-01-03 23:00:00",
            ],
            "order_status": ["delivered", "canceled", "unavailable"],
        }
    ).to_csv(tmp_path / FILES[0], index=False)
    pd.DataFrame(
        {
            "order_id": ["a", "a", "b"],
            "order_item_id": [1, 2, 1],
            "product_id": ["p", "p", "q"],
        }
    ).to_csv(tmp_path / FILES[1], index=False)
    pd.DataFrame({"product_id": ["p", "q"], "product_category_name": ["books", None]}).to_csv(
        tmp_path / FILES[2], index=False
    )
    return tmp_path


def test_item_units_and_final_status_are_not_confused(raw: Path) -> None:
    result = load_olist(raw)
    assert result.sales["units"].sum() == 3
    assert result.audit["orders_without_items"] == 1
    assert result.audit["unknown_category_units"] == 1
    panel = daily_panel(result.sales, pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-02"))
    assert panel.loc["2018-01-01", "books"] == 2
    assert panel.loc["2018-01-02", "books"] == 0
    assert panel.loc["2018-01-02", UNKNOWN_CATEGORY] == 1


def test_exact_duplicate_item_does_not_double_sales(raw: Path) -> None:
    path = raw / FILES[1]
    items = pd.read_csv(path)
    pd.concat([items, items.iloc[[0]]]).to_csv(path, index=False)
    result = load_olist(raw)
    assert len(result.sales) == 3
    assert result.audit["exact_duplicates_removed"] == {"orders": 0, "items": 1, "products": 0}


@pytest.mark.parametrize("corruption", ["duplicate", "missing_product", "missing_order", "bad_id"])
def test_invalid_relations_and_keys_fail(raw: Path, corruption: str) -> None:
    path = raw / FILES[1]
    items = pd.read_csv(path, dtype="string")
    if corruption == "duplicate":
        row = items.iloc[[0]].copy()
        row["product_id"] = "q"
        items = pd.concat([items, row])
    elif corruption == "missing_product":
        items.loc[0, "product_id"] = "missing"
    elif corruption == "missing_order":
        items.loc[0, "order_id"] = "missing"
    else:
        items.loc[0, "order_item_id"] = "0"
    items.to_csv(path, index=False)
    with pytest.raises(DataError):
        load_olist(raw)


@pytest.mark.parametrize("date", ["bad-date", ""])
def test_invalid_purchase_date_fails(raw: Path, date: str) -> None:
    path = raw / FILES[0]
    orders = pd.read_csv(path)
    orders.loc[0, "order_purchase_timestamp"] = date
    orders.to_csv(path, index=False)
    with pytest.raises(DataError, match="timestamp"):
        load_olist(raw)


def test_missing_store_day_is_not_silently_zero_filled(raw: Path) -> None:
    sales = load_olist(raw).sales
    sales.loc[sales["order_id"] == "b", "date"] = pd.Timestamp("2018-01-03")
    with pytest.raises(DataError, match="Entire store days"):
        daily_panel(sales, pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-03"))


def test_download_rejects_unverified_existing_archive(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "olist-v2.zip").write_bytes(b"not official data")
    with pytest.raises(DataError, match="checksum"):
        fetch_olist(tmp_path)


def test_extraction_allowlist_and_conflict_protection(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    destination = tmp_path / "raw"
    with ZipFile(archive, "w") as bundle:
        for name in FILES:
            bundle.writestr(name, "column\nvalue\n")
        bundle.writestr("../escaped.txt", "do not extract")
    hashes = extract_selected(archive, destination)
    assert set(hashes) == set(FILES)
    assert not (tmp_path / "escaped.txt").exists()
    assert extract_selected(archive, destination) == hashes
    (destination / FILES[0]).write_text("custom", encoding="utf-8")
    with pytest.raises(DataError, match="differs"):
        extract_selected(archive, destination)
    assert (destination / FILES[0]).read_text(encoding="utf-8") == "custom"
