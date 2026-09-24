"""Olist item rows become gross ordered units, without future status filtering."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from demon_plus.data.download import FILES, sha256
from demon_plus.errors import DataError

UNKNOWN_CATEGORY = "__unknown__"


@dataclass(frozen=True)
class SalesDataset:
    sales: pd.DataFrame
    audit: dict[str, object]


def _read_table(path: Path, columns: list[str], key: list[str]) -> tuple[pd.DataFrame, int]:
    try:
        table = pd.read_csv(path, dtype="string")
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataError(
            f"Cannot read {path.name}; run fetch-data or check the CSV format."
        ) from exc
    if not set(columns).issubset(table.columns):
        raise DataError(f"Missing required columns in {path.name}.")
    duplicates = int(table.duplicated().sum())
    table = table.drop_duplicates().copy()
    if table.empty or table[key].isna().any().any():
        raise DataError(f"Empty table or missing primary key in {path.name}.")
    if table.duplicated(key).any():
        raise DataError(f"Conflicting duplicate primary keys in {path.name}.")
    return table[columns], duplicates


def load_olist(raw: Path) -> SalesDataset:
    orders, order_duplicates = _read_table(
        raw / FILES[0],
        ["order_id", "order_purchase_timestamp", "order_status"],
        ["order_id"],
    )
    items, item_duplicates = _read_table(
        raw / FILES[1], ["order_id", "order_item_id", "product_id"], ["order_id", "order_item_id"]
    )
    products, product_duplicates = _read_table(
        raw / FILES[2], ["product_id", "product_category_name"], ["product_id"]
    )
    if not items["order_item_id"].str.fullmatch(r"[1-9][0-9]*").all():
        raise DataError("Order item IDs must be positive integers.")
    if items["product_id"].isna().any():
        raise DataError("Item product IDs must not be missing.")
    try:
        orders["date"] = pd.to_datetime(
            orders["order_purchase_timestamp"], format="%Y-%m-%d %H:%M:%S", errors="raise"
        ).dt.normalize()
    except (ValueError, TypeError) as exc:
        raise DataError("Invalid order purchase timestamp; expected YYYY-MM-DD HH:MM:SS.") from exc
    if orders["date"].isna().any():
        raise DataError("Order purchase timestamps must not be missing.")
    if not items["order_id"].isin(orders["order_id"]).all():
        raise DataError("Some items refer to missing orders.")
    if not items["product_id"].isin(products["product_id"]).all():
        raise DataError("Some items refer to missing products.")
    joined = items.merge(orders, on="order_id", validate="many_to_one").merge(
        products, on="product_id", validate="many_to_one"
    )
    joined["category"] = joined["product_category_name"].fillna(UNKNOWN_CATEGORY)
    sales = joined[["order_id", "order_item_id", "product_id", "category", "date"]].copy()
    sales["units"] = 1
    status_counts = {
        str(key): int(value) for key, value in orders["order_status"].value_counts().items()
    }
    monthly = sales.groupby(sales["date"].dt.strftime("%Y-%m")).size()
    product_counts = sales.groupby("product_id").size()
    audit: dict[str, object] = {
        "target": "gross_ordered_units_by_purchase_date",
        "orders": len(orders),
        "item_rows": len(items),
        "products": len(products),
        "categories": int(sales["category"].nunique()),
        "orders_without_items": int((~orders["order_id"].isin(items["order_id"])).sum()),
        "unknown_category_units": int((sales["category"] == UNKNOWN_CATEGORY).sum()),
        "exact_duplicates_removed": {
            "orders": order_duplicates,
            "items": item_duplicates,
            "products": product_duplicates,
        },
        "first_item_date": str(sales["date"].min().date()),
        "last_item_date": str(sales["date"].max().date()),
        "order_status_counts": status_counts,
        "monthly_units": {str(key): int(value) for key, value in monthly.items()},
        "products_with_one_unit": int((product_counts == 1).sum()),
        "products_with_at_most_10_units": int((product_counts <= 10).sum()),
        "input_sha256": {name: sha256(raw / name) for name in FILES[:3]},
    }
    return SalesDataset(sales=sales.sort_values("date").reset_index(drop=True), audit=audit)


def daily_panel(sales: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fill category-day zeros only inside an explicitly accepted observation window."""
    if start > end or sales["date"].min() > start or sales["date"].max() < end:
        raise DataError("Sales do not cover the requested observation window.")
    window = sales.loc[sales["date"].between(start, end)]
    dates = pd.date_range(start, end, freq="D")
    observed_dates = pd.DatetimeIndex(window["date"].unique())
    if not dates.isin(observed_dates).all():
        raise DataError("Entire store days are absent; investigate gaps before zero-filling.")
    panel = window.groupby(["date", "category"])["units"].sum().unstack("category")
    return panel.reindex(dates).fillna(0).astype(float).rename_axis("date")
