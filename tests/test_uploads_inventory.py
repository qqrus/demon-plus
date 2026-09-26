from datetime import date

import pandas as pd
import pytest

from demon_plus.api.contracts import Delivery, Point, StockItem
from demon_plus.data.uploads import merge_daily_csv
from demon_plus.errors import DataError
from demon_plus.inventory.risk import calculate_risk


def test_append_is_idempotent_and_preserves_history() -> None:
    history = pd.DataFrame(
        {"a": [1.0], "b": [2.0]}, index=pd.date_range("2018-01-01", periods=1, name="date")
    )
    content = b"date,category,units\n2018-01-02,a,0\n2018-01-02,b,3\n2018-01-02,b,3\n"
    merged = merge_daily_csv(content, history, 1000)
    assert len(merged) == 2
    pd.testing.assert_frame_equal(merged, merge_daily_csv(content, merged, 1000))
    assert history.iloc[0].tolist() == [1, 2]


@pytest.mark.parametrize(
    "rows",
    [
        "2018-01-02,a,1",  # missing category
        "2018-01-03,a,1\n2018-01-03,b,1",  # missing day
        "2018-01-02,a,-1\n2018-01-02,b,1",
        "2018-01-02,a,inf\n2018-01-02,b,1",
        "2018-01-02,a,0.5\n2018-01-02,b,1",
        "2018-01-02,a,1\n2018-01-02,a,2\n2018-01-02,b,1",
        "2018-01-01,a,8\n2018-01-01,b,2",  # past correction
        "2018-01-02,unknown,1\n2018-01-02,b,1",
        "2018-02-30,a,1\n2018-02-30,b,1",
    ],
)
def test_invalid_upload_is_rejected(rows: str) -> None:
    history = pd.DataFrame({"a": [1.0], "b": [2.0]}, index=pd.date_range("2018-01-01", periods=1))
    with pytest.raises(DataError):
        merge_daily_csv(("date,category,units\n" + rows).encode(), history, 1000)


def test_stock_flow_receipts_and_shortage_date() -> None:
    points = [Point(date=date(2018, 1, 2), units=6), Point(date=date(2018, 1, 3), units=6)]
    item = StockItem(
        category="a",
        on_hand=5,
        target_remaining=1,
        deliveries=[Delivery(date=date(2018, 1, 3), units=20)],
    )
    risk = calculate_risk(item, points, date(2018, 1, 1))
    assert risk.shortage_date == date(2018, 1, 2)
    assert risk.end_balance == 13
    assert risk.excess_units == 12
    assert risk.status == "shortage"  # Late delivery does not erase the earlier stockout.
    item.deliveries = [Delivery(date=date(2018, 1, 2), units=20)]
    assert calculate_risk(item, points, date(2018, 1, 1)).status == "excess"
    item.deliveries = [Delivery(date=date(2018, 1, 1), units=20)]
    with pytest.raises(DataError):
        calculate_risk(item, points, date(2018, 1, 1))
