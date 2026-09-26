from datetime import date
from typing import Literal

from demon_plus.api.contracts import Point, StockItem, StockRisk
from demon_plus.errors import DataError


def calculate_risk(item: StockItem, forecast: list[Point], origin: date) -> StockRisk:
    if any(
        delivery.date <= origin or delivery.date > forecast[-1].date for delivery in item.deliveries
    ):
        raise DataError("Поставка должна попадать в выбранный период прогноза после даты среза.")
    balance = item.on_hand
    shortage = None
    balances: list[Point] = []
    for point in forecast:
        # Receipts are available at the beginning of the day; unmet demand is carried forward.
        balance += sum(
            delivery.units for delivery in item.deliveries if delivery.date == point.date
        )
        balance -= point.units
        if balance < 0 and shortage is None:
            shortage = point.date
        balances.append(Point(date=point.date, units=balance))
    excess = max(0, balance - item.target_remaining)
    status: Literal["shortage", "excess", "balanced"] = (
        "shortage" if shortage else "excess" if excess > 0 else "balanced"
    )
    return StockRisk(
        category=item.category,
        forecast_units=sum(point.units for point in forecast),
        end_balance=balance,
        shortage_date=shortage,
        excess_units=excess,
        status=status,
        balances=balances,
    )
