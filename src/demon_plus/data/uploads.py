"""Append-only daily category CSV: explicit zeros, complete dates, no double counting."""

import io

import numpy as np
import pandas as pd

from demon_plus.errors import DataError


def merge_daily_csv(content: bytes, history: pd.DataFrame, max_bytes: int) -> pd.DataFrame:
    if not content or len(content) > max_bytes:
        raise DataError("Файл пуст или превышает допустимый размер.")
    try:
        table = pd.read_csv(io.BytesIO(content), dtype="string", encoding="utf-8-sig")
    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise DataError("Не удалось прочитать CSV: нужен UTF-8 и разделитель запятая.") from exc
    if list(table.columns) != ["date", "category", "units"] or table.empty or len(table) > 100_000:
        raise DataError("Ожидаются колонки date,category,units и от 1 до 100000 строк.")
    if table.isna().any().any() or not table["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise DataError("Пропуски запрещены; дата должна иметь формат YYYY-MM-DD.")
    try:
        table["date"] = pd.to_datetime(table["date"], format="%Y-%m-%d", errors="raise")
        table["units"] = pd.to_numeric(table["units"], errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise DataError("Проверьте даты и числовые значения units.") from exc
    values = table["units"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise DataError("units — конечное неотрицательное целое количество единиц.")
    if not table["category"].isin(history.columns).all():
        raise DataError("CSV содержит неизвестную категорию; используйте категории из приложения.")
    table = table.drop_duplicates()
    if table.duplicated(["date", "category"]).any():
        raise DataError("Есть разные количества для одной даты и категории.")
    incoming = table.pivot(index="date", columns="category", values="units").reindex(
        columns=history.columns
    )
    if incoming.isna().any().any():
        raise DataError("Каждый загружаемый день должен содержать все категории, включая нули.")
    overlap = incoming.index.intersection(history.index)
    if not np.array_equal(incoming.loc[overlap].to_numpy(), history.loc[overlap].to_numpy()):
        raise DataError(
            "Файл изменяет уже известные продажи. В MVP разрешено только добавление дат."
        )
    new = incoming.loc[~incoming.index.isin(history.index)]
    if not new.empty and new.index.min() <= history.index.max():
        raise DataError("Новые даты должны идти после текущей истории.")
    result = pd.concat([history, new]).sort_index().astype(float).rename_axis("date")
    expected = pd.date_range(result.index[0], periods=len(result), freq="D")
    if not result.index.equals(expected):
        raise DataError(
            "В датах есть разрыв. Загрузите все дни подряд, явно указав нулевые продажи."
        )
    return result
