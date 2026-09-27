"""Run: uv run streamlit run src/demon_plus/ui/app.py"""

from typing import Any, cast

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demon_plus.config import Settings

LABELS = {
    "cama_mesa_banho": "Текстиль для дома",
    "esporte_lazer": "Спорт и отдых",
    "moveis_decoracao": "Мебель и декор",
    "beleza_saude": "Красота и здоровье",
    "informatica_acessorios": "Компьютерные аксессуары",
    "utilidades_domesticas": "Товары для дома",
    "relogios_presentes": "Часы и подарки",
    "telefonia": "Телефония",
    "ferramentas_jardim": "Садовые инструменты",
    "brinquedos": "Игрушки",
}


def call_api(method: str, path: str, **kwargs: Any) -> Any:
    settings = Settings()
    try:
        response = httpx.request(method, settings.api_url.rstrip("/") + path, timeout=60, **kwargs)
        if response.is_error:
            try:
                message = response.json().get("detail", "Не удалось выполнить запрос.")
            except ValueError:
                message = "API вернул неожиданный ответ. Проверьте журнал сервиса."
            st.error(str(message))
            st.stop()
        return (
            response.content
            if any(
                kind in response.headers.get("content-type", "")
                for kind in ("text/csv", "text/html")
            )
            else response.json()
        )
    except httpx.HTTPError:
        st.error("API недоступен. Запустите сервер FastAPI и проверьте DEMON_PLUS_API_URL.")
        st.stop()


def show_forecast(status: dict[str, Any]) -> None:
    left, right = st.columns([3, 1])
    category = left.selectbox(
        "Категория", status["categories"], format_func=lambda value: LABELS.get(value, value)
    )
    horizon = right.selectbox("Горизонт, дней", [7, 14, 30], index=1)
    if st.button("Построить прогноз", type="primary"):
        with st.spinner("Рассчитываем прогноз…"):
            st.session_state["forecast"] = call_api(
                "POST", "/api/v1/forecasts", json={"category": category, "horizon": horizon}
            )
    result = st.session_state.get("forecast")
    if not result:
        st.info("Выберите категорию и горизонт, затем постройте прогноз.")
        return
    if result["dataset_id"] != status["dataset_id"] or result["model_id"] != status["model_id"]:
        st.warning("Данные или модель обновились. Постройте новый прогноз.")
    st.subheader(f"{LABELS.get(result['category'], result['category'])} · {result['horizon']} дней")
    columns = st.columns(3)
    columns[0].metric("Ожидаемые продажи", f"{result['total']:.1f} ед.")
    columns[1].metric("Дата среза", result["origin"])
    columns[2].metric("Модель", result["model_name"])
    figure = go.Figure()
    for key, label, color in [
        ("history", "История", "#64748b"),
        ("forecast", "Прогноз", "#0f9d92"),
    ]:
        points = result[key]
        figure.add_trace(
            go.Scatter(
                x=[point["date"] for point in points],
                y=[point["units"] for point in points],
                name=label,
                mode="lines",
                line={"color": color, "width": 3},
            )
        )
    figure.update_layout(
        height=420,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis_title="Заказанные единицы",
        legend={"orientation": "h"},
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Точечный прогноз без интервала неопределённости. "
        "Продажи отражают заказанные единицы, а не весь потенциальный спрос."
    )
    st.download_button(
        "Скачать прогноз CSV",
        pd.DataFrame(result["forecast"]).to_csv(index=False).encode("utf-8"),
        file_name="forecast.csv",
        mime="text/csv",
    )


def show_inventory(status: dict[str, Any]) -> None:
    st.subheader("Риски запасов")
    st.write(
        "Ожидаемый остаток = начальный запас + поставки − прогноз продаж. "
        "Расчёт выполняется по категориям."
    )
    synthetic = st.checkbox("Использую синтетические остатки для демо", value=True)
    if synthetic:
        st.warning("Показанные начальные остатки — синтетический пример, не данные Olist.")
    horizon = st.selectbox("Период оценки, дней", [7, 14, 30], index=1)
    initial = pd.DataFrame(
        {
            "category": status["categories"],
            "on_hand": [100.0] * len(status["categories"]),
            "target_remaining": [20.0] * len(status["categories"]),
        }
    )
    stocks = st.data_editor(
        initial,
        disabled=["category"],
        hide_index=True,
        column_config={
            "category": "Категория",
            "on_hand": st.column_config.NumberColumn("Текущий запас", min_value=0),
            "target_remaining": st.column_config.NumberColumn("Целевой остаток", min_value=0),
        },
        key="stocks",
    )
    st.caption(
        "Поставки необязательны. Дата должна попадать в выбранный горизонт; "
        "поступление считается в начале дня."
    )
    deliveries = st.data_editor(
        pd.DataFrame(columns=["category", "date", "units"]),
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "category": st.column_config.SelectboxColumn("Категория", options=status["categories"]),
            "date": st.column_config.TextColumn("Дата YYYY-MM-DD"),
            "units": st.column_config.NumberColumn("Количество", min_value=0),
        },
        key="deliveries",
    )
    if st.button("Рассчитать риски", type="primary"):
        if (
            deliveries.isna().any().any()
            or not deliveries["category"].isin(status["categories"]).all()
        ):
            st.error("Заполните категорию, дату и количество для каждой поставки.")
            return
        items = stocks.to_dict("records")
        for item in items:
            item["deliveries"] = deliveries.loc[
                deliveries["category"] == item["category"], ["date", "units"]
            ].to_dict("records")
        try:
            result = call_api(
                "POST",
                "/api/v1/inventory",
                json={"horizon": horizon, "items": items, "synthetic": synthetic},
            )
        except (ValueError, TypeError):
            st.error("Заполните все поля остатков и поставок корректными значениями.")
            return
        risks = pd.DataFrame(result["risks"]).drop(columns=["balances"])
        risks["status"] = risks["status"].map(
            {"shortage": "Дефицит", "excess": "Излишек", "balanced": "В пределах цели"}
        )
        st.dataframe(
            risks.rename(
                columns={
                    "category": "Категория",
                    "forecast_units": "Прогноз, ед.",
                    "end_balance": "Остаток в конце",
                    "shortage_date": "Первый дефицит",
                    "excess_units": "Излишек, ед.",
                    "status": "Статус",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Отрицательный баланс — накопленный неудовлетворённый спрос. "
            "Это сценарий, а не вероятность дефицита."
        )


def show_data(status: dict[str, Any]) -> None:
    st.subheader("Обновление продаж")
    st.write(
        "CSV: date,category,units. Все категории на каждый день; отсутствие продаж укажите нулём. "
        "Новые дни идут подряд после текущей истории."
    )
    st.caption(
        f"История сейчас заканчивается {status['last_date']}. "
        "Совпадающие строки можно загрузить повторно; изменять прошлые продажи нельзя."
    )
    if st.button("Подготовить шаблон CSV"):
        st.session_state["template"] = call_api("GET", "/api/v1/sales/template")
    if "template" in st.session_state:
        st.download_button(
            "Скачать шаблон с нулями — замените их своими продажами",
            st.session_state["template"],
            file_name="sales-template.csv",
            mime="text/csv",
        )
    uploaded = st.file_uploader("Файл продаж", type="csv", max_upload_size=5)
    if uploaded is not None and st.button("Загрузить продажи", type="primary"):
        result = call_api(
            "POST", "/api/v1/sales", files={"file": ("sales.csv", uploaded.getvalue(), "text/csv")}
        )
        st.success(
            f"Добавлено дней: {result['added_days']}. Последняя дата: {result['last_date']}."
        )
    st.divider()
    st.subheader("Переобучение")
    st.write(
        "Нужно не менее 30 новых дней после обучения активной модели. "
        "Кандидат сравнивается с текущей моделью и baseline. Применение выполняется отдельно."
    )
    if st.button("Запустить переобучение"):
        result = call_api("POST", "/api/v1/jobs/retrain")
        st.info(result["message"])
    st.button("Обновить статусы задач")
    jobs = call_api("GET", "/api/v1/jobs")
    for job in jobs:
        with st.expander(f"{job['status']} · {job['job_id'][:8]}", expanded=True):
            st.write(job["message"])
            if job["candidate_wape"] is not None:
                st.write(
                    {
                        "WAPE кандидата, %": job["candidate_wape"],
                        "WAPE текущей модели, %": job["current_wape"],
                        "WAPE baseline, %": job["baseline_wape"],
                    }
                )
                st.caption("Скользящее окно обновления; это не новый независимый итоговый тест.")
            if (
                job["status"] == "completed"
                and job["eligible"]
                and st.button("Применить модель", key=job["job_id"])
            ):
                call_api("POST", f"/api/v1/jobs/{job['job_id']}/activate")
                st.rerun()


def show_quality() -> None:
    st.subheader("Качество модели")
    result = call_api("GET", "/api/v1/metrics")
    st.caption(result["note"])
    if result["active_model_id"] != result["initial_model_id"]:
        st.warning(
            "Активная модель обновлена. Ниже — метрики исходной модели; "
            "результаты кандидата доступны в разделе «Данные»."
        )
    for split, title in [("validation", "Валидация"), ("holdout", "Итоговый июльский тест")]:
        st.markdown(f"**{title}**")
        if not result[split]:
            st.info("Отчёт ещё не создан.")
            continue
        rows = [
            {
                "Модель": row["model"],
                "Горизонт": row["horizon"],
                "MAE": row["daily"]["mae"],
                "RMSE": row["daily"]["rmse"],
                "WAPE, %": row["daily"]["wape_pct"],
            }
            for row in result[split]["aggregate"]
        ]
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    st.divider()
    st.subheader("Мониторинг новых данных и прогнозов")
    monitoring = call_api("GET", "/api/v1/monitoring")
    drift = monitoring["data_drift"]
    st.caption(f"Факты по {monitoring['as_of']}. Отчёт обновляется при открытии раздела.")
    if drift["status"] == "warning":
        st.warning("Обнаружено изменение распределения продаж.")
    st.info(drift["message"])
    if drift["columns"]:
        st.caption(
            f"История: {drift['reference_start']} — {drift['reference_end']}; "
            f"новое окно: {drift['current_start']} — {drift['current_end']}"
        )
        st.dataframe(pd.DataFrame(drift["columns"]), hide_index=True, width="stretch")
    if monitoring["html_available"] and st.button("Подготовить отчёт Evidently"):
        st.download_button(
            "Скачать HTML",
            call_api("GET", "/api/v1/monitoring/html"),
            file_name="drift.html",
            mime="text/html",
        )
    st.markdown("**Ошибка сохранённых прогнозов**")
    warnings_count = sum(
        row["status"] == "warning" and row["model_id"] == monitoring["model_id"]
        for row in monitoring["quality"]
    )
    if warnings_count:
        st.warning(f"У активной модели обнаружено ухудшение качества: {warnings_count} групп.")
    st.caption(
        "Сравниваются первое и последнее полные неперекрывающиеся окна одной модели, "
        "категории и горизонта. Повторные запросы не увеличивают выборку."
    )
    if not monitoring["quality"]:
        st.info("Сначала сохраните прогноз и загрузите фактические продажи за его горизонт.")
    for row in monitoring["quality"]:
        with st.expander(f"{row['category']} · {row['horizon']} дней · {row['model_id'][:8]}"):
            labels = {
                "waiting": "Ожидаем факты",
                "reference_only": "Ждём второе полное окно",
                "undefined": "WAPE не определён: сумма фактов равна нулю",
                "stable": "Ухудшение выше порога не обнаружено",
                "warning": "Качество ухудшилось",
            }
            if row["status"] == "warning":
                st.warning(labels[row["status"]])
            else:
                st.info(labels[row["status"]])
            st.write(
                {
                    "Полных прогнозов": row["matured"],
                    "Ожидают факты": row["pending"],
                    "Первый срез": row["reference_origin"],
                    "Последний срез": row["current_origin"],
                    "Порог WAPE, %": row["threshold_wape"],
                }
            )
            if row["reference"]:
                st.write({"Первое окно": row["reference"], "Последнее окно": row["current"]})


def main() -> None:
    st.set_page_config(page_title="Demon Plus", page_icon="📈", layout="wide")
    st.title("Demon Plus")
    st.caption("Прогноз продаж · Планирование запасов · Качество модели")
    health = call_api("GET", "/health")
    if not health["ready"]:
        st.warning(health["message"])
        st.stop()
    status = cast(dict[str, Any], call_api("GET", "/api/v1/status"))
    st.sidebar.title("Рабочее пространство")
    page = st.sidebar.radio("Раздел", ["Прогноз", "Запасы", "Данные", "Качество"])
    st.sidebar.info(
        f"Историческое демо Olist\n\nДанные по {status['last_date']}"
        f"\n\nМодель: {status['model_name']}"
    )
    st.sidebar.caption(
        "Оценки запасов относятся к категориям и не гарантируют наличие конкретного товара."
    )
    if page == "Прогноз":
        show_forecast(status)
    elif page == "Запасы":
        show_inventory(status)
    elif page == "Данные":
        show_data(status)
    else:
        show_quality()


if __name__ == "__main__":
    main()
