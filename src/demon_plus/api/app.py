"""FastAPI factory; run one worker for the local job executor."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, cast
from uuid import uuid4

import structlog
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from demon_plus.api.contracts import (
    ActiveModel,
    ForecastRequest,
    ForecastResponse,
    ImportResponse,
    InventoryRequest,
    InventoryResponse,
    Job,
    StatusResponse,
)
from demon_plus.config import Settings
from demon_plus.errors import DataError
from demon_plus.logging import configure_logging
from demon_plus.service import Service


def create_app(settings: Settings | None = None, service: Service | None = None) -> FastAPI:
    configuration = settings or Settings()
    configure_logging(configuration.log_level)
    logger = structlog.get_logger()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        instance = service or Service(configuration)
        app.state.service = instance
        app.state.startup_error = None
        try:
            instance.bootstrap()
        except (DataError, OSError, ValueError, KeyError):
            app.state.startup_error = (
                "Подготовьте модель командами fetch-data и train, затем перезапустите API."
            )
            logger.warning("application_not_ready")
        yield
        if service is None:
            instance.close()

    app = FastAPI(
        title="Demon Plus",
        version="0.1.0",
        lifespan=lifespan,
        description="Локальный MVP: исторические прогнозы по категориям, запасы и версии продаж.",
    )

    def current(request: Request) -> Service:
        if request.app.state.startup_error:
            raise HTTPException(503, detail=request.app.state.startup_error)
        return cast(Service, request.app.state.service)

    @app.middleware("http")
    async def access_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            request_id=request_id,
        )
        return response

    @app.exception_handler(DataError)
    async def data_error(request: Request, exc: DataError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Проверьте поля запроса и допустимые значения.",
                "errors": [
                    {"field": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
                    for error in exc.errors()
                ],
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("request_failed", error_type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": "Ошибка сервиса. Проверьте локальные артефакты и журнал API."},
        )

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        return {
            "status": "ok",
            "ready": request.app.state.startup_error is None,
            "message": request.app.state.startup_error,
        }

    @app.get("/api/v1/status", response_model=StatusResponse)
    def status(request: Request) -> StatusResponse:
        return current(request).status()

    @app.post("/api/v1/forecasts", response_model=ForecastResponse)
    def predict(payload: ForecastRequest, request: Request) -> ForecastResponse:
        return current(request).forecast(payload)

    @app.get("/api/v1/forecasts/{forecast_id}", response_model=ForecastResponse)
    def saved_forecast(forecast_id: str, request: Request) -> ForecastResponse:
        value = current(request).store.forecast_payload(forecast_id)
        if value is None:
            raise HTTPException(404, detail="Прогноз не найден.")
        return ForecastResponse.model_validate_json(value)

    @app.post("/api/v1/inventory", response_model=InventoryResponse)
    def inventory(payload: InventoryRequest, request: Request) -> InventoryResponse:
        return current(request).inventory(payload)

    @app.get("/api/v1/sales/template")
    def template(request: Request) -> Response:
        return Response(
            current(request).template(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="sales-template.csv"'},
        )

    @app.post("/api/v1/sales", response_model=ImportResponse)
    async def import_sales(request: Request, file: Annotated[UploadFile, File()]) -> ImportResponse:
        try:
            content = await file.read(configuration.max_upload_bytes + 1)
        finally:
            await file.close()
        return current(request).import_sales(content)

    @app.get("/api/v1/metrics")
    def metrics(request: Request) -> dict[str, object]:
        return current(request).metrics()

    @app.get("/api/v1/jobs", response_model=list[Job])
    def jobs(request: Request) -> list[Job]:
        return current(request).jobs()

    @app.post("/api/v1/jobs/retrain", response_model=Job, status_code=202)
    def retrain(request: Request) -> Job:
        try:
            return current(request).start_retraining()
        except DataError as exc:
            raise HTTPException(409, detail=str(exc)) from exc

    @app.post("/api/v1/jobs/{job_id}/activate", response_model=ActiveModel)
    def activate(job_id: str, request: Request) -> ActiveModel:
        try:
            return current(request).activate(job_id)
        except DataError as exc:
            raise HTTPException(409, detail=str(exc)) from exc

    return app
