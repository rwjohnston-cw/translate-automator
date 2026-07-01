from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
import contextlib
from pathlib import Path
import json

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import Settings
from app.jobs import JobStore, process_job
from app.models import (
    LLM_PROVIDER_DEEPSEEK,
    LLM_PROVIDER_GEMINI,
    LLM_PROVIDER_OPENAI,
    POSITION_VARIANT_CENTERED_5,
    POSITION_VARIANT_GRID_20,
    POSITION_VARIANT_SPLIT_6,
    POSITION_VARIANT_STANDARD,
    JobStatus,
)
from app.openai_service import OpenAIService
from app.pdf_processing import PDFValidationError, validate_pdf_upload
from app.rate_limit import InMemoryRateLimiter, RateLimitRule, resolve_client_ip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

LANGUAGE_OPTIONS = [
    "English",
    "German",
    "French",
    "Italian",
    "Spanish",
    "Dutch",
    "Portuguese",
    "Polish",
    "Czech",
    "Swedish",
    "Danish",
    "Norwegian",
    "Finnish",
    "Japanese",
    "Korean",
    "Simplified Chinese",
    "Traditional Chinese",
    "Arabic",
    "Hebrew",
    "Other...",
]

PROVIDER_OPTIONS = [
    {"value": LLM_PROVIDER_OPENAI, "label": "OpenAI"},
    {"value": LLM_PROVIDER_GEMINI, "label": "Gemini API"},
    {"value": LLM_PROVIDER_DEEPSEEK, "label": "DeepSeek"},
]

POSITION_VARIANT_OPTIONS = [
    {"value": POSITION_VARIANT_STANDARD, "label": "3-point (top / middle / bottom)"},
    {"value": POSITION_VARIANT_CENTERED_5, "label": "5-point centered"},
    {"value": POSITION_VARIANT_SPLIT_6, "label": "6-point split (left / right)"},
    {
        "value": POSITION_VARIANT_GRID_20,
        "label": "20-point grid (a1-a4, b1-b4, c1-c4, d1-d4, e1-e5)",
    },
]

MODEL_OPTIONS = {
    LLM_PROVIDER_OPENAI: [
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5-mini",
        "gpt-5-nano",
    ],
    LLM_PROVIDER_GEMINI: [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemma-4-31b-it",
    ],
    LLM_PROVIDER_DEEPSEEK: [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ],
}


def _resolve_target_language(target_language: str, custom_target_language: str | None) -> str:
    raw = (target_language or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Please choose a target language.")
    if raw == "Other...":
        custom = (custom_target_language or "").strip()
        if not custom:
            raise HTTPException(
                status_code=400,
                detail="Please enter a custom target language.",
            )
        return custom
    return raw


async def _read_upload_bytes(upload: UploadFile, max_upload_bytes: int) -> bytes:
    total = 0
    chunks: list[bytes] = []
    max_upload_mb = max_upload_bytes / (1024 * 1024)
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"The uploaded file is too large. Maximum upload size is {max_upload_mb:g} MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _cleanup_loop(app: FastAPI) -> None:
    while True:
        try:
            removed = app.state.job_store.cleanup_expired()
            if removed:
                LOGGER.info("Removed expired jobs count=%s", removed)
        except Exception:
            LOGGER.exception("Periodic cleanup failure")
        await asyncio.sleep(app.state.settings.cleanup_interval_seconds)


def _to_bool(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_override(
    raw_value: str | None,
    *,
    field_label: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field_label} must be an integer.") from exc
    if parsed < minimum or parsed > maximum:
        raise HTTPException(
            status_code=400,
            detail=f"{field_label} must be between {minimum} and {maximum}.",
        )
    return parsed


def _resolve_testing_options(
    *,
    testing_mode: bool,
    llm_provider: str | None,
    llm_model: str | None,
    positioning_variant: str | None,
) -> tuple[str, str | None, str, bool]:
    if not testing_mode:
        return (LLM_PROVIDER_OPENAI, None, POSITION_VARIANT_STANDARD, False)

    provider = (llm_provider or LLM_PROVIDER_OPENAI).strip().lower()
    if provider not in {LLM_PROVIDER_OPENAI, LLM_PROVIDER_GEMINI, LLM_PROVIDER_DEEPSEEK}:
        raise HTTPException(status_code=400, detail="Invalid testing provider.")

    variant = (positioning_variant or POSITION_VARIANT_STANDARD).strip()
    if variant not in {
        POSITION_VARIANT_STANDARD,
        POSITION_VARIANT_CENTERED_5,
        POSITION_VARIANT_SPLIT_6,
        POSITION_VARIANT_GRID_20,
    }:
        raise HTTPException(status_code=400, detail="Invalid page positioning variant.")

    model = (llm_model or "").strip() or None
    return (provider, model, variant, True)


def _resolve_effective_model(settings: Settings, provider: str, model_override: str | None) -> str:
    if model_override and model_override.strip():
        return model_override.strip()
    if provider == LLM_PROVIDER_OPENAI:
        return settings.openai_model
    if provider == LLM_PROVIDER_GEMINI:
        return settings.gemini_model
    if provider == LLM_PROVIDER_DEEPSEEK:
        return settings.deepseek_model
    return settings.openai_model


def _resolve_rate_limit_rule(request: Request, settings: Settings) -> RateLimitRule | None:
    method = request.method.upper()
    path = request.url.path

    if method == "POST" and path == "/api/jobs":
        limit = settings.rate_limit_create_requests
        window_seconds = settings.rate_limit_create_window_seconds
        scope = "create_job"
    elif path.startswith("/api/"):
        limit = settings.rate_limit_api_requests
        window_seconds = settings.rate_limit_api_window_seconds
        scope = "api"
    else:
        return None

    if limit <= 0 or window_seconds <= 0:
        return None
    return RateLimitRule(scope=scope, limit=limit, window_seconds=window_seconds)


def create_app(
    *,
    settings: Settings | None = None,
    job_store: JobStore | None = None,
    openai_service: OpenAIService | None = None,
) -> FastAPI:
    cfg = settings or Settings()
    store = job_store or JobStore(cfg)
    service = openai_service or OpenAIService(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cleanup_task = asyncio.create_task(_cleanup_loop(app))
        app.state.active_tasks = {}
        try:
            yield
        finally:
            cleanup_task = app.state.cleanup_task
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task

            for task in list(app.state.active_tasks.values()):
                task.cancel()

    app = FastAPI(title="Vocal Score Translator", lifespan=lifespan)
    app.state.settings = cfg
    app.state.job_store = store
    app.state.openai_service = service
    app.state.rate_limiter = InMemoryRateLimiter()

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        settings = app.state.settings
        if not settings.rate_limit_enabled:
            return await call_next(request)

        rule = _resolve_rate_limit_rule(request, settings)
        if rule is None:
            return await call_next(request)

        client_ip = resolve_client_ip(request, trust_proxy_headers=settings.trust_proxy_headers)
        limited, retry_after = app.state.rate_limiter.hit(
            scope=rule.scope,
            client_key=client_ip,
            limit=rule.limit,
            window_seconds=rule.window_seconds,
        )
        if limited:
            LOGGER.warning(
                "Rate limit exceeded scope=%s ip=%s method=%s path=%s retry_after_s=%s",
                rule.scope,
                client_ip,
                request.method,
                request.url.path,
                retry_after,
            )
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please wait and try again."},
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    @app.get("/")
    async def index(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "language_options": LANGUAGE_OPTIONS,
                "max_upload_mb": app.state.settings.effective_max_upload_mb,
                "max_upload_mb_display": f"{app.state.settings.effective_max_upload_mb:g}",
                "default_owned_batch_size": app.state.settings.owned_batch_size,
                "default_context_pages": app.state.settings.context_pages,
                "provider_options": PROVIDER_OPTIONS,
                "position_variant_options": POSITION_VARIANT_OPTIONS,
                "model_options_json": json.dumps(MODEL_OPTIONS),
                "app_version": app.state.settings.resolved_app_version,
            },
        )

    @app.post("/api/jobs")
    async def create_job_api(
        pdf_file: UploadFile = File(...),
        target_language: str | None = Form(None),
        custom_target_language: str | None = Form(None),
        testing_mode: str | None = Form(None),
        llm_provider: str | None = Form(None),
        llm_model: str | None = Form(None),
        positioning_variant: str | None = Form(None),
        owned_batch_size: str | None = Form(None),
        context_pages: str | None = Form(None),
    ):
        raw_selected_provider = (llm_provider or "").strip().lower() or None
        raw_selected_model = (llm_model or "").strip() or None
        raw_selected_variant = (positioning_variant or "").strip() or None
        raw_selected_owned_batch_size = (owned_batch_size or "").strip() or None
        raw_selected_context_pages = (context_pages or "").strip() or None

        resolved_language = _resolve_target_language(target_language or "", custom_target_language)
        provider, model_override, position_variant, resolved_testing_mode = _resolve_testing_options(
            testing_mode=_to_bool(testing_mode),
            llm_provider=llm_provider,
            llm_model=llm_model,
            positioning_variant=positioning_variant,
        )
        owned_batch_size_override = None
        context_pages_override = None
        if resolved_testing_mode:
            owned_batch_size_override = _parse_int_override(
                owned_batch_size,
                field_label="Owned batch size",
                minimum=1,
                maximum=app.state.settings.max_pages,
            )
            context_pages_override = _parse_int_override(
                context_pages,
                field_label="Context pages",
                minimum=0,
                maximum=app.state.settings.max_pages,
            )
        if not pdf_file.filename:
            raise HTTPException(status_code=400, detail="Please choose a PDF file.")
        payload = await _read_upload_bytes(pdf_file, app.state.settings.max_upload_bytes)
        effective_owned_batch_size = (
            owned_batch_size_override
            if owned_batch_size_override is not None
            else app.state.settings.owned_batch_size
        )
        effective_context_pages = (
            context_pages_override
            if context_pages_override is not None
            else app.state.settings.context_pages
        )
        effective_model = _resolve_effective_model(app.state.settings, provider, model_override)
        LOGGER.info(
            (
                "Received upload filename=%s bytes=%s target_language=%s "
                "selected_testing_mode=%s selected_provider=%s selected_model=%s "
                "selected_position_variant=%s selected_owned_batch_size=%s selected_context_pages=%s "
                "effective_provider=%s effective_model=%s effective_position_variant=%s "
                "effective_owned_batch_size=%s effective_context_pages=%s"
            ),
            pdf_file.filename,
            len(payload),
            resolved_language,
            resolved_testing_mode,
            raw_selected_provider or "default",
            raw_selected_model or "default",
            raw_selected_variant or "default",
            raw_selected_owned_batch_size or "default",
            raw_selected_context_pages or "default",
            provider,
            effective_model,
            position_variant,
            effective_owned_batch_size,
            effective_context_pages,
        )
        try:
            page_count = validate_pdf_upload(
                filename=pdf_file.filename,
                payload=payload,
                max_upload_bytes=app.state.settings.max_upload_bytes,
                max_pages=app.state.settings.max_pages,
            )
        except PDFValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        manifest = app.state.job_store.create_job(
            original_filename=pdf_file.filename,
            target_language=resolved_language,
            page_count=page_count,
            llm_provider=provider,
            llm_model=model_override,
            positioning_variant=position_variant,
            testing_mode=resolved_testing_mode,
            owned_batch_size_override=owned_batch_size_override,
            context_pages_override=context_pages_override,
        )
        app.state.job_store.input_pdf_path(manifest.job_id).write_bytes(payload)
        LOGGER.info(
            "Stored upload job_id=%s input_path=%s page_count=%s",
            manifest.job_id,
            app.state.job_store.input_pdf_path(manifest.job_id),
            page_count,
        )
        app.state.job_store.set_status(
            job_id=manifest.job_id,
            status=JobStatus.QUEUED,
            message="Queued for processing...",
            progress=0.0,
        )

        task = asyncio.create_task(
            process_job(
                job_store=app.state.job_store,
                settings=app.state.settings,
                openai_service=app.state.openai_service,
                job_id=manifest.job_id,
            )
        )
        app.state.active_tasks[manifest.job_id] = task

        def _task_done(_: asyncio.Task, job_id: str = manifest.job_id) -> None:
            app.state.active_tasks.pop(job_id, None)

        task.add_done_callback(_task_done)

        return JSONResponse(
            {
                "job_id": manifest.job_id,
                "status_url": f"/api/jobs/{manifest.job_id}",
                "page_url": f"/jobs/{manifest.job_id}",
            }
        )

    @app.get("/jobs/{job_id}")
    async def job_page(job_id: str, request: Request):
        manifest = app.state.job_store.get_job(job_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return templates.TemplateResponse(
            request=request,
            name="job.html",
            context={
                "request": request,
                "manifest": manifest,
                "status_url": f"/api/jobs/{manifest.job_id}",
            },
        )

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str):
        app.state.job_store.cleanup_expired()
        manifest = app.state.job_store.get_job(job_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail="Job not found.")

        download_url = None
        log_url = None
        if manifest.status == JobStatus.COMPLETE and manifest.output_available:
            download_url = f"/api/jobs/{manifest.job_id}/download"
        if manifest.status == JobStatus.COMPLETE and manifest.log_available:
            log_url = f"/api/jobs/{manifest.job_id}/log"

        return JSONResponse(
            {
                "status": manifest.status.value,
                "message": manifest.message,
                "progress": manifest.progress,
                "current_batch": manifest.current_batch,
                "total_batches": manifest.total_batches,
                "download_url": download_url,
                "log_url": log_url,
                "error": manifest.error,
            }
        )

    @app.get("/api/jobs/{job_id}/download")
    async def download(job_id: str):
        app.state.job_store.cleanup_expired()
        manifest = app.state.job_store.get_job(job_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        output_path = app.state.job_store.output_pdf_path(job_id)
        if manifest.status != JobStatus.COMPLETE or not manifest.output_available:
            raise HTTPException(status_code=404, detail="Translated PDF not found.")
        if output_path.exists():
            return FileResponse(
                path=output_path,
                media_type="application/pdf",
                filename=manifest.download_filename or "translated_score.pdf",
            )
        cached_output = app.state.job_store.get_cached_output_pdf(job_id)
        if cached_output is None:
            raise HTTPException(status_code=404, detail="Translated PDF not found.")

        filename = manifest.download_filename or "translated_score.pdf"
        return Response(
            content=cached_output,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/jobs/{job_id}/log")
    async def download_log(job_id: str):
        app.state.job_store.cleanup_expired()
        manifest = app.state.job_store.get_job(job_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        log_path = app.state.job_store.llm_log_json_path(job_id)
        if manifest.status != JobStatus.COMPLETE or not manifest.log_available:
            raise HTTPException(status_code=404, detail="LLM log not found.")
        if log_path.exists():
            return FileResponse(
                path=log_path,
                media_type="application/json",
                filename=f"{manifest.safe_original_stem}_llm_log.json",
            )
        cached_log = app.state.job_store.get_cached_llm_log(job_id)
        if cached_log is None:
            raise HTTPException(status_code=404, detail="LLM log not found.")
        filename = f"{manifest.safe_original_stem}_llm_log.json"
        return Response(
            content=cached_log,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return app


app = create_app()

