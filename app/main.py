from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
import contextlib
from datetime import datetime, timezone
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
    TRANSLATION_WORKFLOW_BATCH,
    TRANSLATION_WORKFLOW_CANONICAL,
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

REASONING_OPTIONS = {
    LLM_PROVIDER_OPENAI: {
        "__default__": ["none", "minimal", "low", "medium", "high", "xhigh"],
        # Source: OpenAI model pages (Jul 2026)
        "gpt-5.5": ["none", "low", "medium", "high", "xhigh"],
        "gpt-5.4": ["none", "low", "medium", "high", "xhigh"],
        "gpt-5.4-mini": ["none", "low", "medium", "high", "xhigh"],
        "gpt-5.4-nano": ["none", "low", "medium", "high", "xhigh"],
        # Older GPT-5 mini/nano generations may not support "none"/"xhigh";
        # keep these constrained to their documented legacy levels.
        "gpt-5-mini": ["minimal", "low", "medium", "high"],
        "gpt-5-nano": ["minimal", "low", "medium", "high"],
    },
    LLM_PROVIDER_GEMINI: {
        "__default__": ["low", "medium", "high"],
        # Source: Gemini thinking guide + Gemma-on-Gemini docs (Jul 2026)
        "gemini-2.5-flash-lite": ["low", "medium", "high"],
        "gemini-2.5-flash": ["low", "medium", "high"],
        "gemini-2.5-pro": ["low", "medium", "high"],
        "gemini-3-flash-preview": ["low", "medium", "high"],
        "gemini-3.1-flash-lite": ["low", "medium", "high"],
        "gemma-4-31b-it": ["high"],
    },
    LLM_PROVIDER_DEEPSEEK: {
        "__default__": ["high", "max"],
        # Source: DeepSeek Chat Completions docs (Jul 2026)
        "deepseek-v4-flash": ["high", "max"],
        "deepseek-v4-pro": ["high", "max"],
    },
}

REASONING_API_FIELD_BY_PROVIDER = {
    LLM_PROVIDER_OPENAI: "reasoning.effort",
    LLM_PROVIDER_GEMINI: "generationConfig.thinkingConfig.thinkingLevel",
    LLM_PROVIDER_DEEPSEEK: "reasoning_effort",
}

WORKFLOW_MODE_OPTIONS = [
    {
        "value": TRANSLATION_WORKFLOW_BATCH,
        "label": "Single-pass batch translate + place",
    },
    {
        "value": TRANSLATION_WORKFLOW_CANONICAL,
        "label": "Two-pass: canonical translation, then placement",
    },
]


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
    translation_workflow: str | None,
) -> tuple[str, str | None, str, str, bool]:
    if not testing_mode:
        return (
            LLM_PROVIDER_OPENAI,
            None,
            POSITION_VARIANT_STANDARD,
            TRANSLATION_WORKFLOW_BATCH,
            False,
        )

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
    workflow = (translation_workflow or TRANSLATION_WORKFLOW_BATCH).strip()
    if workflow not in {TRANSLATION_WORKFLOW_BATCH, TRANSLATION_WORKFLOW_CANONICAL}:
        raise HTTPException(status_code=400, detail="Invalid translation workflow mode.")
    return (provider, model, variant, workflow, True)


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


def _resolve_supported_reasoning_efforts(provider: str, model_name: str | None) -> tuple[str, ...]:
    provider_map = REASONING_OPTIONS.get(provider, {})
    cleaned_model = (model_name or "").strip()
    if cleaned_model:
        model_specific = provider_map.get(cleaned_model)
        if model_specific:
            return tuple(model_specific)
    default_values = provider_map.get("__default__", [])
    return tuple(default_values)


def _resolve_effective_reasoning_effort(
    *,
    settings: Settings,
    provider: str,
    model_name: str,
    selected_reasoning_effort: str | None,
) -> str | None:
    cleaned_selected = (selected_reasoning_effort or "").strip() or None
    if cleaned_selected is None:
        if provider == LLM_PROVIDER_OPENAI:
            return settings.openai_reasoning_effort
        return None

    allowed = _resolve_supported_reasoning_efforts(provider, model_name)
    if allowed and cleaned_selected not in allowed:
        allowed_label = ", ".join(allowed)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid reasoning effort '{cleaned_selected}' for {provider}:{model_name}. "
                f"Allowed values: {allowed_label}."
            ),
        )
    return cleaned_selected


def _resolve_reasoning_api_field(provider: str, model_name: str) -> str:
    if provider == LLM_PROVIDER_GEMINI and model_name.startswith("gemini-2.5-"):
        return "generationConfig.thinkingConfig.thinkingBudget"
    return REASONING_API_FIELD_BY_PROVIDER.get(provider, "n/a")


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
    app.state.last_resume_attempt = {}

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

    def _start_processing_task(job_id: str, *, reason: str) -> bool:
        active_task = app.state.active_tasks.get(job_id)
        if active_task is not None and not active_task.done():
            return False
        task = asyncio.create_task(
            process_job(
                job_store=app.state.job_store,
                settings=app.state.settings,
                openai_service=app.state.openai_service,
                job_id=job_id,
            )
        )
        app.state.active_tasks[job_id] = task

        def _task_done(_: asyncio.Task, current_job_id: str = job_id) -> None:
            app.state.active_tasks.pop(current_job_id, None)

        task.add_done_callback(_task_done)
        LOGGER.info("Started processing task job_id=%s reason=%s", job_id, reason)
        return True

    def _maybe_resume_stalled_job(manifest) -> None:
        if manifest.status in {JobStatus.COMPLETE, JobStatus.FAILED}:
            return
        now = datetime.now(tz=timezone.utc)
        seconds_since_update = (now - manifest.updated_at).total_seconds()
        if seconds_since_update < app.state.settings.job_stale_after_seconds:
            return
        cooldown = max(1, app.state.settings.job_resume_cooldown_seconds)
        last_attempt = app.state.last_resume_attempt.get(manifest.job_id)
        if last_attempt and (now - last_attempt).total_seconds() < cooldown:
            return
        app.state.last_resume_attempt[manifest.job_id] = now
        _start_processing_task(manifest.job_id, reason="stale_job_resume")

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
                "workflow_mode_options": WORKFLOW_MODE_OPTIONS,
                "model_options_json": json.dumps(MODEL_OPTIONS),
                "reasoning_options_json": json.dumps(REASONING_OPTIONS),
                "reasoning_api_field_json": json.dumps(REASONING_API_FIELD_BY_PROVIDER),
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
        llm_reasoning_effort: str | None = Form(None),
        positioning_variant: str | None = Form(None),
        translation_workflow: str | None = Form(None),
        owned_batch_size: str | None = Form(None),
        context_pages: str | None = Form(None),
    ):
        raw_selected_provider = (llm_provider or "").strip().lower() or None
        raw_selected_model = (llm_model or "").strip() or None
        raw_selected_reasoning_effort = (llm_reasoning_effort or "").strip() or None
        raw_selected_variant = (positioning_variant or "").strip() or None
        raw_selected_workflow = (translation_workflow or "").strip() or None
        raw_selected_owned_batch_size = (owned_batch_size or "").strip() or None
        raw_selected_context_pages = (context_pages or "").strip() or None

        resolved_language = _resolve_target_language(target_language or "", custom_target_language)
        (
            provider,
            model_override,
            position_variant,
            resolved_workflow_mode,
            resolved_testing_mode,
        ) = _resolve_testing_options(
            testing_mode=_to_bool(testing_mode),
            llm_provider=llm_provider,
            llm_model=llm_model,
            positioning_variant=positioning_variant,
            translation_workflow=translation_workflow,
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
        effective_reasoning_effort = _resolve_effective_reasoning_effort(
            settings=app.state.settings,
            provider=provider,
            model_name=effective_model,
            selected_reasoning_effort=llm_reasoning_effort if resolved_testing_mode else None,
        )
        reasoning_api_field = _resolve_reasoning_api_field(provider, effective_model)
        LOGGER.info(
            (
                "Received upload filename=%s bytes=%s target_language=%s "
                "selected_testing_mode=%s selected_provider=%s selected_model=%s "
                "selected_reasoning_effort=%s "
                "selected_position_variant=%s selected_workflow=%s "
                "selected_owned_batch_size=%s selected_context_pages=%s "
                "effective_provider=%s effective_model=%s effective_reasoning_effort=%s "
                "reasoning_effort_field=%s "
                "effective_position_variant=%s effective_workflow=%s "
                "effective_owned_batch_size=%s effective_context_pages=%s"
            ),
            pdf_file.filename,
            len(payload),
            resolved_language,
            resolved_testing_mode,
            raw_selected_provider or "default",
            raw_selected_model or "default",
            raw_selected_reasoning_effort or "provider_default",
            raw_selected_variant or "default",
            raw_selected_workflow or "default",
            raw_selected_owned_batch_size or "default",
            raw_selected_context_pages or "default",
            provider,
            effective_model,
            effective_reasoning_effort or "provider_default",
            reasoning_api_field,
            position_variant,
            resolved_workflow_mode,
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
            llm_reasoning_effort=effective_reasoning_effort,
            translation_workflow=resolved_workflow_mode,
            positioning_variant=position_variant,
            testing_mode=resolved_testing_mode,
            owned_batch_size_override=owned_batch_size_override,
            context_pages_override=context_pages_override,
        )
        app.state.job_store.input_pdf_path(manifest.job_id).write_bytes(payload)
        app.state.job_store.persist_input_pdf_artifact(job_id=manifest.job_id, payload=payload)
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

        _start_processing_task(manifest.job_id, reason="job_created")

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
        _maybe_resume_stalled_job(manifest)
        manifest = app.state.job_store.get_job(job_id) or manifest

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
                "updated_at": manifest.updated_at.isoformat(),
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

