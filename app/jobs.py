from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    import redis
except ImportError:  # pragma: no cover - optional in local test env
    redis = None  # type: ignore[assignment]

from app.config import Settings
from app.models import (
    BatchLLMLogEntry,
    CanonicalTranslationResult,
    JobLLMLog,
    JobManifest,
    JobStatus,
    JobTextResult,
    TRANSLATION_WORKFLOW_CANONICAL,
    TranslationPlacement,
    positions_for_variant,
)
from app.openai_service import OpenAIService, OpenAIServiceError, PermanentOpenAIError
from app.pdf_processing import (
    BatchSpec,
    PDFValidationError,
    build_batches,
    build_output_filename,
    clean_and_filter_batch_placements,
    create_translated_pdf,
    merge_batch_results,
    position_order_for_variant,
    render_pdf_pages_to_images,
    sanitize_stem,
    validate_pdf_upload,
)
from app.prompts import build_placement_from_canonical_prompt

LOGGER = logging.getLogger(__name__)


class JobStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.job_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.redis_client = None
        self.redis_key_prefix = settings.redis_key_prefix.strip() or "translate-automator"
        if settings.redis_url:
            if redis is None:
                raise RuntimeError("redis package is required when REDIS_URL is configured.")
            self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=False)
            LOGGER.info("JobStore configured with Redis key_prefix=%s", self.redis_key_prefix)
        else:
            LOGGER.warning("JobStore running without Redis; serverless polling may be inconsistent.")

    def _manifest_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "manifest.json"

    def job_dir(self, job_id: str) -> Path:
        validated = self.validate_job_id(job_id)
        directory = (self.root / validated).resolve()
        root_resolved = self.root.resolve()
        if root_resolved not in directory.parents and directory != root_resolved:
            raise ValueError("Invalid job path.")
        return directory

    def input_pdf_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "input.pdf"

    def rendered_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "rendered"

    def output_pdf_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "output.pdf"

    def translation_json_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "translation.json"

    def text_result_json_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "text_result.json"

    def llm_log_json_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "llm_log.json"

    def canonical_translation_json_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "canonical_translation.json"

    def checkpoint_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "checkpoints"

    def checkpoint_path(self, job_id: str, batch_index: int) -> Path:
        return self.checkpoint_dir(job_id) / f"batch_{batch_index:04d}.json"

    def _redis_manifest_key(self, job_id: str) -> str:
        return f"{self.redis_key_prefix}:job:{job_id}:manifest"

    def _redis_output_key(self, job_id: str) -> str:
        return f"{self.redis_key_prefix}:job:{job_id}:output_pdf"

    def _redis_log_key(self, job_id: str) -> str:
        return f"{self.redis_key_prefix}:job:{job_id}:llm_log"

    def _redis_input_key(self, job_id: str) -> str:
        return f"{self.redis_key_prefix}:job:{job_id}:input_pdf"

    def _redis_text_result_key(self, job_id: str) -> str:
        return f"{self.redis_key_prefix}:job:{job_id}:text_result"

    def _redis_canonical_key(self, job_id: str) -> str:
        return f"{self.redis_key_prefix}:job:{job_id}:canonical_translation"

    def _redis_checkpoint_hash_key(self, job_id: str) -> str:
        return f"{self.redis_key_prefix}:job:{job_id}:batch_checkpoints"

    def _redis_processing_lock_key(self, job_id: str) -> str:
        return f"{self.redis_key_prefix}:job:{job_id}:processing_lock"

    def _redis_ttl_seconds(self, manifest: JobManifest) -> int:
        seconds = int((manifest.expires_at - datetime.now(tz=timezone.utc)).total_seconds())
        return max(1, seconds)

    def _write_manifest_to_redis(self, manifest: JobManifest) -> None:
        if self.redis_client is None:
            return
        try:
            self.redis_client.set(
                self._redis_manifest_key(manifest.job_id),
                manifest.model_dump_json(indent=2).encode("utf-8"),
                ex=self._redis_ttl_seconds(manifest),
            )
        except Exception:
            LOGGER.exception("Failed to write manifest to redis job_id=%s", manifest.job_id)

    def _read_manifest_from_redis(self, job_id: str) -> JobManifest | None:
        if self.redis_client is None:
            return None
        try:
            payload_raw = self.redis_client.get(self._redis_manifest_key(job_id))
            if payload_raw is None:
                return None
            payload = json.loads(payload_raw.decode("utf-8"))
            return JobManifest.model_validate(payload)
        except Exception:
            LOGGER.exception("Failed to read manifest from redis job_id=%s", job_id)
            return None

    @staticmethod
    def validate_job_id(raw_job_id: str) -> str:
        try:
            parsed = uuid.UUID(raw_job_id)
        except Exception as exc:
            raise ValueError("Invalid job id.") from exc
        return str(parsed)

    def create_job(
        self,
        *,
        original_filename: str,
        target_language: str,
        page_count: int,
        llm_provider: str,
        llm_model: str | None,
        llm_reasoning_effort: str | None,
        translation_workflow: str,
        positioning_variant: str,
        testing_mode: bool,
        owned_batch_size_override: int | None = None,
        context_pages_override: int | None = None,
    ) -> JobManifest:
        job_id = str(uuid.uuid4())
        directory = self.job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=False)
        expires_at = datetime.now(tz=timezone.utc) + timedelta(minutes=self.settings.job_ttl_minutes)
        manifest = JobManifest(
            job_id=job_id,
            original_filename=original_filename,
            safe_original_stem=sanitize_stem(original_filename),
            target_language=target_language,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_reasoning_effort=llm_reasoning_effort,
            translation_workflow=translation_workflow,
            positioning_variant=positioning_variant,
            testing_mode=testing_mode,
            owned_batch_size_override=owned_batch_size_override,
            context_pages_override=context_pages_override,
            page_count=page_count,
            expires_at=expires_at,
        )
        self._write_manifest(manifest)
        LOGGER.info(
            (
                "Created job job_id=%s filename=%s target_language=%s page_count=%s "
                "testing_mode=%s provider=%s selected_model=%s selected_reasoning_effort=%s "
                "workflow=%s positioning_variant=%s "
                "selected_owned_batch_size=%s selected_context_pages=%s "
                "effective_owned_batch_size=%s effective_context_pages=%s expires_at=%s"
            ),
            manifest.job_id,
            manifest.original_filename,
            manifest.target_language,
            manifest.page_count,
            manifest.testing_mode,
            manifest.llm_provider,
            manifest.llm_model or "default",
            manifest.llm_reasoning_effort or "provider_default",
            manifest.translation_workflow,
            manifest.positioning_variant,
            manifest.owned_batch_size_override if manifest.owned_batch_size_override is not None else "default",
            manifest.context_pages_override if manifest.context_pages_override is not None else "default",
            manifest.owned_batch_size_override
            if manifest.owned_batch_size_override is not None
            else self.settings.owned_batch_size,
            manifest.context_pages_override
            if manifest.context_pages_override is not None
            else self.settings.context_pages,
            manifest.expires_at.isoformat(),
        )
        return manifest

    def _write_manifest(self, manifest: JobManifest) -> None:
        path = self._manifest_path(manifest.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        self._write_manifest_to_redis(manifest)

    def get_job(self, job_id: str) -> JobManifest | None:
        try:
            valid_job_id = self.validate_job_id(job_id)
        except ValueError:
            return None
        redis_manifest = self._read_manifest_from_redis(valid_job_id)
        if redis_manifest is not None:
            return redis_manifest
        path = self._manifest_path(valid_job_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            manifest = JobManifest.model_validate(payload)
            # Read-through repair: if Redis missed this manifest, republish it so
            # subsequent requests served by other serverless instances can find it.
            self._write_manifest_to_redis(manifest)
            return manifest
        except Exception:
            LOGGER.exception("Failed to read manifest job_id=%s", valid_job_id)
            return None

    def require_job(self, job_id: str) -> JobManifest:
        manifest = self.get_job(job_id)
        if manifest is None:
            raise FileNotFoundError("Job not found.")
        return manifest

    def update_job(self, job_id: str, **updates: object) -> JobManifest:
        manifest = self.require_job(job_id)
        for key, value in updates.items():
            setattr(manifest, key, value)
        manifest.touch()
        self._write_manifest(manifest)
        return manifest

    def set_status(
        self,
        *,
        job_id: str,
        status: JobStatus,
        message: str,
        progress: float,
        current_batch: int | None = None,
        total_batches: int | None = None,
        error: str | None = None,
    ) -> JobManifest:
        progress = min(1.0, max(0.0, progress))
        manifest = self.update_job(
            job_id,
            status=status,
            message=message,
            progress=progress,
            current_batch=current_batch,
            total_batches=total_batches,
            error=error,
        )
        LOGGER.info(
            "Job status job_id=%s status=%s progress=%.1f%% message=%s batch=%s/%s",
            job_id,
            status.value,
            progress * 100,
            message,
            current_batch if current_batch is not None else "-",
            total_batches if total_batches is not None else "-",
        )
        return manifest

    def mark_complete(self, *, job_id: str, download_filename: str) -> JobManifest:
        manifest = self.update_job(
            job_id,
            status=JobStatus.COMPLETE,
            message="Translation complete.",
            progress=1.0,
            current_batch=None,
            total_batches=None,
            error=None,
            download_filename=download_filename,
            output_available=True,
        )
        LOGGER.info(
            "Job complete job_id=%s download_filename=%s",
            job_id,
            download_filename,
        )
        return manifest

    def mark_log_available(self, *, job_id: str) -> JobManifest:
        return self.update_job(job_id, log_available=True)

    def mark_text_result_available(self, *, job_id: str) -> JobManifest:
        return self.update_job(job_id, text_result_available=True)

    def mark_failed(self, *, job_id: str, message: str) -> JobManifest:
        manifest = self.update_job(
            job_id,
            status=JobStatus.FAILED,
            message="Processing failed.",
            error=message,
            current_batch=None,
            total_batches=None,
        )
        LOGGER.warning("Job failed job_id=%s message=%s", job_id, message)
        return manifest

    def persist_llm_log_artifact(self, *, job_id: str) -> None:
        if self.redis_client is None:
            return
        manifest = self.get_job(job_id)
        if manifest is None:
            return
        log_path = self.llm_log_json_path(job_id)
        if not log_path.exists():
            return
        try:
            self.redis_client.set(
                self._redis_log_key(job_id),
                log_path.read_bytes(),
                ex=self._redis_ttl_seconds(manifest),
            )
        except Exception:
            LOGGER.exception("Failed to persist llm log artifact to redis job_id=%s", job_id)

    def persist_output_pdf_artifact(self, *, job_id: str) -> None:
        if self.redis_client is None:
            return
        manifest = self.get_job(job_id)
        if manifest is None:
            return
        output_path = self.output_pdf_path(job_id)
        if not output_path.exists():
            return
        try:
            self.redis_client.set(
                self._redis_output_key(job_id),
                output_path.read_bytes(),
                ex=self._redis_ttl_seconds(manifest),
            )
        except Exception:
            LOGGER.exception("Failed to persist output artifact to redis job_id=%s", job_id)

    def persist_input_pdf_artifact(self, *, job_id: str, payload: bytes | None = None) -> None:
        if self.redis_client is None:
            return
        manifest = self.get_job(job_id)
        if manifest is None:
            return
        if payload is None:
            input_path = self.input_pdf_path(job_id)
            if not input_path.exists():
                return
            payload = input_path.read_bytes()
        try:
            self.redis_client.set(
                self._redis_input_key(job_id),
                payload,
                ex=self._redis_ttl_seconds(manifest),
            )
        except Exception:
            LOGGER.exception("Failed to persist input artifact to redis job_id=%s", job_id)

    def get_cached_output_pdf(self, job_id: str) -> bytes | None:
        if self.redis_client is None:
            return None
        try:
            payload = self.redis_client.get(self._redis_output_key(job_id))
            return payload if payload is not None else None
        except Exception:
            LOGGER.exception("Failed to read output artifact from redis job_id=%s", job_id)
            return None

    def get_cached_input_pdf(self, job_id: str) -> bytes | None:
        if self.redis_client is None:
            return None
        try:
            payload = self.redis_client.get(self._redis_input_key(job_id))
            return payload if payload is not None else None
        except Exception:
            LOGGER.exception("Failed to read input artifact from redis job_id=%s", job_id)
            return None

    def get_cached_llm_log(self, job_id: str) -> bytes | None:
        if self.redis_client is None:
            return None
        try:
            payload = self.redis_client.get(self._redis_log_key(job_id))
            return payload if payload is not None else None
        except Exception:
            LOGGER.exception("Failed to read llm log artifact from redis job_id=%s", job_id)
            return None

    def save_text_result(self, *, job_id: str, text_result: JobTextResult) -> None:
        path = self.text_result_json_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = text_result.model_dump_json(indent=2).encode("utf-8")
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_bytes(payload)
        tmp_path.replace(path)

        if self.redis_client is None:
            return
        manifest = self.get_job(job_id)
        if manifest is None:
            return
        try:
            self.redis_client.set(
                self._redis_text_result_key(job_id),
                payload,
                ex=self._redis_ttl_seconds(manifest),
            )
        except Exception:
            LOGGER.exception("Failed to persist text result to redis job_id=%s", job_id)

    def load_text_result(self, job_id: str) -> JobTextResult | None:
        path = self.text_result_json_path(job_id)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return JobTextResult.model_validate(payload)
            except Exception:
                LOGGER.exception("Failed to parse text result path=%s", path)

        if self.redis_client is None:
            return None
        try:
            cached = self.redis_client.get(self._redis_text_result_key(job_id))
        except Exception:
            LOGGER.exception("Failed to read text result from redis job_id=%s", job_id)
            return None
        if cached is None:
            return None
        try:
            payload = json.loads(cached.decode("utf-8"))
            text_result = JobTextResult.model_validate(payload)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text_result.model_dump_json(indent=2), encoding="utf-8")
            return text_result
        except Exception:
            LOGGER.exception("Failed to parse cached text result job_id=%s", job_id)
            return None

    def save_canonical_translation(
        self,
        *,
        job_id: str,
        canonical: CanonicalTranslationResult,
    ) -> None:
        path = self.canonical_translation_json_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical.model_dump_json(indent=2).encode("utf-8")
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_bytes(payload)
        tmp_path.replace(path)

        if self.redis_client is None:
            return
        manifest = self.get_job(job_id)
        if manifest is None:
            return
        try:
            self.redis_client.set(
                self._redis_canonical_key(job_id),
                payload,
                ex=self._redis_ttl_seconds(manifest),
            )
        except Exception:
            LOGGER.exception("Failed to persist canonical translation to redis job_id=%s", job_id)

    def load_canonical_translation(self, job_id: str) -> CanonicalTranslationResult | None:
        path = self.canonical_translation_json_path(job_id)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return CanonicalTranslationResult.model_validate(payload)
            except Exception:
                LOGGER.exception("Failed to parse canonical translation path=%s", path)

        if self.redis_client is None:
            return None
        try:
            cached = self.redis_client.get(self._redis_canonical_key(job_id))
        except Exception:
            LOGGER.exception("Failed to read canonical translation from redis job_id=%s", job_id)
            return None
        if cached is None:
            return None
        try:
            payload = json.loads(cached.decode("utf-8"))
            canonical = CanonicalTranslationResult.model_validate(payload)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(canonical.model_dump_json(indent=2), encoding="utf-8")
            return canonical
        except Exception:
            LOGGER.exception("Failed to parse cached canonical translation job_id=%s", job_id)
            return None

    def clear_canonical_translation(self, job_id: str) -> None:
        path = self.canonical_translation_json_path(job_id)
        if path.exists():
            with contextlib.suppress(Exception):
                path.unlink()
        if self.redis_client is not None:
            with contextlib.suppress(Exception):
                self.redis_client.delete(self._redis_canonical_key(job_id))

    def create_processing_lock(self, job_id: str):
        if self.redis_client is None:
            return None
        timeout = max(30, self.settings.job_processing_lock_ttl_seconds)
        return self.redis_client.lock(
            self._redis_processing_lock_key(job_id),
            timeout=timeout,
            blocking=False,
        )

    def save_batch_checkpoint(
        self,
        *,
        job_id: str,
        batch_index: int,
        owned_start: int,
        owned_end: int,
        placements: list[TranslationPlacement],
        full_translation: str,
        full_source_text: str = "",
        source_language: str = "",
        batch_log: BatchLLMLogEntry,
    ) -> None:
        payload = {
            "batch_index": batch_index,
            "owned_start": owned_start,
            "owned_end": owned_end,
            "placements": [item.model_dump() for item in placements],
            "full_translation": full_translation,
            "full_source_text": full_source_text,
            "source_language": source_language,
            "batch_log": batch_log.model_dump(),
        }
        manifest = self.get_job(job_id)

        path = self.checkpoint_path(job_id, batch_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(path)

        if self.redis_client is None:
            return
        if manifest is None:
            return
        try:
            self.redis_client.hset(
                self._redis_checkpoint_hash_key(job_id),
                str(batch_index),
                json.dumps(payload).encode("utf-8"),
            )
            self.redis_client.expire(
                self._redis_checkpoint_hash_key(job_id),
                self._redis_ttl_seconds(manifest),
            )
        except Exception:
            LOGGER.exception("Failed to persist batch checkpoint to redis job_id=%s batch=%s", job_id, batch_index)

    def _parse_checkpoint_payload(
        self, payload: dict[str, Any]
    ) -> tuple[int, list[TranslationPlacement], str, str, str, BatchLLMLogEntry]:
        batch_index = int(payload["batch_index"])
        placements_raw = payload.get("placements") or []
        placements = [TranslationPlacement.model_validate(item) for item in placements_raw]
        full_translation = str(payload.get("full_translation") or "")
        full_source_text = str(payload.get("full_source_text") or "")
        source_language = str(payload.get("source_language") or "")
        batch_log = BatchLLMLogEntry.model_validate(payload["batch_log"])
        return batch_index, placements, full_translation, full_source_text, source_language, batch_log

    def load_batch_checkpoints(
        self, job_id: str
    ) -> dict[int, tuple[list[TranslationPlacement], str, str, str, BatchLLMLogEntry]]:
        checkpoints: dict[int, tuple[list[TranslationPlacement], str, str, str, BatchLLMLogEntry]] = {}
        checkpoint_dir = self.checkpoint_dir(job_id)
        if checkpoint_dir.exists():
            for path in sorted(checkpoint_dir.glob("batch_*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    index, placements, full_translation, full_source_text, source_language, batch_log = (
                        self._parse_checkpoint_payload(payload)
                    )
                    checkpoints[index] = (
                        placements,
                        full_translation,
                        full_source_text,
                        source_language,
                        batch_log,
                    )
                except Exception:
                    LOGGER.exception("Failed to parse local checkpoint path=%s", path)

        if self.redis_client is None:
            return checkpoints
        try:
            raw_map = self.redis_client.hgetall(self._redis_checkpoint_hash_key(job_id))
        except Exception:
            LOGGER.exception("Failed to read checkpoints from redis job_id=%s", job_id)
            return checkpoints

        for raw_index, raw_payload in raw_map.items():
            try:
                payload = json.loads(raw_payload.decode("utf-8"))
                index, placements, full_translation, full_source_text, source_language, batch_log = (
                    self._parse_checkpoint_payload(payload)
                )
                if index not in checkpoints:
                    checkpoints[index] = (
                        placements,
                        full_translation,
                        full_source_text,
                        source_language,
                        batch_log,
                    )
            except Exception:
                LOGGER.exception(
                    "Failed to parse redis checkpoint job_id=%s batch=%s",
                    job_id,
                    raw_index.decode("utf-8", errors="ignore")
                    if isinstance(raw_index, (bytes, bytearray))
                    else raw_index,
                )
        return checkpoints

    def clear_batch_checkpoints(self, job_id: str) -> None:
        checkpoint_dir = self.checkpoint_dir(job_id)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        if self.redis_client is not None:
            try:
                self.redis_client.delete(self._redis_checkpoint_hash_key(job_id))
            except Exception:
                LOGGER.exception("Failed to clear redis checkpoints job_id=%s", job_id)

    def cleanup_expired(self) -> int:
        now = datetime.now(tz=timezone.utc)
        removed = 0
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in self.root.iterdir():
            if not directory.is_dir():
                continue
            manifest_path = directory / "manifest.json"
            expire = None
            if manifest_path.exists():
                try:
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest = JobManifest.model_validate(payload)
                    expire = manifest.expires_at
                except Exception:
                    LOGGER.warning("Could not parse manifest during cleanup path=%s", directory)
            if expire is None:
                modified = datetime.fromtimestamp(directory.stat().st_mtime, tz=timezone.utc)
                expire = modified + timedelta(minutes=self.settings.job_ttl_minutes)
            if expire < now:
                shutil.rmtree(directory, ignore_errors=True)
                removed += 1
                LOGGER.info("Removed expired job directory path=%s", directory)
        return removed


def _sort_batch_placements(
    placements: list[TranslationPlacement],
    position_order: dict[str, int],
) -> list[TranslationPlacement]:
    return sorted(placements, key=lambda item: (item.page, position_order.get(item.position, 999)))


def _canonical_reference_text(canonical: CanonicalTranslationResult) -> str:
    lines: list[str] = [
        "CANONICAL_TRANSLATION_REFERENCE",
        "",
        "SOURCE_LANGUAGE:",
        canonical.source_language or "[unknown]",
        "",
        "FULL_SOURCE_TEXT:",
        canonical.full_source_text or "[empty]",
        "",
        "FULL_TRANSLATION:",
        canonical.full_translation or "[empty]",
        "",
        "ALIGNED_LINES:",
    ]
    if not canonical.aligned_lines:
        lines.append("[none]")
    else:
        for idx, item in enumerate(canonical.aligned_lines, start=1):
            lines.append(f"{idx}. SRC: {item.source_text}")
            lines.append(f"   TR: {item.translated_text}")
    lines.extend(
        [
            "",
            "Use this canonical reference as translation truth.",
            "Do not independently re-translate wording unless OCR is clearly mismatched and correction is essential.",
            "Prefer fragments that are exact substrings of translated canonical lines where possible.",
        ]
    )
    return "\n".join(lines)


async def process_job(
    *,
    job_store: JobStore,
    settings: Settings,
    openai_service: OpenAIService,
    job_id: str,
) -> None:
    started = perf_counter()
    processing_lock = job_store.create_processing_lock(job_id)
    lock_heartbeat_task: asyncio.Task[None] | None = None
    if processing_lock is not None:
        try:
            acquired = bool(processing_lock.acquire(blocking=False))
        except Exception:
            LOGGER.exception("Failed to acquire processing lock job_id=%s", job_id)
            return
        if not acquired:
            LOGGER.info("Processing already active elsewhere job_id=%s", job_id)
            return

        heartbeat_interval = max(5, settings.job_processing_lock_heartbeat_seconds)
        lock_ttl = max(30, settings.job_processing_lock_ttl_seconds)

        async def _refresh_processing_lock() -> None:
            while True:
                await asyncio.sleep(heartbeat_interval)
                try:
                    processing_lock.extend(lock_ttl, replace_ttl=True)
                except Exception:
                    LOGGER.exception("Failed to extend processing lock job_id=%s", job_id)

        lock_heartbeat_task = asyncio.create_task(_refresh_processing_lock())

    manifest = job_store.require_job(job_id)
    position_order = position_order_for_variant(manifest.positioning_variant)
    allowed_positions = set(positions_for_variant(manifest.positioning_variant))
    owned_batch_size = (
        manifest.owned_batch_size_override
        if manifest.owned_batch_size_override is not None
        else settings.owned_batch_size
    )
    context_pages = (
        manifest.context_pages_override
        if manifest.context_pages_override is not None
        else settings.context_pages
    )
    LOGGER.info(
        (
            "Starting job job_id=%s page_count=%s target_language=%s testing_mode=%s "
            "provider=%s selected_model=%s selected_reasoning_effort=%s "
            "workflow=%s position_variant=%s "
            "selected_batch_size=%s selected_context_pages=%s "
            "effective_batch_size=%s effective_context_pages=%s "
            "render_dpi=%s image_max_dimension=%s max_parallel_batches=%s"
        ),
        job_id,
        manifest.page_count,
        manifest.target_language,
        manifest.testing_mode,
        manifest.llm_provider,
        manifest.llm_model or "default",
        manifest.llm_reasoning_effort or "provider_default",
        manifest.translation_workflow,
        manifest.positioning_variant,
        manifest.owned_batch_size_override if manifest.owned_batch_size_override is not None else "default",
        manifest.context_pages_override if manifest.context_pages_override is not None else "default",
        owned_batch_size,
        context_pages,
        settings.image_dpi,
        settings.image_max_dimension,
        settings.max_parallel_batches,
    )

    try:
        job_store.set_status(
            job_id=job_id,
            status=JobStatus.VALIDATING,
            message="Validating uploaded PDF...",
            progress=0.02,
        )
        input_pdf_path = job_store.input_pdf_path(job_id)
        if not input_pdf_path.exists():
            cached_input = job_store.get_cached_input_pdf(job_id)
            if cached_input is None:
                raise FileNotFoundError("Input PDF is unavailable for this job.")
            input_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            input_pdf_path.write_bytes(cached_input)
            LOGGER.info("Restored input PDF from redis job_id=%s", job_id)
        payload = input_pdf_path.read_bytes()
        page_count = validate_pdf_upload(
            filename=manifest.original_filename,
            payload=payload,
            max_upload_bytes=settings.max_upload_bytes,
            max_pages=settings.max_pages,
        )
        job_store.update_job(job_id, page_count=page_count)

        job_store.set_status(
            job_id=job_id,
            status=JobStatus.RENDERING,
            message="Rendering pages for recognition...",
            progress=0.08,
        )
        rendered_paths = render_pdf_pages_to_images(
            pdf_path=input_pdf_path,
            output_dir=job_store.rendered_dir(job_id),
            dpi=settings.image_dpi,
            max_dimension=settings.image_max_dimension,
        )
        if len(rendered_paths) != page_count:
            raise RuntimeError("Rendered page count does not match source PDF page count.")
        LOGGER.info(
            "Rendered pages job_id=%s page_count=%s render_dir=%s",
            job_id,
            len(rendered_paths),
            job_store.rendered_dir(job_id),
        )

        job_store.set_status(
            job_id=job_id,
            status=JobStatus.RENDERING,
            message="Rendered score pages.",
            progress=0.25,
        )
        batches = build_batches(
            total_pages=page_count,
            owned_batch_size=owned_batch_size,
            context_pages=context_pages,
        )
        max_parallel_batches = max(1, settings.max_parallel_batches)
        LOGGER.info(
            "Prepared batches job_id=%s total_batches=%s max_parallel_batches=%s",
            job_id,
            len(batches),
            max_parallel_batches,
        )

        is_canonical_mode = manifest.translation_workflow == TRANSLATION_WORKFLOW_CANONICAL
        canonical_result: CanonicalTranslationResult | None = None
        canonical_reference_block: str | None = None

        total_batches = len(batches)
        llm_entries_by_index: dict[int, BatchLLMLogEntry] = {}
        placement_groups_by_index: dict[int, list[TranslationPlacement]] = {}
        full_translations_by_index: dict[int, str] = {}
        full_source_texts_by_index: dict[int, str] = {}
        source_languages_by_index: dict[int, str] = {}
        analysis_base_progress = 0.40 if is_canonical_mode else 0.25
        analysis_span = 0.45 if is_canonical_mode else 0.60

        if is_canonical_mode:
            canonical_result = job_store.load_canonical_translation(job_id)
            if canonical_result is None:
                job_store.set_status(
                    job_id=job_id,
                    status=JobStatus.ANALYSING,
                    message="Building canonical full-score translation...",
                    progress=0.33,
                    current_batch=0,
                    total_batches=total_batches if total_batches else None,
                )
                canonical_batch = BatchSpec(
                    index=0,
                    owned_start=1,
                    owned_end=page_count,
                    supplied_start=1,
                    supplied_end=page_count,
                )
                canonical_result, canonical_log = await openai_service.translate_canonical(
                    batch=canonical_batch,
                    target_language=manifest.target_language,
                    image_paths=rendered_paths,
                    provider=manifest.llm_provider,
                    model_name=manifest.llm_model,
                    reasoning_effort=manifest.llm_reasoning_effort,
                    position_variant=manifest.positioning_variant,
                    user_header_override=(
                        f"target_language: {manifest.target_language}\n"
                        "MODE: canonical_translation_only\n"
                        f"SUPPLIED_PAGES: 1-{page_count}\n"
                        f"OWNED_PAGES: 1-{page_count}\n"
                        "Return canonical full translation and aligned lines."
                    ),
                )
                llm_entries_by_index[0] = canonical_log
                job_store.save_canonical_translation(job_id=job_id, canonical=canonical_result)
            else:
                LOGGER.info("Loaded cached canonical translation job_id=%s", job_id)

            canonical_reference_block = _canonical_reference_text(canonical_result)
            job_store.set_status(
                job_id=job_id,
                status=JobStatus.ANALYSING,
                message="Canonical translation ready. Placing text on pages...",
                progress=0.40,
                current_batch=0,
                total_batches=total_batches if total_batches else None,
            )

        checkpoints = job_store.load_batch_checkpoints(job_id)
        for checkpoint_index, (
            placements,
            full_translation,
            full_source_text,
            source_language,
            batch_log,
        ) in checkpoints.items():
            placement_groups_by_index[checkpoint_index] = placements
            full_translations_by_index[checkpoint_index] = full_translation
            full_source_texts_by_index[checkpoint_index] = full_source_text
            source_languages_by_index[checkpoint_index] = source_language
            llm_entries_by_index[checkpoint_index] = batch_log

        pending_batches = [batch for batch in batches if batch.index not in placement_groups_by_index]
        completed_batches = len(placement_groups_by_index)
        if completed_batches:
            LOGGER.info(
                "Resuming job from checkpoints job_id=%s completed_batches=%s total_batches=%s",
                job_id,
                completed_batches,
                total_batches,
            )

        async def _run_single_batch(
            batch_spec,
        ) -> tuple[int, int, int, list[TranslationPlacement], str, str, str, BatchLLMLogEntry]:
            LOGGER.info(
                "Running batch job_id=%s batch=%s owned=%s-%s supplied=%s-%s",
                job_id,
                batch_spec.index,
                batch_spec.owned_start,
                batch_spec.owned_end,
                batch_spec.supplied_start,
                batch_spec.supplied_end,
            )
            batch_result, batch_log = await openai_service.translate_batch(
                batch=batch_spec,
                target_language=manifest.target_language,
                image_paths=rendered_paths,
                provider=manifest.llm_provider,
                model_name=manifest.llm_model,
                reasoning_effort=manifest.llm_reasoning_effort,
                position_variant=manifest.positioning_variant,
                system_prompt_override=(
                    build_placement_from_canonical_prompt(manifest.positioning_variant)
                    if is_canonical_mode
                    else None
                ),
                extra_user_text_blocks=[canonical_reference_block] if canonical_reference_block else None,
            )
            filtered = clean_and_filter_batch_placements(
                placements=batch_result.placements,
                batch=batch_spec,
                allowed_positions=allowed_positions,
                logger=LOGGER,
            )
            LOGGER.info(
                "Batch complete job_id=%s batch=%s filtered_placements=%s",
                job_id,
                batch_spec.index,
                len(filtered),
            )
            return (
                batch_spec.index,
                batch_spec.owned_start,
                batch_spec.owned_end,
                _sort_batch_placements(filtered, position_order),
                (
                    canonical_result.full_translation
                    if is_canonical_mode and canonical_result is not None
                    else batch_result.full_translation
                ),
                (
                    canonical_result.full_source_text
                    if is_canonical_mode and canonical_result is not None
                    else batch_result.full_source_text
                ),
                (
                    canonical_result.source_language
                    if is_canonical_mode and canonical_result is not None
                    else batch_result.source_language
                ),
                batch_log,
            )

        if total_batches:
            job_store.set_status(
                job_id=job_id,
                status=JobStatus.ANALYSING,
                message=(
                    "Resuming translation from saved checkpoints..."
                    if completed_batches
                    else "Analysing score pages..."
                ),
                progress=analysis_base_progress + (completed_batches / total_batches * analysis_span),
                current_batch=completed_batches,
                total_batches=total_batches,
            )
            if pending_batches:
                semaphore = asyncio.Semaphore(max_parallel_batches)
                tasks: list[
                    asyncio.Task[
                        tuple[int, int, int, list[TranslationPlacement], str, str, str, BatchLLMLogEntry]
                    ]
                ] = []

                async def _run_with_limit(batch_spec):
                    async with semaphore:
                        return await _run_single_batch(batch_spec)

                tasks = [asyncio.create_task(_run_with_limit(batch_spec)) for batch_spec in pending_batches]
                try:
                    for completed_task in asyncio.as_completed(tasks):
                        (
                            batch_index,
                            owned_start,
                            owned_end,
                            sorted_placements,
                            full_translation,
                            full_source_text,
                            source_language,
                            batch_log,
                        ) = await completed_task
                        job_store.save_batch_checkpoint(
                            job_id=job_id,
                            batch_index=batch_index,
                            owned_start=owned_start,
                            owned_end=owned_end,
                            placements=sorted_placements,
                            full_translation=full_translation,
                            full_source_text=full_source_text,
                            source_language=source_language,
                            batch_log=batch_log,
                        )
                        placement_groups_by_index[batch_index] = sorted_placements
                        full_translations_by_index[batch_index] = full_translation
                        full_source_texts_by_index[batch_index] = full_source_text
                        source_languages_by_index[batch_index] = source_language
                        llm_entries_by_index[batch_index] = batch_log
                        completed_batches += 1
                        batch_progress = (
                            analysis_base_progress + completed_batches / total_batches * analysis_span
                        )
                        job_store.set_status(
                            job_id=job_id,
                            status=JobStatus.ANALYSING,
                            message=f"Analysed pages {owned_start}-{owned_end}.",
                            progress=batch_progress,
                            current_batch=completed_batches,
                            total_batches=total_batches,
                        )
                except Exception:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
        ordered_batch_indexes = sorted(placement_groups_by_index)
        placement_groups = [placement_groups_by_index[index] for index in ordered_batch_indexes]
        full_translations = [full_translations_by_index.get(index, "") for index in ordered_batch_indexes]
        full_source_texts = [full_source_texts_by_index.get(index, "") for index in ordered_batch_indexes]
        resolved_source_language = next(
            (value.strip() for value in source_languages_by_index.values() if value and value.strip()),
            "",
        )

        merged = merge_batch_results(
            target_language=manifest.target_language,
            position_order=position_order,
            placement_groups=placement_groups,
            full_translations=full_translations,
            full_source_texts=full_source_texts,
            source_language=resolved_source_language,
            full_translation_override=(
                canonical_result.full_translation
                if is_canonical_mode and canonical_result is not None
                else None
            ),
            full_source_text_override=(
                canonical_result.full_source_text
                if is_canonical_mode and canonical_result is not None
                else None
            ),
            source_language_override=(
                canonical_result.source_language
                if is_canonical_mode and canonical_result is not None
                else None
            ),
        )
        job_store.translation_json_path(job_id).write_text(
            merged.model_dump_json(indent=2),
            encoding="utf-8",
        )
        job_store.save_text_result(
            job_id=job_id,
            text_result=JobTextResult(
                source_language=merged.source_language,
                full_source_text=merged.full_source_text,
                target_language=merged.target_language,
                full_translation=merged.full_translation,
            ),
        )
        job_store.mark_text_result_available(job_id=job_id)
        LOGGER.info(
            "Merged translation job_id=%s total_placements=%s",
            job_id,
            len(merged.placements),
        )
        llm_entry_indexes = sorted(llm_entries_by_index)
        llm_entries = [llm_entries_by_index[index] for index in llm_entry_indexes]
        total_input_tokens = sum(entry.input_tokens or 0 for entry in llm_entries)
        total_output_tokens = sum(entry.output_tokens or 0 for entry in llm_entries)
        total_tokens = sum(entry.total_tokens or 0 for entry in llm_entries)
        total_cost = round(sum(entry.total_cost_usd for entry in llm_entries), 8)
        llm_log = JobLLMLog(
            job_id=job_id,
            provider=manifest.llm_provider,
            model=manifest.llm_model or (
                settings.openai_model
                if manifest.llm_provider == "openai"
                else settings.gemini_model
                if manifest.llm_provider == "gemini"
                else settings.deepseek_model
            ),
            reasoning_effort=manifest.llm_reasoning_effort,
            source_pdf_page_count=page_count,
            total_batches=len(llm_entries),
            total_pages_sent=sum(entry.pages_sent_count for entry in llm_entries),
            totals={
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost,
                "job_duration_seconds": 0.0,
            },
            entries=llm_entries,
        )
        job_store.llm_log_json_path(job_id).write_text(
            llm_log.model_dump_json(indent=2),
            encoding="utf-8",
        )
        job_store.mark_log_available(job_id=job_id)
        job_store.persist_llm_log_artifact(job_id=job_id)
        LOGGER.info(
            "Saved LLM request log job_id=%s path=%s total_cost_usd=%.6f",
            job_id,
            job_store.llm_log_json_path(job_id),
            total_cost,
        )

        job_store.set_status(
            job_id=job_id,
            status=JobStatus.CREATING_PDF,
            message="Creating translated PDF...",
            progress=0.90,
            current_batch=None,
            total_batches=total_batches if total_batches else None,
        )
        output_path = job_store.output_pdf_path(job_id)
        create_translated_pdf(
            original_pdf_path=input_pdf_path,
            output_pdf_path=output_path,
            translation_result=merged,
            output_font_size=settings.output_font_size,
            min_font_size=settings.min_font_size,
            output_background_opacity=settings.output_background_opacity,
            position_variant=manifest.positioning_variant,
        )
        download_filename = build_output_filename(
            manifest.safe_original_stem,
            manifest.target_language,
        )
        job_store.persist_output_pdf_artifact(job_id=job_id)

        # End-to-end runtime from process start until output is ready.
        llm_log.totals["job_duration_seconds"] = round(perf_counter() - started, 4)
        job_store.llm_log_json_path(job_id).write_text(
            llm_log.model_dump_json(indent=2),
            encoding="utf-8",
        )
        job_store.persist_llm_log_artifact(job_id=job_id)

        job_store.clear_batch_checkpoints(job_id)
        job_store.clear_canonical_translation(job_id)
        job_store.mark_complete(job_id=job_id, download_filename=download_filename)
    except PDFValidationError as exc:
        LOGGER.warning("Validation failed job_id=%s reason=%s", job_id, exc)
        job_store.mark_failed(job_id=job_id, message=str(exc))
    except PermanentOpenAIError as exc:
        LOGGER.warning("Provider non-retryable failure job_id=%s reason=%s", job_id, exc)
        job_store.mark_failed(job_id=job_id, message="Selected provider could not process this score.")
    except OpenAIServiceError as exc:
        LOGGER.warning("Provider failure job_id=%s reason=%s", job_id, exc)
        job_store.mark_failed(job_id=job_id, message="Temporary provider error. Please retry.")
    except Exception:
        LOGGER.exception("Unexpected processing error job_id=%s", job_id)
        job_store.mark_failed(
            job_id=job_id,
            message="Unexpected server error while processing the score.",
        )
    finally:
        if lock_heartbeat_task is not None:
            lock_heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await lock_heartbeat_task
        if processing_lock is not None:
            with contextlib.suppress(Exception):
                processing_lock.release()
        elapsed = perf_counter() - started
        LOGGER.info("Finished job job_id=%s duration_s=%.2f", job_id, elapsed)

