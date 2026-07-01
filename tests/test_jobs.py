from __future__ import annotations

import asyncio
import json

import app.jobs as jobs_module
from app.config import Settings
from app.jobs import JobStore, process_job
from app.models import LLM_PROVIDER_OPENAI, JobStatus, TranslationPlacement, TranslationResult
from app.pdf_processing import BatchSpec


class _FakeOpenAIService:
    def __init__(self) -> None:
        self.active_calls = 0
        self.max_active_calls = 0

    async def translate_batch(
        self,
        *,
        batch: BatchSpec,
        target_language: str,
        image_paths,
        provider: str,
        model_name: str | None,
        position_variant: str,
    ):
        del image_paths, position_variant
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        await asyncio.sleep(0.05)
        self.active_calls -= 1

        result = TranslationResult(
            target_language=target_language,
            placements=[
                TranslationPlacement(
                    page=batch.owned_start,
                    position="top",
                    translated_text=f"batch-{batch.index}",
                )
            ],
        )
        log_entry = jobs_module.BatchLLMLogEntry(
            batch_index=batch.index,
            provider=provider,
            model=model_name or "default",
            reasoning_effort=None,
            owned_pages=list(batch.owned_pages),
            supplied_pages=list(batch.supplied_pages),
            pages_sent_count=len(list(batch.supplied_pages)),
            duration_seconds=0.05,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_cost_usd=0.0,
            output_cost_usd=0.0,
            total_cost_usd=0.0,
            pricing_source=None,
            pricing_notes=None,
            prompt_sent={"batch": batch.index},
            information_received={"provider": provider},
        )
        return result, log_entry


def test_process_job_runs_batches_concurrently_and_keeps_log_order(tmp_path, monkeypatch):
    settings = Settings(
        openai_api_key="sk-test",
        job_root=tmp_path / "jobs",
        owned_batch_size=1,
        context_pages=0,
        max_parallel_batches=3,
    )
    job_store = JobStore(settings)
    manifest = job_store.create_job(
        original_filename="score.pdf",
        target_language="English",
        page_count=3,
        llm_provider=LLM_PROVIDER_OPENAI,
        llm_model=None,
        positioning_variant="standard_3",
        testing_mode=True,
    )
    job_store.input_pdf_path(manifest.job_id).write_bytes(b"%PDF-1.4\nfake")

    def fake_validate_pdf_upload(**kwargs):
        del kwargs
        return 3

    def fake_render_pdf_pages_to_images(**kwargs):
        del kwargs
        return [tmp_path / "p1.png", tmp_path / "p2.png", tmp_path / "p3.png"]

    def fake_build_batches(*, total_pages: int, owned_batch_size: int, context_pages: int):
        del total_pages, owned_batch_size, context_pages
        return [
            BatchSpec(index=1, owned_start=1, owned_end=1, supplied_start=1, supplied_end=1),
            BatchSpec(index=2, owned_start=2, owned_end=2, supplied_start=2, supplied_end=2),
            BatchSpec(index=3, owned_start=3, owned_end=3, supplied_start=3, supplied_end=3),
        ]

    def fake_clean_and_filter_batch_placements(*, placements, batch, allowed_positions, logger):
        del batch, allowed_positions, logger
        return placements

    def fake_merge_batch_results(*, target_language: str, position_order: dict[str, int], placement_groups):
        del position_order
        merged = [placement for group in placement_groups for placement in group]
        return TranslationResult(target_language=target_language, placements=merged)

    def fake_create_translated_pdf(**kwargs):
        output_pdf_path = kwargs["output_pdf_path"]
        output_pdf_path.write_bytes(b"%PDF-1.4\ntranslated")

    monkeypatch.setattr(jobs_module, "validate_pdf_upload", fake_validate_pdf_upload)
    monkeypatch.setattr(jobs_module, "render_pdf_pages_to_images", fake_render_pdf_pages_to_images)
    monkeypatch.setattr(jobs_module, "build_batches", fake_build_batches)
    monkeypatch.setattr(jobs_module, "clean_and_filter_batch_placements", fake_clean_and_filter_batch_placements)
    monkeypatch.setattr(jobs_module, "merge_batch_results", fake_merge_batch_results)
    monkeypatch.setattr(jobs_module, "create_translated_pdf", fake_create_translated_pdf)

    fake_openai_service = _FakeOpenAIService()
    asyncio.run(
        process_job(
            job_store=job_store,
            settings=settings,
            openai_service=fake_openai_service,
            job_id=manifest.job_id,
        )
    )

    completed_manifest = job_store.require_job(manifest.job_id)
    assert completed_manifest.status == JobStatus.COMPLETE
    assert fake_openai_service.max_active_calls > 1

    llm_log_payload = json.loads(job_store.llm_log_json_path(manifest.job_id).read_text(encoding="utf-8"))
    assert [entry["batch_index"] for entry in llm_log_payload["entries"]] == [1, 2, 3]
