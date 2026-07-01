from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, create_model, field_validator

POSITION_VARIANT_STANDARD = "standard_3"
POSITION_VARIANT_CENTERED_5 = "centered_5"
POSITION_VARIANT_SPLIT_6 = "split_6"
POSITION_VARIANT_GRID_20 = "grid_20"

PositionVariant = Literal["standard_3", "centered_5", "split_6", "grid_20"]

LLM_PROVIDER_OPENAI = "openai"
LLM_PROVIDER_GEMINI = "gemini"
LLM_PROVIDER_DEEPSEEK = "deepseek"
LLMProvider = Literal["openai", "gemini", "deepseek"]

TRANSLATION_WORKFLOW_BATCH = "batch_translate_and_place"
TRANSLATION_WORKFLOW_CANONICAL = "canonical_then_place"
TranslationWorkflow = Literal["batch_translate_and_place", "canonical_then_place"]

POSITION_VARIANT_MAP: dict[str, tuple[str, ...]] = {
    POSITION_VARIANT_STANDARD: ("top", "middle", "bottom"),
    POSITION_VARIANT_CENTERED_5: (
        "top",
        "upper_middle",
        "middle",
        "lower_middle",
        "bottom",
    ),
    POSITION_VARIANT_SPLIT_6: (
        "top_left",
        "top_right",
        "middle_left",
        "middle_right",
        "bottom_left",
        "bottom_right",
    ),
    POSITION_VARIANT_GRID_20: (
        "a1",
        "a2",
        "a3",
        "a4",
        "b1",
        "b2",
        "b3",
        "b4",
        "c1",
        "c2",
        "c3",
        "c4",
        "d1",
        "d2",
        "d3",
        "d4",
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
    ),
}

ALL_POSITION_LABELS = tuple(
    label for labels in POSITION_VARIANT_MAP.values() for label in labels
)
PositionLabel = Literal[
    "top",
    "middle",
    "bottom",
    "upper_middle",
    "lower_middle",
    "top_left",
    "top_right",
    "middle_left",
    "middle_right",
    "bottom_left",
    "bottom_right",
    "a1",
    "a2",
    "a3",
    "a4",
    "b1",
    "b2",
    "b3",
    "b4",
    "c1",
    "c2",
    "c3",
    "c4",
    "d1",
    "d2",
    "d3",
    "d4",
    "e1",
    "e2",
    "e3",
    "e4",
    "e5",
]


def positions_for_variant(position_variant: str) -> tuple[str, ...]:
    return POSITION_VARIANT_MAP.get(position_variant, POSITION_VARIANT_MAP[POSITION_VARIANT_STANDARD])


def build_dynamic_response_model(allowed_positions: Sequence[str]) -> type[BaseModel]:
    position_literal = Literal.__getitem__(tuple(allowed_positions))
    placement_model = create_model(
        "DynamicTranslationPlacement",
        page=(int, Field(description="One-based global PDF page number.")),
        position=(position_literal, Field(description="Placement position label.")),
        translated_text=(str, Field(description="Translated sung text fragment.")),
    )
    return create_model(
        "DynamicTranslationResult",
        target_language=(str, ...),
        full_translation=(
            str,
            Field(
                description=(
                    "Complete translated poem/text for the owned pages, preserving intended line "
                    "breaks and stanza breaks."
                )
            ),
        ),
        placements=(list[placement_model], ...),
    )


class TranslationPlacement(BaseModel):
    page: int = Field(description="The one-based page number in the original uploaded PDF.")
    position: PositionLabel
    translated_text: str = Field(
        description="The translated sung text to print at this location."
    )

    @field_validator("translated_text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()


class TranslationResult(BaseModel):
    target_language: str
    full_translation: str = ""
    placements: list[TranslationPlacement]

    @field_validator("full_translation")
    @classmethod
    def _normalize_full_translation(cls, value: str) -> str:
        return value.replace("\r\n", "\n").replace("\r", "\n").strip()


class CanonicalTranslationLine(BaseModel):
    source_text: str = Field(
        description="One original-language sung line or phrase from the reconstructed source text."
    )
    translated_text: str = Field(
        description="The corresponding target-language translation of that line or phrase."
    )

    @field_validator("source_text", "translated_text")
    @classmethod
    def _strip_line(cls, value: str) -> str:
        return value.strip()


class CanonicalTranslationResult(BaseModel):
    target_language: str
    full_translation: str = Field(
        description="Complete target-language translation with intended poetic line breaks."
    )
    aligned_lines: list[CanonicalTranslationLine] = Field(
        description=(
            "Line-by-line alignment between reconstructed source text and translation. "
            "Use reading order and keep alignment semantic, not syllabic."
        )
    )

    @field_validator("full_translation")
    @classmethod
    def _normalize_full_translation(cls, value: str) -> str:
        return value.replace("\r\n", "\n").replace("\r", "\n").strip()


class BatchLLMLogEntry(BaseModel):
    batch_index: int
    provider: str
    model: str
    reasoning_effort: str | None = None
    owned_pages: list[int]
    supplied_pages: list[int]
    pages_sent_count: int
    duration_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    pricing_source: str | None = None
    pricing_notes: str | None = None
    prompt_sent: dict[str, Any]
    information_received: dict[str, Any]


class JobLLMLog(BaseModel):
    job_id: str
    provider: str
    model: str
    reasoning_effort: str | None = None
    source_pdf_page_count: int
    total_batches: int
    total_pages_sent: int
    totals: dict[str, float | int]
    entries: list[BatchLLMLogEntry]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class JobStatus(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    RENDERING = "rendering"
    ANALYSING = "analysing"
    CREATING_PDF = "creating_pdf"
    COMPLETE = "complete"
    FAILED = "failed"


class JobManifest(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    message: str = "Queued"
    progress: float = 0.0
    current_batch: int | None = None
    total_batches: int | None = None
    error: str | None = None
    download_filename: str | None = None
    output_available: bool = False
    log_available: bool = False
    original_filename: str
    safe_original_stem: str
    target_language: str
    llm_provider: LLMProvider = LLM_PROVIDER_OPENAI
    llm_model: str | None = None
    translation_workflow: TranslationWorkflow = TRANSLATION_WORKFLOW_BATCH
    positioning_variant: PositionVariant = POSITION_VARIANT_STANDARD
    testing_mode: bool = False
    owned_batch_size_override: int | None = None
    context_pages_override: int | None = None
    page_count: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    expires_at: datetime

    def touch(self) -> None:
        self.updated_at = datetime.now(tz=timezone.utc)

