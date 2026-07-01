from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


ReasoningEffort = Literal["low", "medium", "high"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.5"
    openai_reasoning_effort: ReasoningEffort = "medium"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash-lite"
    deepseek_api_key: SecretStr | None = None
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_base_url: str = "https://api.deepseek.com"

    max_upload_mb: int = 50
    max_pages: int = 100
    owned_batch_size: int = 6
    context_pages: int = 1
    max_parallel_batches: int = 2
    image_dpi: int = 240
    image_max_dimension: int = 3500
    job_ttl_minutes: int = 60
    output_font_size: float = 14.0
    output_background_opacity: float = 0.88

    job_root: Path = Path("/tmp/score-translator/jobs")
    cleanup_interval_seconds: int = 300

    @computed_field
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @computed_field
    @property
    def min_font_size(self) -> float:
        return 7.0

