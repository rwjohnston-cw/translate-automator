from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any, Sequence

import httpx
from pydantic import BaseModel
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.pricing import estimate_cost_usd
from app.config import Settings
from app.models import (
    BatchLLMLogEntry,
    CanonicalTranslationResult,
    LLM_PROVIDER_DEEPSEEK,
    LLM_PROVIDER_GEMINI,
    LLM_PROVIDER_OPENAI,
    TranslationResult,
    build_dynamic_response_model,
    positions_for_variant,
)
from app.pdf_processing import BatchSpec, build_batch_header, build_image_data_url
from app.prompts import (
    build_canonical_translation_prompt,
    build_system_prompt,
)

LOGGER = logging.getLogger(__name__)

GEMINI_BUDGET_BY_EFFORT = {
    "low": 1024,
    "medium": 8192,
    "high": 16384,
}


class OpenAIServiceError(RuntimeError):
    """Base OpenAI processing error."""


class TransientOpenAIError(OpenAIServiceError):
    """Retryable OpenAI processing error."""


class PermanentOpenAIError(OpenAIServiceError):
    """Non-retryable OpenAI processing error."""


class OpenAIService:
    def __init__(
        self,
        settings: Settings,
        client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        if client is not None:
            self.openai_client = client
        else:
            api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
            self.openai_client = AsyncOpenAI(api_key=api_key) if api_key else None
        self.http_client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(120))

    def _require_client(self) -> AsyncOpenAI:
        if self.openai_client is None:
            raise PermanentOpenAIError(
                "OpenAI is not configured. Set OPENAI_API_KEY on the server."
            )
        return self.openai_client

    def build_user_content(
        self,
        *,
        batch: BatchSpec,
        target_language: str,
        image_paths: Sequence[Any],
        position_variant: str,
        header_override: str | None = None,
        extra_text_blocks: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        header_text = header_override or build_batch_header(batch, target_language, position_variant)
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": header_text,
            }
        ]
        for text in extra_text_blocks or ():
            cleaned = text.strip()
            if not cleaned:
                continue
            content.append({"type": "input_text", "text": cleaned})
        for page_number in batch.supplied_pages:
            image_path = image_paths[page_number - 1]
            content.append({"type": "input_text", "text": f"PDF_PAGE_NUMBER: {page_number}"})
            content.append(
                {
                    "type": "input_image",
                    "image_url": build_image_data_url(image_path),
                    "detail": "original",
                }
            )
        return content

    async def translate_batch(
        self,
        *,
        batch: BatchSpec,
        target_language: str,
        image_paths: Sequence[Any],
        provider: str,
        model_name: str | None,
        reasoning_effort: str | None = None,
        position_variant: str,
        system_prompt_override: str | None = None,
        user_header_override: str | None = None,
        extra_user_text_blocks: Sequence[str] | None = None,
    ) -> tuple[TranslationResult, BatchLLMLogEntry]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(TransientOpenAIError),
            reraise=True,
        ):
            with attempt:
                return await self._translate_batch_once(
                    batch=batch,
                    target_language=target_language,
                    image_paths=image_paths,
                    provider=provider,
                    model_name=model_name,
                    reasoning_effort=reasoning_effort,
                    position_variant=position_variant,
                    system_prompt_override=system_prompt_override,
                    user_header_override=user_header_override,
                    extra_user_text_blocks=extra_user_text_blocks,
                )
        raise RuntimeError("Retry loop terminated unexpectedly.")

    async def _translate_batch_once(
        self,
        *,
        batch: BatchSpec,
        target_language: str,
        image_paths: Sequence[Any],
        provider: str,
        model_name: str | None,
        reasoning_effort: str | None = None,
        position_variant: str,
        system_prompt_override: str | None = None,
        user_header_override: str | None = None,
        extra_user_text_blocks: Sequence[str] | None = None,
    ) -> tuple[TranslationResult, BatchLLMLogEntry]:
        allowed_positions = positions_for_variant(position_variant)
        schema_model = build_dynamic_response_model(allowed_positions)
        user_content = self.build_user_content(
            batch=batch,
            target_language=target_language,
            image_paths=image_paths,
            position_variant=position_variant,
            header_override=user_header_override,
            extra_text_blocks=extra_user_text_blocks,
        )
        system_prompt = system_prompt_override or build_system_prompt(position_variant)
        prompt_sent = self._build_prompt_log(system_prompt=system_prompt, user_content=user_content)
        model = self._resolve_model(provider=provider, model_name=model_name)
        effective_reasoning_effort = self._resolve_reasoning_effort(
            provider=provider,
            requested_reasoning_effort=reasoning_effort,
        )
        started = perf_counter()
        try:
            result, usage_info, information_received = await self._translate_structured(
                provider=provider,
                model=model,
                reasoning_effort=effective_reasoning_effort,
                system_prompt=system_prompt,
                user_content=user_content,
                schema_model=schema_model,
                payload_kind="translation",
                allowed_positions=allowed_positions,
            )
        except Exception as exc:
            if isinstance(exc, OpenAIServiceError):
                raise
            raise TransientOpenAIError("Unexpected provider communication error.") from exc
        finally:
            elapsed = perf_counter() - started
            LOGGER.info(
                "Provider batch call completed provider=%s model=%s batch=%s owned=%s-%s duration_s=%.2f",
                provider,
                model,
                batch.index,
                batch.owned_start,
                batch.owned_end,
                elapsed,
            )
        input_tokens = usage_info.get("input_tokens")
        output_tokens = usage_info.get("output_tokens")
        total_tokens = usage_info.get("total_tokens")
        costs = estimate_cost_usd(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        entry = BatchLLMLogEntry(
            batch_index=batch.index,
            provider=provider,
            model=model,
            reasoning_effort=effective_reasoning_effort,
            owned_pages=list(batch.owned_pages),
            supplied_pages=list(batch.supplied_pages),
            pages_sent_count=len(list(batch.supplied_pages)),
            duration_seconds=round(elapsed, 4),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost_usd=costs["input_cost_usd"],
            output_cost_usd=costs["output_cost_usd"],
            total_cost_usd=costs["total_cost_usd"],
            pricing_source=costs["pricing_source"],
            pricing_notes=costs["pricing_notes"],
            prompt_sent=prompt_sent,
            information_received=information_received,
        )
        return result, entry

    async def translate_canonical(
        self,
        *,
        batch: BatchSpec,
        target_language: str,
        image_paths: Sequence[Any],
        provider: str,
        model_name: str | None,
        reasoning_effort: str | None = None,
        position_variant: str,
        user_header_override: str | None = None,
    ) -> tuple[CanonicalTranslationResult, BatchLLMLogEntry]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(TransientOpenAIError),
            reraise=True,
        ):
            with attempt:
                return await self._translate_canonical_once(
                    batch=batch,
                    target_language=target_language,
                    image_paths=image_paths,
                    provider=provider,
                    model_name=model_name,
                    reasoning_effort=reasoning_effort,
                    position_variant=position_variant,
                    user_header_override=user_header_override,
                )
        raise RuntimeError("Retry loop terminated unexpectedly.")

    async def _translate_canonical_once(
        self,
        *,
        batch: BatchSpec,
        target_language: str,
        image_paths: Sequence[Any],
        provider: str,
        model_name: str | None,
        reasoning_effort: str | None = None,
        position_variant: str,
        user_header_override: str | None = None,
    ) -> tuple[CanonicalTranslationResult, BatchLLMLogEntry]:
        user_content = self.build_user_content(
            batch=batch,
            target_language=target_language,
            image_paths=image_paths,
            position_variant=position_variant,
            header_override=user_header_override,
        )
        system_prompt = build_canonical_translation_prompt(position_variant)
        prompt_sent = self._build_prompt_log(system_prompt=system_prompt, user_content=user_content)
        model = self._resolve_model(provider=provider, model_name=model_name)
        effective_reasoning_effort = self._resolve_reasoning_effort(
            provider=provider,
            requested_reasoning_effort=reasoning_effort,
        )
        started = perf_counter()
        try:
            result, usage_info, information_received = await self._translate_structured(
                provider=provider,
                model=model,
                reasoning_effort=effective_reasoning_effort,
                system_prompt=system_prompt,
                user_content=user_content,
                schema_model=CanonicalTranslationResult,
                payload_kind="canonical",
                allowed_positions=None,
            )
        except Exception as exc:
            if isinstance(exc, OpenAIServiceError):
                raise
            raise TransientOpenAIError("Unexpected provider communication error.") from exc
        finally:
            elapsed = perf_counter() - started
            LOGGER.info(
                "Provider canonical call completed provider=%s model=%s pages=%s-%s duration_s=%.2f",
                provider,
                model,
                batch.owned_start,
                batch.owned_end,
                elapsed,
            )
        input_tokens = usage_info.get("input_tokens")
        output_tokens = usage_info.get("output_tokens")
        costs = estimate_cost_usd(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        entry = BatchLLMLogEntry(
            batch_index=batch.index,
            provider=provider,
            model=model,
            reasoning_effort=effective_reasoning_effort,
            owned_pages=list(batch.owned_pages),
            supplied_pages=list(batch.supplied_pages),
            pages_sent_count=len(list(batch.supplied_pages)),
            duration_seconds=round(elapsed, 4),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=usage_info.get("total_tokens"),
            input_cost_usd=costs["input_cost_usd"],
            output_cost_usd=costs["output_cost_usd"],
            total_cost_usd=costs["total_cost_usd"],
            pricing_source=costs["pricing_source"],
            pricing_notes=costs["pricing_notes"],
            prompt_sent=prompt_sent,
            information_received=information_received,
        )
        return result, entry

    async def _translate_structured(
        self,
        *,
        provider: str,
        model: str,
        reasoning_effort: str | None,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        schema_model: type[Any],
        payload_kind: str,
        allowed_positions: Sequence[str] | None,
    ) -> tuple[Any, dict[str, int | None], dict[str, Any]]:
        if provider == LLM_PROVIDER_OPENAI:
            return await self._translate_structured_openai(
                model=model,
                reasoning_effort=reasoning_effort,
                system_prompt=system_prompt,
                user_content=user_content,
                schema_model=schema_model,
                payload_kind=payload_kind,
            )
        if provider == LLM_PROVIDER_GEMINI:
            gemini_schema = self._resolve_gemini_schema(
                payload_kind=payload_kind,
                allowed_positions=allowed_positions,
            )
            return await self._translate_structured_gemini(
                model=model,
                reasoning_effort=reasoning_effort,
                system_prompt=system_prompt,
                user_content=user_content,
                schema_model=schema_model,
                gemini_schema=gemini_schema,
                payload_kind=payload_kind,
            )
        if provider == LLM_PROVIDER_DEEPSEEK:
            return await self._translate_structured_deepseek(
                model=model,
                reasoning_effort=reasoning_effort,
                system_prompt=system_prompt,
                user_content=user_content,
                schema_model=schema_model,
                payload_kind=payload_kind,
            )
        raise PermanentOpenAIError("Unsupported provider.")

    async def _translate_structured_openai(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        schema_model: type[Any],
        payload_kind: str,
    ) -> tuple[Any, dict[str, int | None], dict[str, Any]]:
        client = self._require_client()
        request_payload: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "text_format": schema_model,
        }
        if reasoning_effort:
            request_payload["reasoning"] = {"effort": reasoning_effort}
        try:
            response = await client.responses.parse(**request_payload)
        except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
            raise TransientOpenAIError("OpenAI request timed out or was rate-limited.") from exc
        except APIStatusError as exc:
            if exc.status_code >= 500 or exc.status_code == 429:
                raise TransientOpenAIError("OpenAI temporary server error.") from exc
            raise PermanentOpenAIError("OpenAI request failed.") from exc

        output_entries = getattr(response, "output", []) or []
        for entry in output_entries:
            if getattr(entry, "type", None) == "refusal":
                raise PermanentOpenAIError("The model refused this request.")
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise TransientOpenAIError("OpenAI returned an empty response.")
        structured_payload = self._validate_structured_payload(
            parsed,
            schema_model=schema_model,
            payload_kind=payload_kind,
        )
        usage = self._extract_openai_usage(response)
        info_received = {
            "response_id": getattr(response, "id", None),
            "output_types": [getattr(entry, "type", None) for entry in output_entries],
            "structured_output": structured_payload.model_dump(),
        }
        return structured_payload, usage, info_received

    async def _translate_structured_gemini(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        schema_model: type[Any],
        gemini_schema: dict[str, Any],
        payload_kind: str,
    ) -> tuple[Any, dict[str, int | None], dict[str, Any]]:
        api_key = self._normalize_api_key(
            self.settings.gemini_api_key.get_secret_value() if self.settings.gemini_api_key else None
        )
        if not api_key:
            raise PermanentOpenAIError("Gemini is not configured. Set GEMINI_API_KEY on the server.")

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        parts = self._to_gemini_parts(user_content)
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": gemini_schema,
            },
        }
        thinking_config = self._gemini_thinking_config(
            model=model,
            reasoning_effort=reasoning_effort,
        )
        if thinking_config is not None:
            payload["generationConfig"]["thinkingConfig"] = thinking_config
        try:
            response = await self.http_client.post(
                endpoint,
                headers={"x-goog-api-key": api_key},
                json=payload,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientOpenAIError("Gemini request timed out or network failed.") from exc

        if response.status_code >= 500 or response.status_code == 429:
            detail = self._extract_http_error_message(response)
            raise TransientOpenAIError(
                f"Gemini temporary server error (status {response.status_code}): {detail}"
            )
        if response.status_code >= 400:
            detail = self._extract_http_error_message(response)
            raise PermanentOpenAIError(
                f"Gemini request failed (status {response.status_code}): {detail}"
            )

        payload_json = response.json()
        candidates = payload_json.get("candidates") or []
        if not candidates:
            raise TransientOpenAIError("Gemini returned no candidates.")
        text = ""
        for part in candidates[0].get("content", {}).get("parts", []):
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text = part["text"]
                break
        if not text.strip():
            raise TransientOpenAIError("Gemini returned an empty response.")
        try:
            parsed_json = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TransientOpenAIError("Gemini returned invalid JSON.") from exc
        structured_payload = self._validate_structured_payload(
            parsed_json,
            schema_model=schema_model,
            payload_kind=payload_kind,
        )
        usage = self._extract_gemini_usage(payload_json)
        info_received = {
            "candidate_count": len(candidates),
            "raw_text": text,
            "structured_output": structured_payload.model_dump(),
        }
        return structured_payload, usage, info_received

    async def _translate_structured_deepseek(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        schema_model: type[Any],
        payload_kind: str,
    ) -> tuple[Any, dict[str, int | None], dict[str, Any]]:
        api_key = self._normalize_api_key(
            self.settings.deepseek_api_key.get_secret_value() if self.settings.deepseek_api_key else None
        )
        if not api_key:
            raise PermanentOpenAIError("DeepSeek is not configured. Set DEEPSEEK_API_KEY on the server.")

        endpoint = f"{self.settings.deepseek_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._to_deepseek_content(user_content)},
            ],
            "response_format": {"type": "json_object"},
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        try:
            response = await self.http_client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientOpenAIError("DeepSeek request timed out or network failed.") from exc

        if response.status_code >= 500 or response.status_code == 429:
            detail = self._extract_http_error_message(response)
            raise TransientOpenAIError(
                f"DeepSeek temporary server error (status {response.status_code}): {detail}"
            )
        if response.status_code >= 400:
            detail = self._extract_http_error_message(response)
            raise PermanentOpenAIError(
                f"DeepSeek request failed (status {response.status_code}): {detail}"
            )

        payload_json = response.json()
        choices = payload_json.get("choices") or []
        if not choices:
            raise TransientOpenAIError("DeepSeek returned no choices.")
        message = choices[0].get("message", {})
        if message.get("refusal"):
            raise PermanentOpenAIError("The model refused this request.")
        content = message.get("content")
        if isinstance(content, list):
            text_chunks = [item.get("text", "") for item in content if isinstance(item, dict)]
            content = "\n".join(chunk for chunk in text_chunks if chunk)
        if not isinstance(content, str) or not content.strip():
            raise TransientOpenAIError("DeepSeek returned an empty response.")
        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError as exc:
            raise TransientOpenAIError("DeepSeek returned invalid JSON.") from exc
        structured_payload = self._validate_structured_payload(
            parsed_json,
            schema_model=schema_model,
            payload_kind=payload_kind,
        )
        usage = self._extract_deepseek_usage(payload_json)
        info_received = {
            "choice_count": len(choices),
            "raw_text": content,
            "structured_output": structured_payload.model_dump(),
        }
        return structured_payload, usage, info_received

    def _validate_structured_payload(
        self,
        parsed: Any,
        schema_model: type[Any],
        payload_kind: str,
    ) -> Any:
        try:
            if isinstance(parsed, BaseModel):
                parsed = parsed.model_dump()
            if (
                payload_kind == "translation"
                and isinstance(parsed, dict)
                and "full_translation" not in parsed
            ):
                placements = parsed.get("placements")
                fallback_lines: list[str] = []
                if isinstance(placements, list):
                    for item in placements:
                        if isinstance(item, dict):
                            text = str(item.get("translated_text", "")).strip()
                            if text:
                                fallback_lines.append(text)
                parsed["full_translation"] = "\n".join(fallback_lines)
            return schema_model.model_validate(parsed)
        except Exception as exc:
            raise TransientOpenAIError("Provider returned invalid structured output.") from exc

    @staticmethod
    def _extract_data_url_base64(data_url: str) -> str:
        if "," not in data_url:
            return data_url
        return data_url.split(",", 1)[1]

    @staticmethod
    def _normalize_api_key(raw_key: str | None) -> str | None:
        if raw_key is None:
            return None
        normalized = (
            raw_key.strip()
            .replace("\ufeff", "")
            .replace("\u200b", "")
            .replace("\u200c", "")
            .replace("\u200d", "")
        )
        return normalized or None

    @staticmethod
    def _extract_http_error_message(response: Any) -> str:
        message: str | None = None
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                candidate = error.get("message")
                if isinstance(candidate, str) and candidate.strip():
                    message = candidate.strip()
            if message is None:
                candidate = payload.get("message")
                if isinstance(candidate, str) and candidate.strip():
                    message = candidate.strip()
        if message is None:
            raw_text = getattr(response, "text", "")
            if isinstance(raw_text, str) and raw_text.strip():
                message = raw_text.strip()
        if not message:
            return "No provider error details returned."
        return message[:500]

    @staticmethod
    def _resolve_gemini_schema(
        *,
        payload_kind: str,
        allowed_positions: Sequence[str] | None,
    ) -> dict[str, Any]:
        if payload_kind == "translation":
            if allowed_positions is None:
                raise PermanentOpenAIError("Missing allowed positions for translation schema.")
            return OpenAIService._build_gemini_response_schema(allowed_positions)
        if payload_kind == "canonical":
            return OpenAIService._build_gemini_canonical_response_schema()
        raise PermanentOpenAIError("Unsupported Gemini payload schema.")

    @staticmethod
    def _build_gemini_response_schema(allowed_positions: Sequence[str]) -> dict[str, Any]:
        # Gemini's responseSchema parser rejects JSON Schema refs/defs.
        # Build an inline schema that uses only supported primitive fields.
        return {
            "type": "OBJECT",
            "properties": {
                "target_language": {"type": "STRING"},
                "full_translation": {"type": "STRING"},
                "placements": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "page": {"type": "INTEGER"},
                            "position": {"type": "STRING", "enum": list(allowed_positions)},
                            "translated_text": {"type": "STRING"},
                        },
                        "required": ["page", "position", "translated_text"],
                    },
                },
            },
            "required": ["target_language", "full_translation", "placements"],
        }

    @staticmethod
    def _build_gemini_canonical_response_schema() -> dict[str, Any]:
        return {
            "type": "OBJECT",
            "properties": {
                "target_language": {"type": "STRING"},
                "full_translation": {"type": "STRING"},
                "aligned_lines": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "source_text": {"type": "STRING"},
                            "translated_text": {"type": "STRING"},
                        },
                        "required": ["source_text", "translated_text"],
                    },
                },
            },
            "required": ["target_language", "full_translation", "aligned_lines"],
        }

    def _to_gemini_parts(self, user_content: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for item in user_content:
            if item.get("type") == "input_text":
                parts.append({"text": item["text"]})
            elif item.get("type") == "input_image":
                encoded = self._extract_data_url_base64(item["image_url"])
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": encoded,
                        }
                    }
                )
        return parts

    def _to_deepseek_content(self, user_content: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for item in user_content:
            if item.get("type") == "input_text":
                content.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "input_image":
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": item["image_url"]},
                    }
                )
        return content

    def _resolve_model(self, *, provider: str, model_name: str | None) -> str:
        if model_name and model_name.strip():
            return model_name.strip()
        if provider == LLM_PROVIDER_OPENAI:
            return self.settings.openai_model
        if provider == LLM_PROVIDER_GEMINI:
            return self.settings.gemini_model
        if provider == LLM_PROVIDER_DEEPSEEK:
            return self.settings.deepseek_model
        return self.settings.openai_model

    def _gemini_thinking_config(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
    ) -> dict[str, Any] | None:
        if not reasoning_effort:
            return None
        normalized = reasoning_effort.strip().lower()
        if not normalized:
            return None
        # Gemini 2.5 models use token budget control instead of semantic levels.
        if model.startswith("gemini-2.5-"):
            budget = GEMINI_BUDGET_BY_EFFORT.get(normalized)
            if budget is None:
                raise PermanentOpenAIError(
                    f"Unsupported Gemini 2.5 reasoning effort '{normalized}'."
                )
            return {"thinkingBudget": budget}
        return {"thinkingLevel": normalized}

    def _resolve_reasoning_effort(
        self,
        *,
        provider: str,
        requested_reasoning_effort: str | None,
    ) -> str | None:
        if requested_reasoning_effort is not None:
            cleaned = requested_reasoning_effort.strip()
            return cleaned or None
        if provider == LLM_PROVIDER_OPENAI:
            return self.settings.openai_reasoning_effort
        return None

    @staticmethod
    def _build_prompt_log(*, system_prompt: str, user_content: Sequence[dict[str, Any]]) -> dict[str, Any]:
        sanitized: list[dict[str, Any]] = []
        for item in user_content:
            if item.get("type") == "input_image":
                sanitized.append(
                    {
                        "type": "input_image",
                        "detail": item.get("detail", "original"),
                        "image_url": "<omitted_base64_data_url>",
                    }
                )
            else:
                sanitized.append(item)
        pages = [
            entry.get("text")
            for entry in sanitized
            if entry.get("type") == "input_text" and str(entry.get("text", "")).startswith("PDF_PAGE_NUMBER:")
        ]
        return {
            "system_prompt": system_prompt,
            "user_content": sanitized,
            "page_markers": pages,
            "image_count": sum(1 for entry in user_content if entry.get("type") == "input_image"),
        }

    @staticmethod
    def _as_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _extract_openai_usage(self, response: Any) -> dict[str, int | None]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
        input_tokens = self._as_int(getattr(usage, "input_tokens", None))
        if input_tokens is None:
            input_tokens = self._as_int(getattr(usage, "prompt_tokens", None))
        if input_tokens is None and isinstance(usage, dict):
            input_tokens = self._as_int(usage.get("input_tokens") or usage.get("prompt_tokens"))
        output_tokens = self._as_int(getattr(usage, "output_tokens", None))
        if output_tokens is None:
            output_tokens = self._as_int(getattr(usage, "completion_tokens", None))
        if output_tokens is None and isinstance(usage, dict):
            output_tokens = self._as_int(usage.get("output_tokens") or usage.get("completion_tokens"))
        total_tokens = self._as_int(getattr(usage, "total_tokens", None))
        if total_tokens is None and isinstance(usage, dict):
            total_tokens = self._as_int(usage.get("total_tokens"))
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def _extract_gemini_usage(self, payload_json: dict[str, Any]) -> dict[str, int | None]:
        usage = payload_json.get("usageMetadata") or {}
        input_tokens = self._as_int(usage.get("promptTokenCount"))
        output_tokens = self._as_int(usage.get("candidatesTokenCount"))
        total_tokens = self._as_int(usage.get("totalTokenCount"))
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def _extract_deepseek_usage(self, payload_json: dict[str, Any]) -> dict[str, int | None]:
        usage = payload_json.get("usage") or {}
        input_tokens = self._as_int(usage.get("prompt_tokens"))
        output_tokens = self._as_int(usage.get("completion_tokens"))
        total_tokens = self._as_int(usage.get("total_tokens"))
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

