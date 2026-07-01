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
    LLM_PROVIDER_DEEPSEEK,
    LLM_PROVIDER_GEMINI,
    LLM_PROVIDER_OPENAI,
    TranslationResult,
    build_dynamic_response_model,
    positions_for_variant,
)
from app.pdf_processing import BatchSpec, build_batch_header, build_image_data_url
from app.prompts import build_system_prompt

LOGGER = logging.getLogger(__name__)


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
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": build_batch_header(batch, target_language, position_variant),
            }
        ]
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
        position_variant: str,
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
                    position_variant=position_variant,
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
        position_variant: str,
    ) -> tuple[TranslationResult, BatchLLMLogEntry]:
        allowed_positions = positions_for_variant(position_variant)
        schema_model = build_dynamic_response_model(allowed_positions)
        user_content = self.build_user_content(
            batch=batch,
            target_language=target_language,
            image_paths=image_paths,
            position_variant=position_variant,
        )
        system_prompt = build_system_prompt(position_variant)
        prompt_sent = self._build_prompt_log(system_prompt=system_prompt, user_content=user_content)
        model = self._resolve_model(provider=provider, model_name=model_name)
        started = perf_counter()
        try:
            if provider == LLM_PROVIDER_OPENAI:
                result, usage_info, information_received = await self._translate_batch_openai(
                    model=model,
                    system_prompt=system_prompt,
                    user_content=user_content,
                    schema_model=schema_model,
                )
            elif provider == LLM_PROVIDER_GEMINI:
                result, usage_info, information_received = await self._translate_batch_gemini(
                    model=model,
                    system_prompt=system_prompt,
                    user_content=user_content,
                    schema_model=schema_model,
                    allowed_positions=allowed_positions,
                )
            elif provider == LLM_PROVIDER_DEEPSEEK:
                result, usage_info, information_received = await self._translate_batch_deepseek(
                    model=model,
                    system_prompt=system_prompt,
                    user_content=user_content,
                    schema_model=schema_model,
                )
            else:
                raise PermanentOpenAIError("Unsupported provider.")
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
            reasoning_effort=(
                self.settings.openai_reasoning_effort if provider == LLM_PROVIDER_OPENAI else None
            ),
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

    async def _translate_batch_openai(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        schema_model: type[Any],
    ) -> tuple[TranslationResult, dict[str, int | None], dict[str, Any]]:
        client = self._require_client()
        try:
            response = await client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                text_format=schema_model,
                reasoning={"effort": self.settings.openai_reasoning_effort},
            )
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
        translation = self._validate_translation_payload(parsed, schema_model)
        usage = self._extract_openai_usage(response)
        info_received = {
            "response_id": getattr(response, "id", None),
            "output_types": [getattr(entry, "type", None) for entry in output_entries],
            "structured_output": translation.model_dump(),
        }
        return translation, usage, info_received

    async def _translate_batch_gemini(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        schema_model: type[Any],
        allowed_positions: Sequence[str],
    ) -> tuple[TranslationResult, dict[str, int | None], dict[str, Any]]:
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
                "responseSchema": self._build_gemini_response_schema(allowed_positions),
            },
        }
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
        translation = self._validate_translation_payload(parsed_json, schema_model)
        usage = self._extract_gemini_usage(payload_json)
        info_received = {
            "candidate_count": len(candidates),
            "raw_text": text,
            "structured_output": translation.model_dump(),
        }
        return translation, usage, info_received

    async def _translate_batch_deepseek(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        schema_model: type[Any],
    ) -> tuple[TranslationResult, dict[str, int | None], dict[str, Any]]:
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
        translation = self._validate_translation_payload(parsed_json, schema_model)
        usage = self._extract_deepseek_usage(payload_json)
        info_received = {
            "choice_count": len(choices),
            "raw_text": content,
            "structured_output": translation.model_dump(),
        }
        return translation, usage, info_received

    def _validate_translation_payload(
        self,
        parsed: Any,
        schema_model: type[Any],
    ) -> TranslationResult:
        try:
            if isinstance(parsed, BaseModel):
                parsed = parsed.model_dump()
            structured = schema_model.model_validate(parsed)
            return TranslationResult.model_validate(structured.model_dump())
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
    def _build_gemini_response_schema(allowed_positions: Sequence[str]) -> dict[str, Any]:
        # Gemini's responseSchema parser rejects JSON Schema refs/defs.
        # Build an inline schema that uses only supported primitive fields.
        return {
            "type": "OBJECT",
            "properties": {
                "target_language": {"type": "STRING"},
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
            "required": ["target_language", "placements"],
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

