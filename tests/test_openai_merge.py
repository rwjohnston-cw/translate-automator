from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from app.config import Settings
from app.models import POSITION_VARIANT_STANDARD, TranslationPlacement, TranslationResult
from app.openai_service import OpenAIService, PermanentOpenAIError
from app.pdf_processing import (
    BatchSpec,
    clean_and_filter_batch_placements,
    merge_batch_results,
    position_order_for_variant,
)


def test_result_filtering_and_merge_rules():
    batch = BatchSpec(index=1, owned_start=2, owned_end=3, supplied_start=1, supplied_end=4)
    filtered = clean_and_filter_batch_placements(
        placements=[
            TranslationPlacement(page=1, position="top", translated_text="outside"),
            TranslationPlacement(page=2, position="middle", translated_text="  Kyrie "),
            TranslationPlacement(page=3, position="top", translated_text=""),
            TranslationPlacement(page=3, position="top", translated_text="Eleison"),
        ],
        batch=batch,
        allowed_positions={"top", "middle", "bottom"},
    )
    assert [(p.page, p.position, p.translated_text) for p in filtered] == [
        (2, "middle", "Kyrie"),
        (3, "top", "Eleison"),
    ]

    merged = merge_batch_results(
        target_language="English",
        position_order=position_order_for_variant(POSITION_VARIANT_STANDARD),
        placement_groups=[
            [
                TranslationPlacement(page=2, position="bottom", translated_text="Have mercy"),
                TranslationPlacement(page=2, position="top", translated_text="Lord"),
                TranslationPlacement(page=2, position="top", translated_text="Lord"),
                TranslationPlacement(page=2, position="top", translated_text="O Lord"),
            ]
        ],
    )
    assert [(p.page, p.position, p.translated_text) for p in merged.placements] == [
        (2, "top", "Lord"),
        (2, "top", "O Lord"),
        (2, "bottom", "Have mercy"),
    ]


class _FakeResponses:
    def __init__(self):
        self.last_kwargs = None

    async def parse(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            output=[],
            output_parsed=TranslationResult(
                target_language="English",
                placements=[TranslationPlacement(page=2, position="top", translated_text="Text")],
            ),
        )


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


def test_openai_service_builds_interleaved_markers_and_no_base64_in_logs(tmp_path: Path, caplog):
    image_paths = []
    for idx in range(3):
        image_path = tmp_path / f"page_{idx + 1:04d}.png"
        Image.new("RGB", (60, 40), color="white").save(image_path)
        image_paths.append(image_path)

    fake_client = _FakeClient()
    settings = Settings(openai_api_key="sk-test")
    service = OpenAIService(settings=settings, client=fake_client)
    batch = BatchSpec(index=1, owned_start=2, owned_end=2, supplied_start=1, supplied_end=3)

    translation_result, batch_log = asyncio.run(
        service.translate_batch(
            batch=batch,
            target_language="English",
            image_paths=image_paths,
            provider="openai",
            model_name=None,
            position_variant=POSITION_VARIANT_STANDARD,
        )
    )
    assert translation_result.placements[0].translated_text == "Text"
    assert batch_log.pages_sent_count == 3
    assert batch_log.prompt_sent["image_count"] == 3

    kwargs = fake_client.responses.last_kwargs
    assert kwargs is not None
    user_content = kwargs["input"][1]["content"]
    assert user_content[0]["type"] == "input_text"
    assert "target_language: English" in user_content[0]["text"]
    assert "OWNED_PAGES: 2-2" in user_content[0]["text"]

    marker_and_image_types = [item["type"] for item in user_content[1:]]
    assert marker_and_image_types == [
        "input_text",
        "input_image",
        "input_text",
        "input_image",
        "input_text",
        "input_image",
    ]
    assert user_content[1]["text"] == "PDF_PAGE_NUMBER: 1"
    assert user_content[2]["image_url"].startswith("data:image/png;base64,")

    joined_logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "data:image/png;base64" not in joined_logs


class _FakeHTTPResponse:
    def __init__(self, *, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("No JSON payload")
        return self._payload


class _FakeHTTPClient:
    def __init__(self, response: _FakeHTTPResponse):
        self._response = response
        self.last_kwargs = None

    async def post(self, url, **kwargs):
        self.last_kwargs = {"url": url, **kwargs}
        return self._response


def _contains_key_deep(payload, key: str) -> bool:
    if isinstance(payload, dict):
        if key in payload:
            return True
        return any(_contains_key_deep(value, key) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_key_deep(value, key) for value in payload)
    return False


def test_gemini_uses_header_auth_and_surfaces_provider_error_details(tmp_path: Path):
    image_path = tmp_path / "page_0001.png"
    Image.new("RGB", (60, 40), color="white").save(image_path)
    fake_http = _FakeHTTPClient(
        _FakeHTTPResponse(
            status_code=400,
            payload={"error": {"message": "API key not valid. Please pass a valid API key."}},
        )
    )
    settings = Settings(gemini_api_key="AQ.test-key\u200b ")
    service = OpenAIService(settings=settings, http_client=fake_http)
    batch = BatchSpec(index=1, owned_start=1, owned_end=1, supplied_start=1, supplied_end=1)

    with pytest.raises(PermanentOpenAIError) as exc_info:
        asyncio.run(
            service.translate_batch(
                batch=batch,
                target_language="English",
                image_paths=[image_path],
                provider="gemini",
                model_name="gemini-2.5-flash",
                position_variant=POSITION_VARIANT_STANDARD,
            )
        )

    assert "API key not valid" in str(exc_info.value)
    assert fake_http.last_kwargs is not None
    assert fake_http.last_kwargs["headers"]["x-goog-api-key"] == "AQ.test-key"
    assert "params" not in fake_http.last_kwargs


def test_gemini_response_schema_is_inline_and_ref_free(tmp_path: Path):
    image_path = tmp_path / "page_0001.png"
    Image.new("RGB", (60, 40), color="white").save(image_path)
    fake_http = _FakeHTTPClient(
        _FakeHTTPResponse(
            status_code=200,
            payload={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        '{"target_language":"English","placements":'
                                        '[{"page":1,"position":"top","translated_text":"Text"}]}'
                                    )
                                }
                            ]
                        }
                    }
                ]
            },
        )
    )
    settings = Settings(gemini_api_key="AQ.test-key")
    service = OpenAIService(settings=settings, http_client=fake_http)
    batch = BatchSpec(index=1, owned_start=1, owned_end=1, supplied_start=1, supplied_end=1)

    translation_result, _ = asyncio.run(
        service.translate_batch(
            batch=batch,
            target_language="English",
            image_paths=[image_path],
            provider="gemini",
            model_name="gemini-2.5-flash",
            position_variant=POSITION_VARIANT_STANDARD,
        )
    )

    assert translation_result.placements[0].position == "top"
    assert fake_http.last_kwargs is not None
    response_schema = fake_http.last_kwargs["json"]["generationConfig"]["responseSchema"]
    assert response_schema["type"] == "OBJECT"
    assert response_schema["properties"]["placements"]["items"]["type"] == "OBJECT"
    assert response_schema["properties"]["placements"]["items"]["properties"]["position"]["enum"] == [
        "top",
        "middle",
        "bottom",
    ]
    assert not _contains_key_deep(response_schema, "$defs")
    assert not _contains_key_deep(response_schema, "$ref")

