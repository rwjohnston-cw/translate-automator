from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import JobStatus, JobTextResult


def _build_client(tmp_path: Path, **overrides: object) -> TestClient:
    settings = Settings(
        openai_api_key="sk-test",
        max_upload_mb=1,
        max_pages=100,
        job_root=tmp_path / "jobs",
        cleanup_interval_seconds=3600,
        **overrides,
    )
    app = create_app(settings=settings)
    return TestClient(app)


def test_reject_non_pdf_upload(tmp_path: Path):
    with _build_client(tmp_path) as client:
        response = client.post(
            "/api/jobs",
            data={"target_language": "English"},
            files={"pdf_file": ("notes.txt", b"plain text", "text/plain")},
        )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_reject_oversized_pdf(tmp_path: Path):
    with _build_client(tmp_path) as client:
        oversized = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10)
        response = client.post(
            "/api/jobs",
            data={"target_language": "English"},
            files={"pdf_file": ("big.pdf", oversized, "application/pdf")},
        )
    assert response.status_code == 413
    assert response.json()["detail"] == "The uploaded file is too large. Maximum upload size is 1 MB."


def test_reject_missing_language(tmp_path: Path, make_pdf_bytes):
    with _build_client(tmp_path) as client:
        response = client.post(
            "/api/jobs",
            data={"target_language": ""},
            files={"pdf_file": ("score.pdf", make_pdf_bytes(page_count=1), "application/pdf")},
        )
    assert response.status_code == 400
    assert "language" in response.json()["detail"].lower()


def test_invalid_job_id_returns_404(tmp_path: Path):
    with _build_client(tmp_path) as client:
        status_response = client.get("/api/jobs/not-a-uuid")
        download_response = client.get("/api/jobs/not-a-uuid/download")
        log_response = client.get("/api/jobs/not-a-uuid/log")

    assert status_response.status_code == 404
    assert download_response.status_code == 404
    assert log_response.status_code == 404


def test_reject_invalid_batching_overrides(tmp_path: Path, make_pdf_bytes):
    with _build_client(tmp_path) as client:
        response = client.post(
            "/api/jobs",
            data={
                "target_language": "English",
                "testing_mode": "on",
                "owned_batch_size": "0",
                "context_pages": "1",
            },
            files={"pdf_file": ("score.pdf", make_pdf_bytes(page_count=1), "application/pdf")},
        )
    assert response.status_code == 400
    assert "Owned batch size" in response.json()["detail"]


def test_reject_invalid_workflow_mode(tmp_path: Path, make_pdf_bytes):
    with _build_client(tmp_path) as client:
        response = client.post(
            "/api/jobs",
            data={
                "target_language": "English",
                "testing_mode": "on",
                "translation_workflow": "bad-mode",
            },
            files={"pdf_file": ("score.pdf", make_pdf_bytes(page_count=1), "application/pdf")},
        )
    assert response.status_code == 400
    assert "workflow" in response.json()["detail"].lower()


def test_reject_invalid_reasoning_effort_for_model(tmp_path: Path, make_pdf_bytes):
    with _build_client(tmp_path) as client:
        response = client.post(
            "/api/jobs",
            data={
                "target_language": "English",
                "testing_mode": "on",
                "llm_provider": "openai",
                "llm_model": "gpt-5-mini",
                "llm_reasoning_effort": "none",
            },
            files={"pdf_file": ("score.pdf", make_pdf_bytes(page_count=1), "application/pdf")},
        )
    assert response.status_code == 400
    assert "reasoning effort" in response.json()["detail"].lower()


def test_accept_valid_reasoning_effort_for_model(tmp_path: Path, make_pdf_bytes):
    with _build_client(tmp_path) as client:
        response = client.post(
            "/api/jobs",
            data={
                "target_language": "English",
                "testing_mode": "on",
                "llm_provider": "openai",
                "llm_model": "gpt-5-mini",
                "llm_reasoning_effort": "minimal",
            },
            files={"pdf_file": ("score.pdf", make_pdf_bytes(page_count=1), "application/pdf")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"]
    assert payload["page_url"].startswith("/jobs/")


def test_rate_limit_blocks_repeated_job_creation(tmp_path: Path, make_pdf_bytes):
    with _build_client(
        tmp_path,
        rate_limit_create_requests=2,
        rate_limit_create_window_seconds=600,
        rate_limit_api_requests=1000,
    ) as client:
        first = client.post(
            "/api/jobs",
            data={"target_language": "English"},
            files={"pdf_file": ("score-1.pdf", make_pdf_bytes(page_count=1), "application/pdf")},
        )
        second = client.post(
            "/api/jobs",
            data={"target_language": "English"},
            files={"pdf_file": ("score-2.pdf", make_pdf_bytes(page_count=1), "application/pdf")},
        )
        blocked = client.post(
            "/api/jobs",
            data={"target_language": "English"},
            files={"pdf_file": ("score-3.pdf", make_pdf_bytes(page_count=1), "application/pdf")},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert blocked.status_code == 429
    assert "too many requests" in blocked.json()["detail"].lower()
    assert int(blocked.headers["Retry-After"]) >= 1


def test_rate_limit_uses_forwarded_ip(tmp_path: Path, make_pdf_bytes):
    with _build_client(
        tmp_path,
        rate_limit_create_requests=1,
        rate_limit_create_window_seconds=600,
        rate_limit_api_requests=1000,
        trust_proxy_headers=True,
    ) as client:
        first = client.post(
            "/api/jobs",
            headers={"x-forwarded-for": "203.0.113.10"},
            data={"target_language": "English"},
            files={"pdf_file": ("score-1.pdf", make_pdf_bytes(page_count=1), "application/pdf")},
        )
        second_same_ip = client.post(
            "/api/jobs",
            headers={"x-forwarded-for": "203.0.113.10"},
            data={"target_language": "English"},
            files={"pdf_file": ("score-2.pdf", make_pdf_bytes(page_count=1), "application/pdf")},
        )
        third_other_ip = client.post(
            "/api/jobs",
            headers={"x-forwarded-for": "203.0.113.11"},
            data={"target_language": "English"},
            files={"pdf_file": ("score-3.pdf", make_pdf_bytes(page_count=1), "application/pdf")},
        )

    assert first.status_code == 200
    assert second_same_ip.status_code == 429
    assert third_other_ip.status_code == 200


def test_text_result_and_ipa_endpoints(tmp_path: Path):
    settings = Settings(
        openai_api_key="sk-test",
        max_upload_mb=1,
        max_pages=100,
        job_root=tmp_path / "jobs",
        cleanup_interval_seconds=3600,
    )
    app = create_app(settings=settings)
    client = TestClient(app)
    store = app.state.job_store
    manifest = store.create_job(
        original_filename="score.pdf",
        target_language="English",
        page_count=1,
        llm_provider="openai",
        llm_model=None,
        llm_reasoning_effort=None,
        translation_workflow="batch_translate_and_place",
        positioning_variant="standard_3",
        testing_mode=False,
    )
    store.update_job(
        manifest.job_id,
        status=JobStatus.COMPLETE,
        output_available=True,
        text_result_available=True,
    )
    store.save_text_result(
        job_id=manifest.job_id,
        text_result=JobTextResult(
            source_language="German",
            full_source_text="Guten Tag Welt",
            target_language="English",
            full_translation="Good day world",
        ),
    )

    status_response = client.get(f"/api/jobs/{manifest.job_id}")
    assert status_response.status_code == 200
    assert status_response.json()["text_result_url"] == f"/api/jobs/{manifest.job_id}/text-result"

    text_result_response = client.get(f"/api/jobs/{manifest.job_id}/text-result")
    assert text_result_response.status_code == 200
    payload = text_result_response.json()
    assert payload["source_language"] == "German"
    assert payload["full_source_text"] == "Guten Tag Welt"
    assert payload["ipa_supported"] is True
    assert payload["default_ipa_variant"] == "de"

    ipa_response = client.post(
        f"/api/jobs/{manifest.job_id}/ipa",
        json={"variant": "de", "words": ["Guten", "Welt", "missing"]},
    )
    assert ipa_response.status_code == 200
    ipa_payload = ipa_response.json()
    assert ipa_payload["ipa_supported"] is True
    assert ipa_payload["variant"] == "de"
    assert "entries" in ipa_payload

    client.close()


def test_ipa_endpoint_reports_unsupported_language(tmp_path: Path):
    settings = Settings(
        openai_api_key="sk-test",
        max_upload_mb=1,
        max_pages=100,
        job_root=tmp_path / "jobs",
        cleanup_interval_seconds=3600,
    )
    app = create_app(settings=settings)
    client = TestClient(app)
    store = app.state.job_store
    manifest = store.create_job(
        original_filename="score.pdf",
        target_language="English",
        page_count=1,
        llm_provider="openai",
        llm_model=None,
        llm_reasoning_effort=None,
        translation_workflow="batch_translate_and_place",
        positioning_variant="standard_3",
        testing_mode=False,
    )
    store.update_job(
        manifest.job_id,
        status=JobStatus.COMPLETE,
        text_result_available=True,
    )
    store.save_text_result(
        job_id=manifest.job_id,
        text_result=JobTextResult(
            source_language="Latin",
            full_source_text="Kyrie eleison",
            target_language="English",
            full_translation="Lord have mercy",
        ),
    )

    text_result_response = client.get(f"/api/jobs/{manifest.job_id}/text-result")
    assert text_result_response.status_code == 200
    assert text_result_response.json()["ipa_supported"] is False

    ipa_response = client.post(
        f"/api/jobs/{manifest.job_id}/ipa",
        json={"words": ["Kyrie"]},
    )
    assert ipa_response.status_code == 200
    assert ipa_response.json()["ipa_supported"] is False
    client.close()

