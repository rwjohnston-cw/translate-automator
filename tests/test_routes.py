from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def _build_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        openai_api_key="sk-test",
        max_upload_mb=1,
        max_pages=100,
        job_root=tmp_path / "jobs",
        cleanup_interval_seconds=3600,
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

