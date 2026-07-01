from __future__ import annotations

import fitz
import pytest


@pytest.fixture
def make_pdf_bytes():
    def _make_pdf_bytes(page_count: int = 1) -> bytes:
        with fitz.open() as doc:
            for index in range(page_count):
                page = doc.new_page()
                page.insert_text((72, 72), f"Synthetic page {index + 1}")
            return doc.tobytes()

    return _make_pdf_bytes

