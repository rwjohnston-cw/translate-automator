from __future__ import annotations

from pathlib import Path

import fitz

from app.models import (
    POSITION_VARIANT_GRID_20,
    POSITION_VARIANT_STANDARD,
    TranslationPlacement,
    TranslationResult,
    positions_for_variant,
)
from app.pdf_processing import build_batches, create_translated_pdf, render_pdf_pages_to_images


def test_render_pdf_pages_to_images_preserves_page_order(tmp_path: Path, make_pdf_bytes):
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(make_pdf_bytes(page_count=3))
    output_dir = tmp_path / "rendered"

    images = render_pdf_pages_to_images(
        pdf_path=pdf_path,
        output_dir=output_dir,
        dpi=200,
        max_dimension=2500,
    )

    assert [path.name for path in images] == [
        "page_0001.png",
        "page_0002.png",
        "page_0003.png",
    ]
    assert all(path.exists() for path in images)


def test_batch_construction_for_required_sizes():
    one = build_batches(total_pages=1, owned_batch_size=6, context_pages=1)
    assert [(b.supplied_start, b.supplied_end, b.owned_start, b.owned_end) for b in one] == [
        (1, 1, 1, 1)
    ]

    six = build_batches(total_pages=6, owned_batch_size=6, context_pages=1)
    assert [(b.supplied_start, b.supplied_end, b.owned_start, b.owned_end) for b in six] == [
        (1, 6, 1, 6)
    ]

    seven = build_batches(total_pages=7, owned_batch_size=6, context_pages=1)
    assert [(b.supplied_start, b.supplied_end, b.owned_start, b.owned_end) for b in seven] == [
        (1, 7, 1, 6),
        (6, 7, 7, 7),
    ]

    twenty = build_batches(total_pages=20, owned_batch_size=6, context_pages=1)
    assert [(b.supplied_start, b.supplied_end, b.owned_start, b.owned_end) for b in twenty] == [
        (1, 7, 1, 6),
        (6, 13, 7, 12),
        (12, 19, 13, 18),
        (18, 20, 19, 20),
    ]


def test_grid_20_variant_positions_are_available():
    labels = positions_for_variant(POSITION_VARIANT_GRID_20)
    assert labels[0] == "a1"
    assert labels[-1] == "e5"
    assert len(labels) == 21


def test_pdf_overlay_preserves_page_geometry_and_adds_text(tmp_path: Path, make_pdf_bytes):
    original_pdf = tmp_path / "original.pdf"
    output_pdf = tmp_path / "output.pdf"
    original_pdf.write_bytes(make_pdf_bytes(page_count=1))

    result = TranslationResult(
        target_language="English",
        placements=[
            TranslationPlacement(page=1, position="top", translated_text="Lamb of God"),
            TranslationPlacement(page=1, position="middle", translated_text="Have mercy"),
            TranslationPlacement(page=1, position="bottom", translated_text="Amen"),
        ],
    )

    create_translated_pdf(
        original_pdf_path=original_pdf,
        output_pdf_path=output_pdf,
        translation_result=result,
        output_font_size=11,
        min_font_size=7,
        output_background_opacity=0.88,
        position_variant=POSITION_VARIANT_STANDARD,
    )

    with fitz.open(original_pdf) as source, fitz.open(output_pdf) as translated:
        assert translated.page_count == source.page_count
        assert translated[0].rect == source[0].rect
        text = translated[0].get_text()
        assert "Lamb of God" in text
        assert "Have mercy" in text
        assert "Amen" in text

