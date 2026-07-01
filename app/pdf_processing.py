from __future__ import annotations

import base64
import html
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import fitz

from app.models import (
    POSITION_VARIANT_CENTERED_5,
    POSITION_VARIANT_GRID_20,
    POSITION_VARIANT_SPLIT_6,
    POSITION_VARIANT_STANDARD,
    TranslationPlacement,
    TranslationResult,
    positions_for_variant,
)

LOGGER = logging.getLogger(__name__)

REGION_RECTS_BY_VARIANT: dict[str, dict[str, tuple[float, float, float, float]]] = {
    POSITION_VARIANT_STANDARD: {
        "top": (0.05, 0.02, 0.95, 0.13),
        "middle": (0.05, 0.44, 0.95, 0.56),
        "bottom": (0.05, 0.87, 0.95, 0.98),
    },
    POSITION_VARIANT_CENTERED_5: {
        "top": (0.05, 0.02, 0.95, 0.12),
        "upper_middle": (0.05, 0.25, 0.95, 0.35),
        "middle": (0.05, 0.44, 0.95, 0.56),
        "lower_middle": (0.05, 0.65, 0.95, 0.75),
        "bottom": (0.05, 0.87, 0.95, 0.98),
    },
    POSITION_VARIANT_SPLIT_6: {
        "top_left": (0.03, 0.02, 0.47, 0.13),
        "top_right": (0.53, 0.02, 0.97, 0.13),
        "middle_left": (0.03, 0.44, 0.47, 0.56),
        "middle_right": (0.53, 0.44, 0.97, 0.56),
        "bottom_left": (0.03, 0.87, 0.47, 0.98),
        "bottom_right": (0.53, 0.87, 0.97, 0.98),
    },
    POSITION_VARIANT_GRID_20: {
        "a1": (0.03, 0.02, 0.24, 0.11),
        "a2": (0.27, 0.02, 0.48, 0.11),
        "a3": (0.52, 0.02, 0.73, 0.11),
        "a4": (0.76, 0.02, 0.97, 0.11),
        "b1": (0.03, 0.20, 0.24, 0.29),
        "b2": (0.27, 0.20, 0.48, 0.29),
        "b3": (0.52, 0.20, 0.73, 0.29),
        "b4": (0.76, 0.20, 0.97, 0.29),
        "c1": (0.03, 0.39, 0.24, 0.48),
        "c2": (0.27, 0.39, 0.48, 0.48),
        "c3": (0.52, 0.39, 0.73, 0.48),
        "c4": (0.76, 0.39, 0.97, 0.48),
        "d1": (0.03, 0.58, 0.24, 0.67),
        "d2": (0.27, 0.58, 0.48, 0.67),
        "d3": (0.52, 0.58, 0.73, 0.67),
        "d4": (0.76, 0.58, 0.97, 0.67),
        "e1": (0.03, 0.83, 0.20, 0.95),
        "e2": (0.22, 0.83, 0.39, 0.95),
        "e3": (0.41, 0.83, 0.58, 0.95),
        "e4": (0.60, 0.83, 0.77, 0.95),
        "e5": (0.79, 0.83, 0.96, 0.95),
    },
}

FONT_FAMILY_CSS = """
font-family:'Noto Sans','Noto Sans Arabic','Noto Sans Hebrew','Noto Sans Devanagari','Noto Sans CJK SC',sans-serif;
"""


class PDFValidationError(ValueError):
    """Raised when a user upload is not an acceptable PDF."""


@dataclass(frozen=True)
class BatchSpec:
    index: int
    owned_start: int
    owned_end: int
    supplied_start: int
    supplied_end: int

    @property
    def owned_pages(self) -> range:
        return range(self.owned_start, self.owned_end + 1)

    @property
    def supplied_pages(self) -> range:
        return range(self.supplied_start, self.supplied_end + 1)

    @property
    def context_only_pages(self) -> list[int]:
        return [p for p in self.supplied_pages if p not in self.owned_pages]


def sanitize_stem(filename: str) -> str:
    stem = Path(filename).stem
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if not normalized:
        return "score"
    return normalized[:120]


def sanitize_language(language: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", language).strip("._-")
    return normalized[:80] or "target"


def validate_pdf_upload(
    *,
    filename: str,
    payload: bytes,
    max_upload_bytes: int,
    max_pages: int,
) -> int:
    if not filename.lower().endswith(".pdf"):
        raise PDFValidationError("Please upload a PDF file.")
    if len(payload) == 0:
        raise PDFValidationError("The uploaded file is empty.")
    if len(payload) > max_upload_bytes:
        max_upload_mb = max_upload_bytes / (1024 * 1024)
        raise PDFValidationError(
            f"The uploaded file is too large. Maximum upload size is {max_upload_mb:g} MB."
        )
    if not payload.startswith(b"%PDF-"):
        raise PDFValidationError("The uploaded file is not a valid PDF.")

    try:
        with fitz.open(stream=payload, filetype="pdf") as doc:
            if doc.needs_pass:
                raise PDFValidationError("Password-protected PDFs are not supported.")
            page_count = doc.page_count
    except PDFValidationError:
        raise
    except Exception as exc:  # pragma: no cover - fitz exception details vary by version
        raise PDFValidationError("Unable to read this PDF file.") from exc

    if page_count < 1:
        raise PDFValidationError("The PDF must contain at least one page.")
    if page_count > max_pages:
        raise PDFValidationError(
            f"The PDF has too many pages ({page_count}). Maximum allowed is {max_pages}."
        )
    return page_count


def render_pdf_pages_to_images(
    *,
    pdf_path: Path,
    output_dir: Path,
    dpi: int,
    max_dimension: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []

    with fitz.open(pdf_path) as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            base_scale = dpi / 72.0
            width = page.rect.width * base_scale
            height = page.rect.height * base_scale
            longest_side = max(width, height)
            resize_factor = min(1.0, max_dimension / longest_side) if longest_side else 1.0
            final_scale = base_scale * resize_factor
            matrix = fitz.Matrix(final_scale, final_scale)

            pix = page.get_pixmap(matrix=matrix, alpha=False)
            page_no = page_index + 1
            output_path = output_dir / f"page_{page_no:04d}.png"
            pix.save(output_path)
            images.append(output_path)

    return images


def build_image_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_batches(total_pages: int, owned_batch_size: int, context_pages: int) -> list[BatchSpec]:
    batches: list[BatchSpec] = []
    if total_pages < 1:
        return batches

    index = 1
    for owned_start in range(1, total_pages + 1, owned_batch_size):
        owned_end = min(total_pages, owned_start + owned_batch_size - 1)
        supplied_start = max(1, owned_start - context_pages)
        supplied_end = min(total_pages, owned_end + context_pages)
        batches.append(
            BatchSpec(
                index=index,
                owned_start=owned_start,
                owned_end=owned_end,
                supplied_start=supplied_start,
                supplied_end=supplied_end,
            )
        )
        index += 1
    return batches


def position_order_for_variant(position_variant: str) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(positions_for_variant(position_variant))}


def build_batch_header(batch: BatchSpec, target_language: str, position_variant: str) -> str:
    context_text = ", ".join(str(p) for p in batch.context_only_pages) or "none"
    allowed_positions = ", ".join(positions_for_variant(position_variant))
    return (
        f"target_language: {target_language}\n"
        f"SUPPLIED_PAGES: {batch.supplied_start}-{batch.supplied_end}\n"
        f"OWNED_PAGES: {batch.owned_start}-{batch.owned_end}\n"
        f"POSITION_VARIANT: {position_variant}\n"
        f"ALLOWED_POSITIONS: {allowed_positions}\n"
        f"CONTEXT_ONLY_PAGES: {context_text}\n"
        "Use context pages for continuity, but return placements only "
        f"for pages {batch.owned_start} through {batch.owned_end}. "
        "Use as few placements as needed for distinct textual events. "
        "Do not duplicate the same text across multiple positions merely because "
        "space is available or multiple voices sing it at once. "
        "Repeating important text on later pages is acceptable when the score text "
        "continues or returns there. "
        "Only split when it improves locality for genuinely distinct events. "
        "Before producing placements, reconstruct the full owned-page text and produce "
        "one complete full_translation value with intended line breaks. "
        "Then derive placements from that complete translation."
    )


def clean_and_filter_batch_placements(
    *,
    placements: Sequence[TranslationPlacement],
    batch: BatchSpec,
    allowed_positions: set[str],
    logger: logging.Logger = LOGGER,
) -> list[TranslationPlacement]:
    filtered: list[TranslationPlacement] = []
    for placement in placements:
        if placement.page < batch.owned_start or placement.page > batch.owned_end:
            logger.warning(
                "Discarding out-of-range placement page=%s owned=%s-%s",
                placement.page,
                batch.owned_start,
                batch.owned_end,
            )
            continue
        text = placement.translated_text.strip()
        if not text:
            logger.warning(
                "Discarding empty translated fragment page=%s position=%s",
                placement.page,
                placement.position,
            )
            continue
        if placement.position not in allowed_positions:
            logger.warning(
                "Discarding invalid position page=%s position=%s",
                placement.page,
                placement.position,
            )
            continue
        filtered.append(
            TranslationPlacement(
                page=placement.page,
                position=placement.position,
                translated_text=text,
            )
        )
    return filtered


def _normalize_line_for_overlap(value: str) -> str:
    return " ".join(value.split())


def _tokenize_for_overlap(value: str) -> list[str]:
    normalized = re.sub(r"[^\w\s]+", " ", value.lower())
    return [token for token in normalized.split() if token]


def _find_token_overlap(
    previous_lines: Sequence[str],
    candidate_lines: Sequence[str],
    *,
    max_tokens: int = 90,
    min_tokens: int = 5,
) -> int:
    prev_tokens = _tokenize_for_overlap(" ".join(previous_lines))
    candidate_tokens = _tokenize_for_overlap(" ".join(candidate_lines))
    max_overlap = min(max_tokens, len(prev_tokens), len(candidate_tokens))
    for size in range(max_overlap, min_tokens - 1, -1):
        if prev_tokens[-size:] == candidate_tokens[:size]:
            return size
    return 0


def _estimate_overlapped_line_count(candidate_lines: Sequence[str], token_overlap: int) -> int:
    if token_overlap <= 0:
        return 0
    consumed = 0
    for index, line in enumerate(candidate_lines):
        consumed += len(_tokenize_for_overlap(line))
        if consumed >= token_overlap:
            return index + 1
    return len(candidate_lines)


def _merge_full_translations(full_translations: Iterable[str]) -> str:
    merged_lines: list[str] = []
    for chunk in full_translations:
        normalized_chunk = chunk.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized_chunk:
            continue
        candidate_lines = normalized_chunk.split("\n")
        if not merged_lines:
            merged_lines.extend(candidate_lines)
            continue

        max_overlap = min(len(merged_lines), len(candidate_lines), 12)
        overlap = 0
        for size in range(max_overlap, 0, -1):
            if [
                _normalize_line_for_overlap(line) for line in merged_lines[-size:]
            ] == [
                _normalize_line_for_overlap(line) for line in candidate_lines[:size]
            ]:
                overlap = size
                break
        if overlap == 0:
            token_overlap = _find_token_overlap(merged_lines, candidate_lines)
            if token_overlap > 0:
                overlap = _estimate_overlapped_line_count(candidate_lines, token_overlap)
        merged_lines.extend(candidate_lines[overlap:])
    return "\n".join(merged_lines).strip()


def merge_batch_results(
    *,
    target_language: str,
    position_order: dict[str, int],
    placement_groups: Iterable[Sequence[TranslationPlacement]],
    full_translations: Iterable[str],
    full_translation_override: str | None = None,
) -> TranslationResult:
    all_items: list[TranslationPlacement] = []
    for group in placement_groups:
        all_items.extend(group)

    all_items = sorted(
        all_items,
        key=lambda item: (item.page, position_order.get(item.position, 999)),
    )

    seen: set[tuple[int, str, str]] = set()
    deduped: list[TranslationPlacement] = []
    for item in all_items:
        normalized_text = " ".join(item.translated_text.split())
        key = (item.page, item.position, normalized_text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            TranslationPlacement(
                page=item.page,
                position=item.position,
                translated_text=item.translated_text.strip(),
            )
        )

    return TranslationResult(
        target_language=target_language,
        full_translation=(
            full_translation_override.strip()
            if full_translation_override and full_translation_override.strip()
            else _merge_full_translations(full_translations)
        ),
        placements=deduped,
    )


def build_output_filename(original_stem: str, target_language: str) -> str:
    return f"{sanitize_stem(original_stem)}_{sanitize_language(target_language)}_translation.pdf"


def _build_font_archive() -> fitz.Archive | None:
    candidates = [
        Path("/usr/share/fonts/truetype/noto"),
        Path("/usr/share/fonts/noto"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return fitz.Archive(str(candidate))
            except Exception:
                continue
    return None


def _region_rect(page_rect: fitz.Rect, position: str, position_variant: str) -> fitz.Rect:
    rects = REGION_RECTS_BY_VARIANT.get(
        position_variant,
        REGION_RECTS_BY_VARIANT[POSITION_VARIANT_STANDARD],
    )
    x0, y0, x1, y1 = rects[position]
    return fitz.Rect(
        page_rect.x0 + page_rect.width * x0,
        page_rect.y0 + page_rect.height * y0,
        page_rect.x0 + page_rect.width * x1,
        page_rect.y0 + page_rect.height * y1,
    )


def _expanded_rect(base: fitz.Rect, page_rect: fitz.Rect, position: str, factor: float) -> fitz.Rect:
    if factor <= 0:
        return fitz.Rect(base)

    expansion = page_rect.height * factor
    if position == "top":
        y0 = base.y0
        y1 = base.y1 + expansion
    elif position == "bottom":
        y0 = base.y0 - expansion
        y1 = base.y1
    else:
        y0 = base.y0 - (expansion / 2)
        y1 = base.y1 + (expansion / 2)

    return fitz.Rect(
        max(page_rect.x0, base.x0),
        max(page_rect.y0, y0),
        min(page_rect.x1, base.x1),
        min(page_rect.y1, y1),
    )


def _rect_for_page_space(page: fitz.Page, rect: fitz.Rect) -> fitz.Rect:
    mapped = fitz.Rect(rect)
    if page.rotation % 360:
        # Placement regions are defined in display orientation. Convert to the
        # page's underlying coordinate system before inserting overlay content.
        mapped = mapped * page.derotation_matrix

    bounds = page.cropbox
    x0 = max(bounds.x0, min(bounds.x1, mapped.x0))
    y0 = max(bounds.y0, min(bounds.y1, mapped.y0))
    x1 = max(bounds.x0, min(bounds.x1, mapped.x1))
    y1 = max(bounds.y0, min(bounds.y1, mapped.y1))
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return fitz.Rect(x0, y0, x1, y1)


def _insert_html_with_fallback(
    *,
    page: fitz.Page,
    position: str,
    text_lines: Sequence[str],
    base_rect: fitz.Rect,
    default_font_size: float,
    min_font_size: float,
    opacity: float,
    archive: fitz.Archive | None,
) -> None:
    escaped_lines = [html.escape(line) for line in text_lines]
    body = "<br/>".join(escaped_lines)
    if not body:
        return

    expansion_steps = [0.0, 0.01, 0.02, 0.03]
    font_sizes = []
    size = default_font_size
    while size >= min_font_size:
        font_sizes.append(round(size, 2))
        size -= 0.5

    for expansion in expansion_steps:
        rect = _expanded_rect(base_rect, page.rect, position, expansion)
        page_space_rect = _rect_for_page_space(page, rect)
        for font_size in font_sizes:
            min_scale = max(0.6, min_font_size / max(font_size, 0.1))
            html_markup = (
                '<div dir="auto" style="'
                "text-align:center;"
                f"font-size:{font_size}pt;"
                "line-height:1.25;"
                "color:#c1121f;"
                f"background:rgba(255,255,255,{opacity});"
                "border:0.6px solid #cccccc;"
                "border-radius:4px;"
                "padding:4px 6px;"
                "word-break:break-word;"
                f"{FONT_FAMILY_CSS}"
                f'">{body}</div>'
            )
            result = page.insert_htmlbox(
                page_space_rect,
                html_markup,
                css="* { box-sizing: border-box; }",
                archive=archive,
                scale_low=min_scale,
            )

            if isinstance(result, tuple):
                spare_height = result[0] if len(result) > 0 else 0
                scale_used = result[1] if len(result) > 1 else 1.0
                if spare_height >= -1e-6 and scale_used >= min_scale - 1e-6:
                    return
            elif isinstance(result, (float, int)):
                if result >= -1e-6:
                    return
            else:
                return

    truncated = html.escape(" ".join(" ".join(text_lines).split())[:450].rstrip())
    if len(truncated) >= 450:
        truncated += "..."
    fallback_rect = _expanded_rect(base_rect, page.rect, position, expansion_steps[-1])
    fallback_page_space_rect = _rect_for_page_space(page, fallback_rect)
    fallback_html = (
        '<div dir="auto" style="'
        "text-align:center;"
        f"font-size:{min_font_size}pt;"
        "line-height:1.25;"
        "color:#c1121f;"
        f"background:rgba(255,255,255,{opacity});"
        "border:0.6px solid #cccccc;"
        "border-radius:4px;"
        "padding:4px 6px;"
        "word-break:break-word;"
        f"{FONT_FAMILY_CSS}"
        f'">{truncated}</div>'
    )
    page.insert_htmlbox(
        fallback_page_space_rect,
        fallback_html,
        css="* { box-sizing: border-box; }",
        archive=archive,
        scale_low=0.6,
    )
    LOGGER.warning("Applied truncated fallback for crowded %s translation box", position)


def _insert_full_translation_page(
    *,
    doc: fitz.Document,
    full_translation: str,
    output_font_size: float,
    min_font_size: float,
    archive: fitz.Archive | None,
) -> bool:
    normalized = full_translation.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized or doc.page_count < 1:
        return False

    reference_rect = doc[0].rect
    page = doc.new_page(pno=0, width=reference_rect.width, height=reference_rect.height)
    margin = min(page.rect.width, page.rect.height) * 0.06
    content_rect = fitz.Rect(
        page.rect.x0 + margin,
        page.rect.y0 + margin,
        page.rect.x1 - margin,
        page.rect.y1 - margin,
    )

    lines = normalized.split("\n")
    escaped_body = "<br/>".join(html.escape(line) if line else "&nbsp;" for line in lines)
    body = (
        '<div dir="auto" style="'
        "text-align:left;"
        "line-height:1.35;"
        "color:#111111;"
        f"{FONT_FAMILY_CSS}"
        '">'
        "<strong>Complete translation</strong><br/><br/>"
        f"{escaped_body}</div>"
    )

    start_size = max(min(18.0, output_font_size + 2.0), min_font_size)
    size = start_size
    while size >= min_font_size:
        result = page.insert_htmlbox(
            content_rect,
            body,
            css=(
                "html, body { margin: 0; padding: 0; background: #ffffff; } "
                f"div {{ font-size: {size:.2f}pt; }}"
            ),
            archive=archive,
            scale_low=max(0.55, min_font_size / max(size, 0.1)),
        )
        if isinstance(result, tuple):
            spare_height = result[0] if len(result) > 0 else 0
            if spare_height >= -1e-6:
                return True
        elif isinstance(result, (float, int)):
            if result >= -1e-6:
                return True
        else:
            return True
        size -= 0.5

    LOGGER.warning("Inserted full translation page using minimal scale fallback")
    page.insert_htmlbox(
        content_rect,
        body,
        css=(
            "html, body { margin: 0; padding: 0; background: #ffffff; } "
            f"div {{ font-size: {min_font_size:.2f}pt; }}"
        ),
        archive=archive,
        scale_low=0.5,
    )
    return True


def create_translated_pdf(
    *,
    original_pdf_path: Path,
    output_pdf_path: Path,
    translation_result: TranslationResult,
    output_font_size: float,
    min_font_size: float,
    output_background_opacity: float,
    position_variant: str,
) -> None:
    positions_in_order = positions_for_variant(position_variant)
    page_map: dict[int, dict[str, list[str]]] = {}
    for placement in translation_result.placements:
        page_positions = page_map.setdefault(
            placement.page,
            {position: [] for position in positions_in_order},
        )
        if placement.position not in page_positions:
            LOGGER.warning(
                "Skipping placement with unexpected position in overlay position=%s variant=%s",
                placement.position,
                position_variant,
            )
            continue
        page_positions[placement.position].append(placement.translated_text)

    archive = _build_font_archive()
    effective_font_size = max(14.0, output_font_size)
    with fitz.open(original_pdf_path) as doc:
        page_offset = 0
        if _insert_full_translation_page(
            doc=doc,
            full_translation=translation_result.full_translation,
            output_font_size=effective_font_size,
            min_font_size=min_font_size,
            archive=archive,
        ):
            page_offset = 1

        for page_number, regions in page_map.items():
            mapped_page_number = page_number + page_offset
            if mapped_page_number < 1 or mapped_page_number > doc.page_count:
                LOGGER.warning("Skipping placement for non-existent page %s", page_number)
                continue
            page = doc.load_page(mapped_page_number - 1)
            for position in positions_in_order:
                lines = regions[position]
                if not lines:
                    continue
                rect = _region_rect(page.rect, position, position_variant)
                _insert_html_with_fallback(
                    page=page,
                    position=position,
                    text_lines=lines,
                    base_rect=rect,
                    default_font_size=effective_font_size,
                    min_font_size=min_font_size,
                    opacity=output_background_opacity,
                    archive=archive,
                )

        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(
            output_pdf_path,
            garbage=3,
            deflate=True,
            use_objstms=1,
        )

