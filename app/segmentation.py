from __future__ import annotations

import re
import threading
from dataclasses import dataclass

from app.ipa_lookup import normalize_display_text, normalize_lookup_word

CJK_VARIANTS = frozenset({"ja", "zh_hans", "zh_hant", "yue"})
CHINESE_VARIANTS = frozenset({"zh_hans", "zh_hant", "yue"})

_JAPANESE_TOKENIZER = None
_JAPANESE_TOKENIZER_LOCK = threading.Lock()
_JIEBA_INITIALIZED = False
_JIEBA_LOCK = threading.Lock()

# Word chars, punctuation runs, and whitespace — mirrors the client tokenizer intent.
_TOKEN_PATTERN = re.compile(
    r"[\w'\u2019\u2018-]+|[^\w\s'\u2019\u2018-]+|\s+",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class SegmentToken:
    text: str
    is_word: bool
    lookup_keys: tuple[str, ...]


def _dedupe_lookup_keys(*keys: str) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        cleaned = normalize_display_text(key.strip())
        if not cleaned:
            continue
        normalized = normalize_lookup_word(cleaned)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(cleaned)
    return tuple(ordered)


def _get_japanese_tokenizer():
    global _JAPANESE_TOKENIZER
    if _JAPANESE_TOKENIZER is not None:
        return _JAPANESE_TOKENIZER
    with _JAPANESE_TOKENIZER_LOCK:
        if _JAPANESE_TOKENIZER is None:
            from sudachipy import Dictionary
            from sudachipy import SplitMode

            _JAPANESE_TOKENIZER = (
                Dictionary(dict="core").create(),
                SplitMode.C,
            )
    return _JAPANESE_TOKENIZER


def _ensure_jieba_initialized() -> None:
    global _JIEBA_INITIALIZED
    if _JIEBA_INITIALIZED:
        return
    with _JIEBA_LOCK:
        if not _JIEBA_INITIALIZED:
            import jieba

            jieba.initialize()
            _JIEBA_INITIALIZED = True


def _segment_default(text: str) -> list[SegmentToken]:
    normalized = normalize_display_text(text)
    if not normalized:
        return []

    tokens: list[SegmentToken] = []
    for match in _TOKEN_PATTERN.finditer(normalized):
        token_text = match.group(0)
        is_word = bool(re.fullmatch(r"[\w'\u2019\u2018-]+", token_text, flags=re.UNICODE))
        lookup_keys: tuple[str, ...] = ()
        if is_word:
            lookup_key = normalize_lookup_word(token_text)
            if lookup_key:
                lookup_keys = (normalize_display_text(token_text),)
        tokens.append(
            SegmentToken(
                text=token_text,
                is_word=is_word,
                lookup_keys=lookup_keys,
            )
        )
    return tokens


def _segment_japanese(text: str) -> list[SegmentToken]:
    normalized = normalize_display_text(text)
    if not normalized:
        return []

    tokenizer, split_mode = _get_japanese_tokenizer()
    morphemes = tokenizer.tokenize(normalized, split_mode)
    tokens: list[SegmentToken] = []
    cursor = 0

    for morpheme in morphemes:
        surface = morpheme.surface()
        if not surface:
            continue

        start = normalized.find(surface, cursor)
        if start == -1:
            start = cursor
        if start > cursor:
            gap = normalized[cursor:start]
            tokens.append(SegmentToken(text=gap, is_word=False, lookup_keys=()))

        lookup_keys = _dedupe_lookup_keys(
            surface,
            morpheme.dictionary_form(),
            morpheme.normalized_form(),
        )
        tokens.append(
            SegmentToken(
                text=surface,
                is_word=bool(lookup_keys),
                lookup_keys=lookup_keys,
            )
        )
        cursor = start + len(surface)

    if cursor < len(normalized):
        tokens.append(
            SegmentToken(
                text=normalized[cursor:],
                is_word=False,
                lookup_keys=(),
            )
        )
    return tokens


def _segment_chinese(variant_code: str, text: str) -> list[SegmentToken]:
    normalized = normalize_display_text(text)
    if not normalized:
        return []

    _ensure_jieba_initialized()
    import jieba

    tokens: list[SegmentToken] = []
    cursor = 0
    for segment in jieba.cut(normalized, HMM=True):
        if not segment:
            continue

        start = normalized.find(segment, cursor)
        if start == -1:
            start = cursor
        if start > cursor:
            gap = normalized[cursor:start]
            tokens.append(SegmentToken(text=gap, is_word=False, lookup_keys=()))

        lookup_keys = _dedupe_lookup_keys(segment)
        tokens.append(
            SegmentToken(
                text=segment,
                is_word=bool(lookup_keys),
                lookup_keys=lookup_keys,
            )
        )
        cursor = start + len(segment)

    if cursor < len(normalized):
        tokens.append(
            SegmentToken(
                text=normalized[cursor:],
                is_word=False,
                lookup_keys=(),
            )
        )
    return tokens


def segment(*, variant_code: str, text: str) -> list[SegmentToken]:
    if variant_code == "ja":
        return _segment_japanese(text)
    if variant_code in CHINESE_VARIANTS:
        return _segment_chinese(variant_code, text)
    return _segment_default(text)
