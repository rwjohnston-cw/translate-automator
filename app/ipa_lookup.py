from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IPA_DATA_DIR = BASE_DIR / "data" / "ipa-dict"

# ipa-dict variant code -> human label
IPA_VARIANT_LABELS: dict[str, str] = {
    "en_US": "American English",
    "en_UK": "Received Pronunciation",
    "fr_FR": "French (France)",
    "fr_QC": "French (Québec)",
    "es_ES": "Spanish (Spain)",
    "es_MX": "Spanish (Mexico)",
    "vi_N": "Vietnamese (Northern)",
    "vi_C": "Vietnamese (Central)",
    "vi_S": "Vietnamese (Southern)",
    "zh_hans": "Mandarin (Simplified)",
    "zh_hant": "Mandarin (Traditional)",
}

# Normalized source-language aliases -> ipa-dict variant codes (ordered)
SOURCE_LANGUAGE_VARIANTS: dict[str, tuple[str, ...]] = {
    "english": ("en_US", "en_UK"),
    "american english": ("en_US", "en_UK"),
    "british english": ("en_UK", "en_US"),
    "french": ("fr_FR", "fr_QC"),
    "spanish": ("es_ES", "es_MX"),
    "german": ("de",),
    "dutch": ("nl",),
    "finnish": ("fi",),
    "japanese": ("ja",),
    "korean": ("ko",),
    "arabic": ("ar",),
    "persian": ("fa",),
    "farsi": ("fa",),
    "swedish": ("sv",),
    "norwegian": ("nb",),
    "norwegian bokmal": ("nb",),
    "norwegian bokmål": ("nb",),
    "romanian": ("ro",),
    "icelandic": ("is",),
    "khmer": ("km",),
    "swahili": ("sw",),
    "esperanto": ("eo",),
    "portuguese": ("pt_BR",),
    "brazilian portuguese": ("pt_BR",),
    "indonesian": ("ma",),
    "cantonese": ("yue",),
    "mandarin": ("zh_hans", "zh_hant"),
    "chinese": ("zh_hans", "zh_hant"),
    "simplified chinese": ("zh_hans", "zh_hant"),
    "traditional chinese": ("zh_hant", "zh_hans"),
    "vietnamese": ("vi_N", "vi_C", "vi_S"),
    "odia": ("or",),
    "jamaican creole": ("jam",),
    "isan": ("tts",),
}


@dataclass(frozen=True)
class IPALanguageInfo:
    ipa_supported: bool
    source_language: str
    variants: tuple[str, ...]
    default_variant: str | None


@dataclass(frozen=True)
class IPALookupResult:
    variant: str | None
    entries: dict[str, str | None]


def _normalize_language_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.strip().casefold())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def resolve_ipa_language(source_language: str) -> IPALanguageInfo:
    cleaned = source_language.strip()
    key = _normalize_language_key(cleaned)
    variants = SOURCE_LANGUAGE_VARIANTS.get(key, ())
    if not variants:
        # Partial match fallback (e.g. "Modern Standard Arabic")
        for alias, alias_variants in SOURCE_LANGUAGE_VARIANTS.items():
            if alias in key or key in alias:
                variants = alias_variants
                break
    return IPALanguageInfo(
        ipa_supported=bool(variants),
        source_language=cleaned,
        variants=variants,
        default_variant=variants[0] if variants else None,
    )


def variant_label(variant_code: str) -> str:
    return IPA_VARIANT_LABELS.get(variant_code, variant_code.replace("_", " "))


def _dict_file_for_variant(variant_code: str) -> Path:
    return IPA_DATA_DIR / f"{variant_code}.txt"


@lru_cache(maxsize=32)
def _load_dictionary(variant_code: str) -> dict[str, str]:
    path = _dict_file_for_variant(variant_code)
    if not path.exists():
        return {}

    entries: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            word, ipa = line.split("\t", 1)
            word_key = normalize_dict_key(word)
            if not word_key:
                continue
            # Keep first pronunciation when duplicates exist.
            entries.setdefault(word_key, ipa.strip())
    return entries


def normalize_display_text(value: str) -> str:
    return unicodedata.normalize("NFC", value or "")


def normalize_lookup_word(word: str) -> str:
    cleaned = normalize_display_text(word.strip())
    cleaned = re.sub(r"^[^\w']+|[^\w']+$", "", cleaned, flags=re.UNICODE)
    return cleaned.casefold()


def normalize_dict_key(word: str) -> str:
    return normalize_display_text(word.strip()).casefold()


def lookup_words(*, variant_code: str, words: list[str]) -> IPALookupResult:
    dictionary = _load_dictionary(variant_code)
    entries: dict[str, str | None] = {}
    for word in words:
        normalized = normalize_lookup_word(word)
        if not normalized:
            continue
        ipa = dictionary.get(normalized)
        entries[normalize_display_text(word)] = ipa
    return IPALookupResult(variant=variant_code, entries=entries)


def tokenize_source_text(text: str) -> list[str]:
    if not text.strip():
        return []
    tokens: list[str] = []
    for match in re.finditer(r"[\w']+", text, flags=re.UNICODE):
        token = match.group(0)
        if token:
            tokens.append(token)
    return tokens
