from __future__ import annotations

import pytest

from app.ipa_lookup import (
    lookup_tokens,
    lookup_words,
    resolve_ipa_language,
    tokenize_source_text,
    variant_label,
)


def test_resolve_ipa_language_english_variants():
    info = resolve_ipa_language("English")
    assert info.ipa_supported is True
    assert info.variants == ("en_US", "en_UK")
    assert info.default_variant == "en_US"


def test_resolve_ipa_language_unsupported():
    info = resolve_ipa_language("Latin")
    assert info.ipa_supported is False
    assert info.variants == ()


def test_lookup_words_from_sample_dictionary(tmp_path, monkeypatch):
    data_dir = tmp_path / "ipa-dict"
    data_dir.mkdir()
    (data_dir / "en_US.txt").write_text("hello\t/həˈloʊ/\nworld\t/wɝld/\n", encoding="utf-8")

    import app.ipa_lookup as ipa_module

    monkeypatch.setattr(ipa_module, "IPA_DATA_DIR", data_dir)
    ipa_module._load_dictionary.cache_clear()

    result = lookup_words(variant_code="en_US", words=["Hello", "world", "missing"])
    assert result.entries["Hello"] == "/həˈloʊ/"
    assert result.entries["world"] == "/wɝld/"
    assert result.entries["missing"] is None


def test_lookup_tokens_surface_hit(tmp_path, monkeypatch):
    data_dir = tmp_path / "ipa-dict"
    data_dir.mkdir()
    (data_dir / "de.txt").write_text("guten\t/ˈɡuːtən/\n", encoding="utf-8")

    import app.ipa_lookup as ipa_module

    monkeypatch.setattr(ipa_module, "IPA_DATA_DIR", data_dir)
    ipa_module._load_dictionary.cache_clear()

    result = lookup_tokens(variant_code="de", text="Guten")
    assert len(result.tokens) == 1
    assert result.tokens[0].text == "Guten"
    assert result.tokens[0].ipa == "/ˈɡuːtən/"
    assert result.tokens[0].matched is None


def test_lookup_tokens_lemma_fallback(tmp_path, monkeypatch):
    data_dir = tmp_path / "ipa-dict"
    data_dir.mkdir()
    (data_dir / "ja.txt").write_text("光る\t/çikaɾɯ/\n", encoding="utf-8")

    import app.ipa_lookup as ipa_module

    monkeypatch.setattr(ipa_module, "IPA_DATA_DIR", data_dir)
    ipa_module._load_dictionary.cache_clear()

    pytest.importorskip("sudachipy")
    result = lookup_tokens(variant_code="ja", text="光れ")
    assert len(result.tokens) == 1
    assert result.tokens[0].text == "光れ"
    assert result.tokens[0].ipa == "/çikaɾɯ/"
    assert result.tokens[0].matched == "光る"


def test_lookup_tokens_per_character_fallback(tmp_path, monkeypatch):
    data_dir = tmp_path / "ipa-dict"
    data_dir.mkdir()
    (data_dir / "zh_hans.txt").write_text(
        "今\t/tɕɪn˥˥/\n天\t/tʰjɛn˥˥/\n",
        encoding="utf-8",
    )

    import app.ipa_lookup as ipa_module

    monkeypatch.setattr(ipa_module, "IPA_DATA_DIR", data_dir)
    ipa_module._load_dictionary.cache_clear()

    pytest.importorskip("jieba")
    result = lookup_tokens(variant_code="zh_hans", text="今天")
    joined = "".join(token.text for token in result.tokens)
    assert joined == "今天"
    word_tokens = [token for token in result.tokens if token.ipa]
    assert any(token.ipa == "/tɕɪn˥˥/ /tʰjɛn˥˥/" for token in word_tokens)


def test_lookup_tokens_miss(tmp_path, monkeypatch):
    data_dir = tmp_path / "ipa-dict"
    data_dir.mkdir()
    (data_dir / "de.txt").write_text("guten\t/ˈɡuːtən/\n", encoding="utf-8")

    import app.ipa_lookup as ipa_module

    monkeypatch.setattr(ipa_module, "IPA_DATA_DIR", data_dir)
    ipa_module._load_dictionary.cache_clear()

    result = lookup_tokens(variant_code="de", text="missing")
    assert len(result.tokens) == 1
    assert result.tokens[0].ipa is None


def test_tokenize_source_text():
    assert tokenize_source_text("Kyrie eleison.") == ["Kyrie", "eleison"]


def test_variant_label():
    assert variant_label("en_UK") == "Received Pronunciation"
