from __future__ import annotations

from app.ipa_lookup import lookup_words, resolve_ipa_language, tokenize_source_text, variant_label


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


def test_tokenize_source_text():
    assert tokenize_source_text("Kyrie eleison.") == ["Kyrie", "eleison"]


def test_variant_label():
    assert variant_label("en_UK") == "Received Pronunciation"
