from __future__ import annotations

import pytest

from app.segmentation import segment


def test_segment_default_preserves_whitespace_and_punctuation():
    text = "Guten Tag, Welt!"
    tokens = segment(variant_code="de", text=text)
    assert "".join(token.text for token in tokens) == text
    assert [token.text for token in tokens if token.is_word] == ["Guten", "Tag", "Welt"]
    assert any(token.text == " " for token in tokens)
    assert any(token.text == "," for token in tokens)


def test_segment_default_round_trip():
    text = "Kyrie eleison.\nDomine!"
    tokens = segment(variant_code="en_US", text=text)
    assert "".join(token.text for token in tokens) == text


def test_segment_japanese_poem_line():
    pytest.importorskip("sudachipy")
    text = "今日も薊の紫に"
    tokens = segment(variant_code="ja", text=text)
    assert "".join(token.text for token in tokens) == text
    assert [token.text for token in tokens if token.is_word] == [
        "今日",
        "も",
        "薊",
        "の",
        "紫",
        "に",
    ]


def test_segment_japanese_lookup_keys_include_lemma_and_normalized_form():
    pytest.importorskip("sudachipy")
    text = "光れ"
    tokens = segment(variant_code="ja", text=text)
    assert len(tokens) == 1
    assert tokens[0].text == "光れ"
    assert "光る" in tokens[0].lookup_keys

    text = "ひとり"
    tokens = segment(variant_code="ja", text=text)
    assert tokens[0].text == "ひとり"
    assert "一人" in tokens[0].lookup_keys


def test_segment_chinese_round_trip():
    pytest.importorskip("jieba")
    text = "今天天气很好"
    tokens = segment(variant_code="zh_hans", text=text)
    assert "".join(token.text for token in tokens) == text
    assert any(token.is_word for token in tokens)
