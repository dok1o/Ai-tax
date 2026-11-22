"""Pure-function tests for the Playground helpers (T-011).

Chips, ratios and the Kazakh-letter heuristic are the parts of the page a
reviewer can check by hand, so they are covered without Streamlit.
"""

from __future__ import annotations

import pytest

from app import text
from src.features import KAZAKH_SPECIFIC

KAZAKH = "Қазақстанның ең биік шыңы қалай аталады?"
RUSSIAN = "Как называется самая высокая вершина Казахстана?"
ENGLISH = "What is the highest peak of Kazakhstan?"


def test_tokenize_falls_back_to_the_labelled_backend() -> None:
    pieces = text.tokenize(KAZAKH)
    assert pieces[:2] == ["Қазақстанның", "ең"]
    assert pieces[-1] == "?"
    assert text.tokenize("   ") == []
    assert text.available_backends()[-1] == text.FALLBACK_TOKENIZER


def test_tokenize_refuses_a_backend_that_is_not_installed() -> None:
    with pytest.raises(ValueError, match="not available offline"):
        text.tokenize(KAZAKH, "tiktoken/cl100k_base")


def test_chips_carry_text_ids_language_and_tokenizer() -> None:
    chips = text.token_chips(KAZAKH, "kk", "whitespace-fallback")
    assert [chip.index for chip in chips] == list(range(len(chips)))
    assert [chip.text for chip in chips] == text.tokenize(KAZAKH)
    assert {chip.lang for chip in chips} == {"kk"}
    assert chips[3].title == "id 3 | whitespace-fallback"
    with pytest.raises(ValueError, match="unsupported language"):
        text.token_chips(KAZAKH, "de")


def test_chips_html_escapes_text_and_puts_the_id_on_hover() -> None:
    chips = text.token_chips("<ү> & сұрақ", "kk")
    markup = text.chips_html(chips)
    assert "&lt;" in markup and "&amp;" in markup
    assert "<ү>" not in markup
    assert 'title="id 0 | whitespace-fallback"' in markup
    assert text.LANG_COLORS["kk"] in markup
    assert text.UNAVAILABLE in text.chips_html([])


def test_ratios_report_unavailable_denominators() -> None:
    assert text.ratio(30, 15) == 2.0
    assert text.ratio(30, 0) is None
    assert text.ratio(30, None) is None
    assert text.ratio(None, 15) is None
    assert text.format_ratio(None) == text.UNAVAILABLE
    assert text.format_ratio(1.23456) == "1.235"


def test_tokens_vs_en_uses_the_english_count_only() -> None:
    ratios = text.tokens_vs_en({"kk": 12, "ru": 9, "en": 6})
    assert ratios == {"kk": 2.0, "ru": 1.5, "en": 1.0}
    assert text.tokens_vs_en({"kk": 12, "ru": 9, "en": 0}) == {
        "kk": None,
        "ru": None,
        "en": None,
    }
    assert text.tokens_vs_en({"kk": 12, "ru": 9, "en": None})["kk"] is None
    assert text.tokens_vs_en({"kk": None, "ru": 9, "en": 3})["kk"] is None


def test_fertility_is_tokens_per_word_and_never_invents_a_value() -> None:
    assert text.fertility(12, 10) == pytest.approx(1.2)
    assert text.fertility(12, 0) is None
    assert text.fertility(None, 10) is None
    assert text.word_count(KAZAKH) == len(KAZAKH.split())


def test_kazakh_letter_heuristic_separates_the_scripts() -> None:
    assert text.kazakh_letter_share(KAZAKH) > 0.0
    assert text.kazakh_letter_share(RUSSIAN) == 0.0
    assert text.kazakh_letter_share(ENGLISH) == 0.0
    assert text.kazakh_letter_share("123 ?!") == 0.0
    assert text.kazakh_letter_share("ә") == 1.0
    assert text.cyrillic_letter_share(KAZAKH) == 1.0
    assert text.cyrillic_letter_share(ENGLISH) == 0.0
    assert KAZAKH_SPECIFIC & set("Қазақстан") == {"Қ", "қ"}
    assert not KAZAKH_SPECIFIC & set("Казахстан")


def test_language_hint_follows_the_letters_not_the_topic() -> None:
    assert text.detect_lang(KAZAKH) == "kk"
    assert text.detect_lang(RUSSIAN) == "ru"
    assert text.detect_lang(ENGLISH) == "en"
    assert text.detect_lang("") == "kk"


def test_live_translation_needs_both_gates() -> None:
    calls: list[tuple[str, str, str]] = []

    def opener(phrase: str, target: str, key: str) -> str:
        calls.append((phrase, target, key))
        return f"mock {target}"

    both = {text.LIVE_ENV: "1", text.KEY_ENV: "secret"}
    assert text.live_translation_enabled(both)
    assert text.live_translation_enabled({text.LIVE_ENV: "true", text.KEY_ENV: "k"})
    assert not text.live_translation_enabled({text.LIVE_ENV: "1"})
    assert not text.live_translation_enabled({text.KEY_ENV: "secret"})
    assert not text.live_translation_enabled({})
    assert not text.live_translation_enabled({text.LIVE_ENV: "0", text.KEY_ENV: "k"})

    for environ in ({}, {text.LIVE_ENV: "1"}, {text.KEY_ENV: "secret"}):
        with pytest.raises(PermissionError, match="live flag and a key"):
            text.translate_text(KAZAKH, "en", environ, opener)
    assert calls == []

    assert text.translate_text(KAZAKH, "en", both, opener) == "mock en"
    assert calls == [(KAZAKH, "en", "secret")]
    with pytest.raises(ValueError, match="unsupported target"):
        text.translate_text(KAZAKH, "de", both, opener)
    with pytest.raises(ValueError, match="nothing to translate"):
        text.translate_text("  ", "en", both, opener)


def test_default_translation_calls_the_network_only_with_an_endpoint() -> None:
    environ = {text.LIVE_ENV: "1", text.KEY_ENV: "secret"}
    with pytest.raises(PermissionError, match=text.TRANSLATE_ENDPOINT_ENV):
        text.translate_text(KAZAKH, "en", environ)
    assert text.TRANSLATE_LABEL == "machine translation, not verified"
