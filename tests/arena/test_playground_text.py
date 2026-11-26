"""Pure-function tests for the Playground helpers (T-011).

Chips, ratios and the Kazakh-letter heuristic are the parts of the page a
reviewer can check by hand, so they are covered without Streamlit.
"""

from __future__ import annotations

import pytest

from app import text, tokenizers
from src.features import KAZAKH_SPECIFIC

KAZAKH = "Қазақстанның ең биік шыңы қалай аталады?"
RUSSIAN = "Как называется самая высокая вершина Казахстана?"
ENGLISH = "What is the highest peak of Kazakhstan?"


def fake_backend() -> tokenizers.Backend:
    """A backend that needs no network: one id per whitespace word."""

    def encode(phrase: str) -> list[tuple[int, str]]:
        return [(700 + index, piece) for index, piece in enumerate(phrase.split())]

    return tokenizers.Backend(identifier="fake/cl100k", encode=encode)


def test_tokenize_without_a_backend_yields_nothing() -> None:
    assert text.tokenize(KAZAKH) == []
    assert text.tokenize(KAZAKH, diagnostic=True)[0] == (None, "Қазақстанның")
    assert text.tokenize("   ", diagnostic=True) == []


def test_chips_from_a_real_backend_carry_model_token_ids() -> None:
    backend = fake_backend()
    chips = text.token_chips(KAZAKH, "kk", backend)
    assert [chip.index for chip in chips] == list(range(len(chips)))
    assert [chip.text for chip in chips] == KAZAKH.split()
    assert [chip.token_id for chip in chips] == list(range(700, 700 + len(chips)))
    assert {chip.lang for chip in chips} == {"kk"}
    assert chips[2].tokenizer == "fake/cl100k"
    assert chips[2].title == "position 2 | token id 702 | fake/cl100k"
    with pytest.raises(ValueError, match="unsupported language"):
        text.token_chips(KAZAKH, "de", backend)


def test_diagnostic_chips_are_labelled_as_words_not_tokens() -> None:
    chips = text.token_chips(KAZAKH, "kk", diagnostic=True)
    assert [chip.token_id for chip in chips] == [None] * len(chips)
    assert chips[0].tokenizer == tokenizers.WORD_SPLIT_ID
    assert chips[0].title == f"position 0 | {tokenizers.WORD_SPLIT_LABEL}"
    assert tokenizers.WORD_SPLIT_LABEL == "words, not tokens"


def test_chips_html_escapes_text_and_puts_the_id_on_hover() -> None:
    markup = text.chips_html(text.token_chips("<ү> & сұрақ", "kk", fake_backend()))
    assert "&lt;" in markup and "&amp;" in markup
    assert "<ү>" not in markup
    assert 'title="position 0 | token id 700 | fake/cl100k"' in markup
    assert text.LANG_COLORS["kk"] in markup
    assert text.UNAVAILABLE in text.chips_html([])


def test_multiplier_formatting_matches_the_summary_cards() -> None:
    assert text.format_multiplier(1.3691) == "x1.37"
    assert text.format_multiplier(1.0) == "x1.00"
    assert text.format_multiplier(None) == text.UNAVAILABLE


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
