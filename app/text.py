"""Playground text helpers (T-011): tokenizers, chips, ratios, language hints.

Pure functions only, so every rule the card asks for is unit-testable without
Streamlit. Nothing here calls a network service: the machine-translation gate is
off unless the operator passes an explicit live flag *and* a key.
"""

from __future__ import annotations

import html
import os
import re
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from src.features import KAZAKH_SPECIFIC, LANGS

FALLBACK_TOKENIZER = "whitespace-fallback"
UNAVAILABLE = "unavailable"
LIVE_ENV = "LT_LIVE"
KEY_ENV = "LT_TRANSLATE_KEY"
TRANSLATE_ENDPOINT_ENV = "LT_TRANSLATE_ENDPOINT"
TRANSLATE_LABEL = "machine translation, not verified"
LANG_COLORS: dict[str, str] = {"kk": "#00AFB9", "ru": "#E8A33D", "en": "#4C6EF5"}
PIECES = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Chip:
    """One token chip: the text, a positional id and the language colour."""

    index: int
    text: str
    lang: str
    tokenizer: str

    @property
    def title(self) -> str:
        """Hover text with the id and the tokenizer that produced it."""
        return f"id {self.index} | {self.tokenizer}"


def local_tokenizers() -> dict[str, Callable[[str], list[str]]]:
    """Tokenizers published by ``src.tokens`` (T-001) when it is usable offline.

    Expected hook: ``TOKENIZER_IDS`` and ``encode_pieces(text, tokenizer_id)``.
    Until T-001 exposes them the registry is empty and the labelled whitespace
    fallback is the only backend, so no chip is ever presented as a model token.
    """
    try:
        from src import tokens as tokens_module
    except ImportError:
        return {}
    identifiers = getattr(tokens_module, "TOKENIZER_IDS", ())
    encode = getattr(tokens_module, "encode_pieces", None)
    if not identifiers or encode is None:
        return {}
    return {
        str(name): (lambda text, name=str(name): list(encode(text, name)))
        for name in identifiers
    }


def available_backends() -> list[str]:
    """Every tokenizer the Playground can use right now, fallback last."""
    return [*sorted(local_tokenizers()), FALLBACK_TOKENIZER]


def tokenize(
    text: str,
    tokenizer: str = FALLBACK_TOKENIZER,
    backends: Mapping[str, Callable[[str], list[str]]] | None = None,
) -> list[str]:
    """Split one phrase into pieces with the selected backend.

    ``backends`` lets the caller pass a registry it already built and cached, so
    a rerun never rediscovers the local tokenizers.
    """
    if not text.strip():
        return []
    registry = local_tokenizers() if backends is None else backends
    backend = registry.get(tokenizer)
    if backend is not None:
        return [str(piece) for piece in backend(text)]
    if tokenizer != FALLBACK_TOKENIZER:
        raise ValueError(f"tokenizer is not available offline: {tokenizer}")
    return PIECES.findall(text)


def token_chips(
    text: str,
    lang: str,
    tokenizer: str = FALLBACK_TOKENIZER,
    backends: Mapping[str, Callable[[str], list[str]]] | None = None,
) -> list[Chip]:
    """Chips for one phrase, numbered in reading order."""
    if lang not in LANGS:
        raise ValueError(f"unsupported language: {lang}")
    pieces = tokenize(text, tokenizer, backends)
    return [
        Chip(index=index, text=piece, lang=lang, tokenizer=tokenizer)
        for index, piece in enumerate(pieces)
    ]


def chips_html(chips: Sequence[Chip]) -> str:
    """Render chips as coloured spans with the token id on hover."""
    if not chips:
        return f"<p><em>{UNAVAILABLE}: empty phrase</em></p>"
    parts = []
    for chip in chips:
        color = LANG_COLORS.get(chip.lang, "#868E96")
        parts.append(
            f'<span title="{html.escape(chip.title)}" style="display:inline-block;'
            f"margin:2px;padding:2px 6px;border-radius:10px;color:white;"
            f'background:{color}">{html.escape(chip.text)}</span>'
        )
    return f'<div style="line-height:2.4">{"".join(parts)}</div>'


def kazakh_letter_share(text: str) -> float:
    """Share of Kazakh-specific letters (ә ғ қ ң ө ұ ү һ і) among all letters."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(char in KAZAKH_SPECIFIC for char in letters) / len(letters)


def cyrillic_letter_share(text: str) -> float:
    """Share of Cyrillic letters, the other half of the language hint."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum("\u0400" <= char <= "\u04ff" for char in letters) / len(letters)


def detect_lang(text: str) -> str:
    """Guess the language of a user phrase: a heuristic, never a measurement."""
    if not text.strip():
        return "kk"
    if cyrillic_letter_share(text) < 0.5:
        return "en"
    return "kk" if kazakh_letter_share(text) > 0.0 else "ru"


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Ratio of two counts, or None when a denominator is missing or zero."""
    if numerator is None or denominator is None:
        return None
    if not denominator:
        return None
    return float(numerator) / float(denominator)


def format_ratio(value: float | None, digits: int = 3) -> str:
    """Human text for a ratio, spelling out the unavailable case."""
    return UNAVAILABLE if value is None else f"{value:.{digits}f}"


def fertility(token_count: int | None, word_count: int | None) -> float | None:
    """Diagnostic tokens per word; unavailable without both counts."""
    return ratio(token_count, word_count)


def tokens_vs_en(counts: Mapping[str, int | None]) -> dict[str, float | None]:
    """Token count of each language divided by the English count."""
    denominator = counts.get("en")
    return {lang: ratio(counts.get(lang), denominator) for lang in LANGS}


def word_count(text: str) -> int:
    """Whitespace-delimited words, the contract definition."""
    return len(text.split())


def live_translation_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Both gates: an explicit live flag and a key present in the environment."""
    env = dict(os.environ if environ is None else environ)
    return env.get(LIVE_ENV, "").lower() in {"1", "true", "yes"} and bool(
        env.get(KEY_ENV)
    )


def translate_text(
    text: str,
    target: str,
    environ: Mapping[str, str] | None = None,
    opener: Callable[[str, str, str], str] | None = None,
) -> str:
    """Translate one phrase; refused unless both live gates are open.

    ``opener`` exists so tests can mock the call: the default implementation is
    the only place that touches the network, and only with an operator-supplied
    endpoint and key. The result is always labelled machine translation.
    """
    env = dict(os.environ if environ is None else environ)
    if not live_translation_enabled(env):
        raise PermissionError("live translation needs the live flag and a key")
    if target not in LANGS:
        raise ValueError(f"unsupported target language: {target}")
    if not text.strip():
        raise ValueError("nothing to translate")
    if opener is not None:
        return opener(text, target, env[KEY_ENV])
    endpoint = env.get(TRANSLATE_ENDPOINT_ENV, "")
    if not endpoint:
        raise PermissionError(f"{TRANSLATE_ENDPOINT_ENV} is not configured")
    request = urllib.request.Request(
        endpoint,
        data=f'{{"text": {text!r}, "target": "{target}"}}'.encode(),
        headers={
            "Authorization": f"Bearer {env[KEY_ENV]}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8").strip()
