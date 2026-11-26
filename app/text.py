"""Playground text helpers (T-011, T-016): chips, ratios, language hints.

Pure functions only, so every rule the card asks for is unit-testable without
Streamlit. Nothing here calls a network service: the machine-translation gate is
off unless the operator passes an explicit live flag *and* a key. Tokenizer
backends live in :mod:`app.token_backends`; this module only renders what they give.
"""

from __future__ import annotations

import html
import os
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from app.token_backends import (
    WORD_SPLIT_ID,
    WORD_SPLIT_LABEL,
    Backend,
    word_split_pieces,
)
from src.features import KAZAKH_SPECIFIC, LANGS

UNAVAILABLE = "unavailable"
LIVE_ENV = "LT_LIVE"
KEY_ENV = "LT_TRANSLATE_KEY"
TRANSLATE_ENDPOINT_ENV = "LT_TRANSLATE_ENDPOINT"
TRANSLATE_LABEL = "machine translation, not verified"
LANG_COLORS: dict[str, str] = {"kk": "#00AFB9", "ru": "#E8A33D", "en": "#4C6EF5"}
BYTE_CHIP_COLOR = "#6E7781"
BYTE_CAPTION = "byte-level BPE splits one letter into several tokens"
BYTE_NOTE = "bytes of a letter split across tokens, not a complete UTF-8 character"


@dataclass(frozen=True)
class Chip:
    """One token chip: position, decoded text, model token id, language colour.

    ``raw`` is set only when the token's bytes are not a complete UTF-8 character.
    """

    index: int
    text: str
    lang: str
    tokenizer: str
    token_id: int | None = None
    raw: bytes | None = None

    @property
    def label(self) -> str:
        """Visible chip text: the decoded piece, or lowercase hex bytes."""
        if self.raw is None:
            return self.text
        return " ".join(f"{byte:02x}" for byte in self.raw)

    @property
    def title(self) -> str:
        """Hover text: the model token id, or the diagnostic label without one."""
        if self.token_id is None:
            return f"position {self.index} | {WORD_SPLIT_LABEL}"
        title = f"position {self.index} | token id {self.token_id} | {self.tokenizer}"
        return title if self.raw is None else f"{title} | {BYTE_NOTE}"


def partial_bytes(backend: Backend | None, token_id: int | None) -> bytes | None:
    """Raw bytes of one token when they do not decode as UTF-8, else ``None``."""
    if backend is None or backend.token_bytes is None or token_id is None:
        return None
    raw = backend.token_bytes(token_id)
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
    return None


def tokenize(
    text: str, backend: Backend | None = None, diagnostic: bool = False
) -> list[tuple[int | None, str]]:
    """Token ids with their decoded strings, or positional words in diagnostic mode.

    Without a backend and without the diagnostic switch the result is empty: the
    page must never present word counts as token counts.
    """
    if backend is not None:
        return [(token_id, piece) for token_id, piece in backend.pieces(text)]
    if not diagnostic:
        return []
    return [(None, piece) for _, piece in word_split_pieces(text)]


def token_chips(
    text: str, lang: str, backend: Backend | None = None, diagnostic: bool = False
) -> list[Chip]:
    """Chips for one phrase, numbered in reading order."""
    if lang not in LANGS:
        raise ValueError(f"unsupported language: {lang}")
    identifier = WORD_SPLIT_ID if backend is None else backend.identifier
    return [
        Chip(
            index=index,
            text=piece,
            lang=lang,
            tokenizer=identifier,
            token_id=token_id,
            raw=partial_bytes(backend, token_id) if "�" in piece else None,
        )
        for index, (token_id, piece) in enumerate(tokenize(text, backend, diagnostic))
    ]


def chips_html(chips: Sequence[Chip]) -> str:
    """Render chips as coloured spans with the token id on hover; bytes muted."""
    if not chips:
        # A span, not a <p>: Streamlit's sanitizer keeps span markup verbatim.
        return (
            f'<div class="chips"><span class="chip-empty">{UNAVAILABLE}: '
            f"nothing tokenized</span></div>"
        )
    parts = []
    for chip in chips:
        color = BYTE_CHIP_COLOR if chip.raw else LANG_COLORS.get(chip.lang, "#868E96")
        kind = "chip chip-bytes" if chip.raw else "chip"
        parts.append(
            f'<span class="{kind}" title="{html.escape(chip.title)}" '
            f'style="border-color:{color};color:{color}">'
            f"{html.escape(chip.label)}</span>"
        )
    return f'<div class="chips">{"".join(parts)}</div>'


def format_multiplier(value: float | None) -> str:
    """tokens_vs_en as the multiplier the summary cards show."""
    return UNAVAILABLE if value is None else f"x{value:.2f}"


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
