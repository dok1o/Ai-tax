"""Pre-tokenization features for the token-cost predictor (T-005).

Every feature is computable from the prompt text and prompt metadata before
any tokenizer runs. The measured ``token_count`` is the prediction target and
``fertility`` divides it by ``word_count``, so both are leaks and are rejected
by :func:`assert_no_leakage`. ``word_count`` here is recomputed from the text
with the contract definition instead of being read from tokens.csv.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import pstdev

import pandas as pd

from src.common.config import PROMPTS_PATH, TOKENS_PATH

LANGS: tuple[str, ...] = ("kk", "ru", "en")
KAZAKH_SPECIFIC_CHARS = frozenset("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
PUNCTUATION_CHARS = frozenset(".,;:!?()[]{}\"'‘’“”-–—…/\\@#$%&*+=<>_|~`^")

PROMPT_COLUMNS: tuple[str, ...] = (
    "prompt_id",
    "q_id",
    "lang",
    "category",
    "prompt",
    "answer_type",
)
TOKEN_COLUMNS: tuple[str, ...] = (
    "prompt_id",
    "q_id",
    "lang",
    "tokenizer",
    "word_count",
    "token_count",
)
TARGET_COLUMN = "token_count"
KEY_COLUMNS: tuple[str, ...] = ("prompt_id", "q_id", "lang", "tokenizer")

TEXT_FEATURES: tuple[str, ...] = (
    "char_count",
    "word_count",
    "mean_word_chars",
    "max_word_chars",
    "std_word_chars",
    "unique_char_ratio",
    "digit_ratio",
    "punct_ratio",
    "space_ratio",
    "letter_count",
    "cyrillic_ratio",
    "latin_ratio",
    "kazakh_specific_ratio",
)
META_FEATURES: tuple[str, ...] = ("lang", "category", "answer_type")
LEAKED_COLUMNS: frozenset[str] = frozenset({TARGET_COLUMN, "fertility"})


def assert_no_leakage(columns: Iterable[str]) -> None:
    """Raise when measured tokenizer outputs appear among model inputs."""
    leaked = sorted(LEAKED_COLUMNS.intersection(columns))
    if leaked:
        raise ValueError(f"measured columns must not be model inputs: {leaked}")


def _is_cyrillic(char: str) -> bool:
    return "\u0400" <= char <= "\u04ff"


def text_features(text: str) -> dict[str, float]:
    """Describe one prompt string without running a tokenizer."""
    total = len(text)
    if total == 0:
        raise ValueError("prompt text is empty")
    words = text.split()
    if not words:
        raise ValueError("prompt text has no whitespace-delimited words")
    lengths = [len(word) for word in words]
    letters = [char for char in text if char.isalpha()]
    letter_total = len(letters) or 1
    stripped = "".join(char for char in text if not char.isspace())
    return {
        "char_count": float(total),
        "word_count": float(len(words)),
        "mean_word_chars": sum(lengths) / len(lengths),
        "max_word_chars": float(max(lengths)),
        "std_word_chars": float(pstdev(lengths)) if len(lengths) > 1 else 0.0,
        "unique_char_ratio": len(set(stripped)) / max(len(stripped), 1),
        "digit_ratio": sum(char.isdigit() for char in text) / total,
        "punct_ratio": sum(char in PUNCTUATION_CHARS for char in text) / total,
        "space_ratio": sum(char.isspace() for char in text) / total,
        "letter_count": float(len(letters)),
        "cyrillic_ratio": sum(_is_cyrillic(char) for char in letters) / letter_total,
        "latin_ratio": sum(char.isascii() and char.isalpha() for char in letters)
        / letter_total,
        "kazakh_specific_ratio": sum(char in KAZAKH_SPECIFIC_CHARS for char in letters)
        / letter_total,
    }


def build_features(prompts: pd.DataFrame) -> pd.DataFrame:
    """Return one feature row per prompt, keyed by ``prompt_id``."""
    missing = [column for column in PROMPT_COLUMNS if column not in prompts.columns]
    if missing:
        raise ValueError(f"prompts.csv is missing columns: {missing}")
    unknown_langs = sorted(set(prompts["lang"].astype(str)) - set(LANGS))
    if unknown_langs:
        raise ValueError(f"prompts.csv has unsupported lang values: {unknown_langs}")
    frame = prompts[list(PROMPT_COLUMNS)].copy()
    frame["prompt_id"] = frame["prompt_id"].astype(str)
    if frame["prompt_id"].duplicated().any():
        raise ValueError("prompts.csv has duplicated prompt_id values")
    expanded = pd.DataFrame(
        [text_features(str(text)) for text in frame["prompt"]], index=frame.index
    )
    return pd.concat([frame.drop(columns=["prompt"]), expanded], axis=1)


def design_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Build the float design matrix shared by every compared model.

    Categorical metadata is one-hot encoded over the sorted levels present in
    ``features`` so linear regression and gradient boosting see identical
    columns. Measured tokenizer outputs never enter the matrix.
    """
    required = (*TEXT_FEATURES, *META_FEATURES)
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise ValueError(f"features are missing columns: {missing}")
    numeric = features[list(TEXT_FEATURES)].astype(float)
    encoded: dict[str, list[float]] = {}
    for column in META_FEATURES:
        values = features[column].astype(str)
        for level in sorted(values.unique()):
            encoded[f"{column}={level}"] = (values == level).astype(float).tolist()
    matrix = pd.concat([numeric.reset_index(drop=True), pd.DataFrame(encoded)], axis=1)
    assert_no_leakage(matrix.columns)
    return matrix


def load_dataset(
    prompts_path: Path = PROMPTS_PATH, tokens_path: Path = TOKENS_PATH
) -> pd.DataFrame:
    """Join contract v1 prompts with measured token counts per tokenizer."""
    for path in (prompts_path, tokens_path):
        if not path.exists():
            raise FileNotFoundError(f"required input is missing: {path}")
    prompts = pd.read_csv(prompts_path, dtype=str, keep_default_na=False)
    tokens = pd.read_csv(tokens_path, dtype=str, keep_default_na=False)
    for name, frame, required in (
        ("prompts.csv", prompts, PROMPT_COLUMNS),
        ("tokens.csv", tokens, TOKEN_COLUMNS),
    ):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
    measured = tokens[["prompt_id", "q_id", "lang", "tokenizer", TARGET_COLUMN]].copy()
    measured[TARGET_COLUMN] = measured[TARGET_COLUMN].astype(int)
    if (measured[TARGET_COLUMN] <= 0).any():
        raise ValueError("tokens.csv has non-positive token_count values")
    dataset = measured.merge(
        build_features(prompts), on="prompt_id", suffixes=("", "_p")
    )
    for column in ("q_id", "lang"):
        mismatched = dataset[dataset[column] != dataset[f"{column}_p"]]
        if not mismatched.empty:
            ids = sorted(set(mismatched["prompt_id"]))
            raise ValueError(
                f"tokens.csv disagrees with prompts.csv on {column}: {ids}"
            )
    dataset = dataset.drop(columns=["q_id_p", "lang_p"])
    if dataset.empty:
        raise ValueError("no prompt has a measured token count")
    return dataset.reset_index(drop=True)


def fold_frame(dataset: pd.DataFrame, folds: Sequence[int]) -> pd.DataFrame:
    """Return the question-level fold assignment, asserting questions stay whole."""
    rows = pd.DataFrame({"q_id": dataset["q_id"].to_numpy(), "fold": list(folds)})
    spread = rows.groupby("q_id")["fold"].nunique()
    broken = sorted(spread[spread > 1].index)
    if broken:
        raise ValueError(f"q_id split across folds: {broken}")
    return (
        rows.drop_duplicates(subset="q_id").sort_values("q_id").reset_index(drop=True)
    )
