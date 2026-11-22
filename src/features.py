"""Feature building and contract-shaped input loading for the ARENA tasks.

Every feature is computable before a tokenizer or a model runs: prompt text
statistics, language, category and answer type. Measured ``token_count`` and
``fertility`` are targets or leaks and are rejected by :func:`assert_no_leakage`.

Standard parts: one-hot encoding, scikit-learn estimators, LightGBM, SHAP.
Ours: the trilingual pre-tokenization feature set, the router formulation and
the cost-quality evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from src.common.config import (
    DATA_DIR,
    LABELS_PATH,
    MODELS_DIR,
    PROMPTS_PATH,
    TOKENS_PATH,
)

LANGS: tuple[str, ...] = ("kk", "ru", "en")
KAZAKH_SPECIFIC = frozenset("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
PUNCTUATION = frozenset(".,;:!?()[]{}\"'‘’“”-–—…/\\@#$%&*+=<>_|~`^")

TEXT_FEATURES: tuple[str, ...] = (
    "n_chars",
    "n_words",
    "mean_word_chars",
    "max_word_chars",
    "unique_char_ratio",
    "digit_share",
    "punct_share",
    "space_share",
    "cyrillic_share",
    "latin_share",
    "kk_share",
)
META_FEATURES: tuple[str, ...] = ("lang", "category")
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
LABEL_COLUMNS: tuple[str, ...] = (
    "model",
    "prompt_id",
    "lang",
    "rep",
    "q_id",
    "is_correct",
)
LEAK_PREFIXES: tuple[str, ...] = ("token_count", "fertility")
TARGET = "token_count"
FIXTURE_DIR = MODELS_DIR / "fixtures"
ROUTES_PATH = DATA_DIR / "routes.csv"


def assert_no_leakage(columns: Iterable[str]) -> None:
    """Raise when measured tokenizer outputs appear among model inputs."""
    leaked = sorted(
        column
        for column in columns
        if any(str(column).startswith(prefix) for prefix in LEAK_PREFIXES)
    )
    if leaked:
        raise ValueError(
            f"measured columns must not be model inputs: {', '.join(leaked)}"
        )


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
        "n_chars": float(total),
        "n_words": float(len(words)),
        "mean_word_chars": sum(lengths) / len(lengths),
        "max_word_chars": float(max(lengths)),
        "unique_char_ratio": len(set(stripped)) / max(len(stripped), 1),
        "digit_share": sum(char.isdigit() for char in text) / total,
        "punct_share": sum(char in PUNCTUATION for char in text) / total,
        "space_share": sum(char.isspace() for char in text) / total,
        "cyrillic_share": sum(_is_cyrillic(char) for char in letters) / letter_total,
        "latin_share": sum(char.isascii() and char.isalpha() for char in letters)
        / letter_total,
        "kk_share": sum(char in KAZAKH_SPECIFIC for char in letters) / letter_total,
    }


def build_features(prompts: pd.DataFrame) -> pd.DataFrame:
    """Return one feature row per prompt, keyed by ``prompt_id``."""
    missing = [column for column in PROMPT_COLUMNS if column not in prompts.columns]
    if missing:
        raise ValueError(f"prompts.csv is missing columns: {missing}")
    frame = prompts[list(PROMPT_COLUMNS)].copy()
    frame["prompt_id"] = frame["prompt_id"].astype(str)
    unknown = sorted(set(frame["lang"]) - set(LANGS))
    if unknown:
        raise ValueError(f"prompts.csv has unsupported lang values: {unknown}")
    if frame["prompt_id"].duplicated().any():
        raise ValueError("prompts.csv has duplicated prompt_id values")
    expanded = pd.DataFrame(
        [text_features(str(text)) for text in frame["prompt"]], index=frame.index
    )
    return pd.concat([frame.drop(columns=["prompt"]), expanded], axis=1)


def design_matrix(
    features: pd.DataFrame, meta_columns: Sequence[str] = META_FEATURES
) -> pd.DataFrame:
    """Build the float matrix every compared model sees, one-hot metadata included."""
    required = (*TEXT_FEATURES, *meta_columns)
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise ValueError(f"features are missing columns: {missing}")
    numeric = features[list(TEXT_FEATURES)].astype(float).reset_index(drop=True)
    encoded: dict[str, list[float]] = {}
    for column in meta_columns:
        values = features[column].astype(str).reset_index(drop=True)
        for level in sorted(values.unique()):
            encoded[f"{column}={level}"] = (values == level).astype(float).tolist()
    matrix = pd.concat([numeric, pd.DataFrame(encoded)], axis=1)
    assert_no_leakage(matrix.columns)
    return matrix


def artifact_name(base: str, suffix: str, is_fixture: bool) -> str:
    """Name a generated file so fixture-derived output is labelled in the name."""
    return f"{base}_fixture.{suffix}" if is_fixture else f"{base}.{suffix}"


def resolve_artifact(directory: Path, base: str, suffix: str) -> tuple[Path, bool]:
    """Return the real artifact when present, else the fixture-labelled one."""
    for candidate, is_fixture in (
        (directory / f"{base}.{suffix}", False),
        (directory / f"{base}_fixture.{suffix}", True),
    ):
        if candidate.exists():
            return candidate, is_fixture
    raise FileNotFoundError(f"no artifact {base}[._fixture].{suffix} in {directory}")


def read_contract_csv(path: Path) -> pd.DataFrame:
    """Read a contract v1 CSV as text so identifiers keep leading zeros."""
    if not path.exists():
        raise FileNotFoundError(f"required input is missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    """Raise when a contract file lacks columns."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def align_matrix(matrix: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Reindex a fresh design matrix to the columns a saved model was fitted on.

    Missing one-hot levels mean an unseen category and are filled with zeros;
    an unexpected text feature is a bug and raises.
    """
    known = set(columns)
    unknown = [
        column for column in matrix.columns if column not in known and "=" not in column
    ]
    if unknown:
        raise ValueError(f"unexpected feature columns: {sorted(unknown)}")
    return matrix.reindex(columns=list(columns), fill_value=0.0)


def load_cost_dataset(prompts_path: Path, tokens_path: Path) -> pd.DataFrame:
    """Join prompts with measured counts, one row per prompt and tokenizer."""
    prompts = read_contract_csv(prompts_path)
    tokens = read_contract_csv(tokens_path)
    require_columns(prompts, PROMPT_COLUMNS, prompts_path.name)
    require_columns(tokens, TOKEN_COLUMNS, tokens_path.name)
    measured = tokens[list(TOKEN_COLUMNS)].copy()
    measured[TARGET] = measured[TARGET].astype(int)
    measured["word_count"] = measured["word_count"].astype(int)
    if (measured[TARGET] <= 0).any():
        raise ValueError(f"{tokens_path.name} has non-positive token_count values")
    measured["fertility"] = measured[TARGET] / measured["word_count"]
    dataset = measured.merge(
        build_features(prompts), on="prompt_id", suffixes=("", "_p")
    )
    for column in ("q_id", "lang"):
        broken = dataset[dataset[column] != dataset[f"{column}_p"]]
        if not broken.empty:
            raise ValueError(
                f"{tokens_path.name} disagrees with prompts.csv on {column} for "
                f"{sorted(set(broken['prompt_id']))}"
            )
    return dataset.drop(columns=["q_id_p", "lang_p"]).reset_index(drop=True)


def fertility_table(dataset: pd.DataFrame) -> pd.DataFrame:
    """Measured fertility per tokenizer and language; a diagnostic, never a feature."""
    return (
        dataset.groupby(["tokenizer", "lang"], sort=True)
        .agg(n=("fertility", "size"), fertility_mean=("fertility", "mean"))
        .round(4)
        .reset_index()
    )


def load_labels(prompts_path: Path, labels_path: Path) -> pd.DataFrame:
    """Join graded observations with pre-call prompt features."""
    labels = read_contract_csv(labels_path)
    require_columns(labels, LABEL_COLUMNS, labels_path.name)
    frame = labels[list(LABEL_COLUMNS)].copy()
    values = frame["is_correct"].str.lower()
    if not values.isin({"true", "false"}).all():
        raise ValueError(f"{labels_path.name} has non-boolean is_correct values")
    frame["is_correct"] = (values == "true").astype(int)
    frame["rep"] = frame["rep"].astype(int)
    features = build_features(read_contract_csv(prompts_path))
    merged = frame.merge(features, on="prompt_id", suffixes=("", "_p"))
    if len(merged) != len(frame):
        raise ValueError(f"{labels_path.name} references unknown prompt_id values")
    return merged.drop(columns=["lang_p", "q_id_p"]).reset_index(drop=True)


def resolve_input(kind: str) -> tuple[Path, bool]:
    """Return the real contract file when present, else the labelled fixture.

    ``is_fixture`` is True whenever the synthetic file is used, so every caller
    can label its outputs.
    """
    real = {
        "prompts": PROMPTS_PATH,
        "tokens": TOKENS_PATH,
        "labels": LABELS_PATH,
        "routes": ROUTES_PATH,
    }[kind]
    if real.exists():
        return real, False
    fixture = FIXTURE_DIR / f"{kind}_fixture.csv"
    if fixture.exists():
        return fixture, True
    raise FileNotFoundError(
        f"neither {real} nor {fixture} exists; run models/fixtures/make_fixtures.py"
    )


def assign_folds_frame(dataset: pd.DataFrame, folds: Sequence[int]) -> pd.DataFrame:
    """Question-level fold table, asserting that no q_id spans two folds."""
    rows = pd.DataFrame({"q_id": dataset["q_id"].to_numpy(), "fold": list(folds)})
    spread = rows.groupby("q_id")["fold"].nunique()
    broken = sorted(spread[spread > 1].index)
    if broken:
        raise ValueError(f"q_id appears in more than one fold: {broken}")
    return rows.drop_duplicates("q_id").sort_values("q_id").reset_index(drop=True)
