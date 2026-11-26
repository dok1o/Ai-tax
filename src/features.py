"""Feature building and contract-shaped input loading for the ARENA tasks.

Every feature is computable before a tokenizer or a model runs: prompt text
statistics, language, category and answer type. Measured ``token_count`` and
``fertility`` are targets or leaks and are rejected by :func:`assert_no_leakage`.

Standard parts: one-hot encoding, scikit-learn estimators, LightGBM, SHAP.
Ours: the trilingual pre-tokenization feature set, the router formulation and
the cost-quality evaluation.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.common.config import (
    DATA_DIR,
    LABELS_PATH,
    MODELS_DIR,
    PROMPTS_PATH,
    RESPONSES_PATH,
    SEED,
    TOKENS_PATH,
)

LANGS: tuple[str, ...] = ("kk", "ru", "en")
# Example mode: resolve the committed fixtures even after a real run has written
# artifacts next to them. The dashboard and the hermetic test session use it.
EXAMPLE_ENV = "LT_FORCE_EXAMPLE"
DEFAULT_SPLITS = 5
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
# LightGBM refuses these characters in feature names ("special JSON characters"),
# and real model identifiers such as `gemma2:2b` carry a colon. One-hot levels are
# translated before they become column names so every compared model sees the same
# matrix. Dots, hyphens and `=` are accepted and stay untouched.
JSON_UNSAFE = str.maketrans({character: "_" for character in ':[]{},"'})
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
RESPONSE_COLUMNS: tuple[str, ...] = (
    "model",
    "prompt_id",
    "lang",
    "rep",
    "q_id",
    "input_tokens",
    "output_tokens",
    "usage_available",
)
# The router's three strategies, named here so the derived route legs and the
# router cannot drift apart.
STRATEGY_DIRECT = "direct"
STRATEGY_VIA_ENGLISH = "via_english"
STRATEGY_STRONGER = "stronger_model"
ROUTE_STRATEGIES: tuple[str, ...] = (
    STRATEGY_DIRECT,
    STRATEGY_VIA_ENGLISH,
    STRATEGY_STRONGER,
)
ROUTE_ROW_COLUMNS: tuple[str, ...] = (
    "q_id",
    "prompt_id",
    "lang",
    "strategy",
    "model",
    "input_tokens",
    "translation_tokens",
    "output_tokens",
    "total_tokens",
    "is_correct",
    "source",
)
# Where the derived legs live: `data/` belongs to Codex, `models/` to ARENA.
DERIVED_ROUTES_PATH = MODELS_DIR / "routes_derived.csv"
DERIVED_SOURCE = "derived"
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
            name = f"{column}={level}".translate(JSON_UNSAFE)
            if name in encoded:
                raise ValueError(
                    f"two {column} levels share the sanitized feature name {name}"
                )
            encoded[name] = (values == level).astype(float).tolist()
    matrix = pd.concat([numeric, pd.DataFrame(encoded)], axis=1)
    assert_no_leakage(matrix.columns)
    return matrix


def artifact_name(base: str, suffix: str, is_fixture: bool) -> str:
    """Name a generated file so fixture-derived output is labelled in the name."""
    return f"{base}_fixture.{suffix}" if is_fixture else f"{base}.{suffix}"


def resolve_artifact(directory: Path, base: str, suffix: str) -> tuple[Path, bool]:
    """Return the real artifact when present, else the fixture-labelled one.

    ``LT_FORCE_EXAMPLE=1`` reverses the preference, so the example mode and the
    test session keep resolving the committed fixtures instead of whatever the
    last real run wrote into the same directory.
    """
    order = (
        (directory / f"{base}.{suffix}", False),
        (directory / f"{base}_fixture.{suffix}", True),
    )
    if os.environ.get(EXAMPLE_ENV) == "1":
        order = tuple(reversed(order))
    for candidate, is_fixture in order:
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


CORRECTNESS_VALUES = frozenset({"true", "false", "0", "1"})
PARSE_STATUS_VALUES = frozenset({"graded", "unparsed"})


def decode_correctness(values: pd.Series, source: str = "labels") -> pd.DataFrame:
    """Decode contract booleans or 0/1 flags into graded 0/1 outcome columns.

    ``true`` and ``1`` mean the answer was correct; ``false`` and ``0`` mean it
    was not, so a numeric flag is never inverted. Any other spelling is rejected
    instead of being silently read as false.
    """
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin(CORRECTNESS_VALUES).all():
        bad = sorted(set(normalized) - CORRECTNESS_VALUES)
        raise ValueError(f"{source} has non-boolean is_correct values: {bad}")
    incorrect = normalized.isin({"false", "0"})
    return pd.DataFrame(
        {"is_correct": (~incorrect).astype(int), "is_incorrect": incorrect.astype(int)},
        index=values.index,
    )


def load_labels(prompts_path: Path, labels_path: Path) -> pd.DataFrame:
    """Join graded observations with pre-call prompt features."""
    labels = read_contract_csv(labels_path)
    require_columns(labels, LABEL_COLUMNS, labels_path.name)
    frame = labels[list(LABEL_COLUMNS)].copy()
    frame["is_correct"] = decode_correctness(
        frame["is_correct"], source=labels_path.name
    )["is_correct"]
    if "parse_status" in labels.columns:
        status = labels["parse_status"].astype(str).str.strip().str.lower()
        unknown = sorted(set(status) - PARSE_STATUS_VALUES)
        if unknown:
            raise ValueError(f"{labels_path.name} has unknown parse_status: {unknown}")
        # Contract v2: an unparsed answer is scored false, whatever the flag says.
        frame.loc[status == "unparsed", "is_correct"] = 0
    frame["rep"] = frame["rep"].astype(int)
    features = build_features(read_contract_csv(prompts_path))
    merged = frame.merge(features, on="prompt_id", suffixes=("", "_p"))
    if len(merged) != len(frame):
        raise ValueError(f"{labels_path.name} references unknown prompt_id values")
    return merged.drop(columns=["lang_p", "q_id_p"]).reset_index(drop=True)


def resolve_input(kind: str, partner: str | None = None) -> tuple[Path, bool]:
    """Return the real contract file when present, else the labelled fixture.

    ``is_fixture`` is True whenever the synthetic file is used, so every caller
    can label its outputs.

    ``partner`` names a second input the caller joins this one on. When the
    partner can only be resolved synthetically, ``kind`` is resolved
    synthetically as well: a real ``data/prompts.csv`` shares no identifier with
    the fixture tables (``m01-kk`` against ``q000-kk``), so a mixed pair joins
    nothing and would fail deep inside training instead of at resolution.
    """
    real = {
        "prompts": PROMPTS_PATH,
        "tokens": TOKENS_PATH,
        "labels": LABELS_PATH,
        "routes": ROUTES_PATH,
    }[kind]
    fixture = FIXTURE_DIR / f"{kind}_fixture.csv"
    if partner is not None and not _partner_is_real(partner) and fixture.exists():
        return fixture, True
    if real.exists():
        return real, False
    if fixture.exists():
        return fixture, True
    raise FileNotFoundError(
        f"neither {real} nor {fixture} exists; run models/fixtures/make_fixtures.py"
    )


def _partner_is_real(partner: str) -> bool:
    """Whether a partner input can be resolved from real data.

    Routes count as real when the observed legs exist *or* the real observations
    needed to derive them do, so real prompts are not pushed onto the fixture
    table just because ``data/routes.csv`` was never published.
    """
    return (
        real_routes_available() if partner == "routes" else _real_path(partner).exists()
    )


def _real_path(kind: str) -> Path:
    """Contract path of one input kind, whether it exists or not."""
    return {
        "prompts": PROMPTS_PATH,
        "tokens": TOKENS_PATH,
        "labels": LABELS_PATH,
        "responses": RESPONSES_PATH,
        "routes": ROUTES_PATH,
    }[kind]


def load_observations(responses_path: Path, labels_path: Path) -> pd.DataFrame:
    """Graded observations with the provider's own token counts, one row each.

    Nothing is imputed: a row without provider usage or without a grade is an
    error, because the router bills exactly these numbers.
    """
    responses = read_contract_csv(responses_path)
    require_columns(responses, RESPONSE_COLUMNS, responses_path.name)
    labels = read_contract_csv(labels_path)
    require_columns(labels, LABEL_COLUMNS, labels_path.name)
    key = ["model", "prompt_id", "lang", "rep"]

    graded = labels[list(LABEL_COLUMNS)].copy()
    graded["is_correct"] = decode_correctness(
        graded["is_correct"], source=labels_path.name
    )["is_correct"]
    if "parse_status" in labels.columns:
        status = labels["parse_status"].astype(str).str.strip().str.lower()
        # Contract v2.1: accuracy_strict scores an unparsed answer as wrong.
        graded.loc[status == "unparsed", "is_correct"] = 0
    graded["rep"] = graded["rep"].astype(int)

    counts = responses[list(RESPONSE_COLUMNS)].copy()
    available = counts["usage_available"].astype(str).str.strip().str.lower()
    if not (available == "true").all():
        missing = int((available != "true").sum())
        raise ValueError(
            f"{responses_path.name} has {missing} rows without provider usage; "
            "refusing to invent a token cost"
        )
    for column in ("input_tokens", "output_tokens"):
        counts[column] = pd.to_numeric(counts[column])
        if counts[column].isna().any() or (counts[column] < 0).any():
            raise ValueError(f"{responses_path.name} has unusable {column} values")
    counts["rep"] = counts["rep"].astype(int)

    data = counts.merge(
        graded[key + ["is_correct"]], on=key, how="left", validate="one_to_one"
    )
    if data["is_correct"].isna().any():
        ungraded = int(data["is_correct"].isna().sum())
        raise ValueError(
            f"{labels_path.name} grades {len(data) - ungraded} of {len(data)} "
            f"observations in {responses_path.name}"
        )
    data["is_correct"] = data["is_correct"].astype(int)
    data["total_tokens"] = data["input_tokens"] + data["output_tokens"]
    return data.sort_values(key).reset_index(drop=True)


def route_roles(observations: pd.DataFrame) -> dict[str, str]:
    """Which observed model plays which strategy, chosen from the data.

    ``direct`` is the cheapest model by mean observed total tokens. ``via_english``
    asks the same model the English prompt of the same question. ``stronger_model``
    is the most accurate model that is not ``direct``, so the router really has two
    native options to choose between.
    """
    grouped = observations.groupby("model").agg(
        mean_total_tokens=("total_tokens", "mean"), accuracy=("is_correct", "mean")
    )
    if len(grouped) < 2:
        raise ValueError("deriving route legs needs at least two observed models")
    direct = str(grouped["mean_total_tokens"].idxmin())
    ranked = grouped.sort_values(
        ["accuracy", "mean_total_tokens"], ascending=[False, True]
    )
    stronger = str(next(name for name in ranked.index if str(name) != direct))
    return {
        STRATEGY_DIRECT: direct,
        STRATEGY_VIA_ENGLISH: direct,
        STRATEGY_STRONGER: stronger,
    }


def build_route_rows(
    observations: pd.DataFrame, prompts: pd.DataFrame, rep: int = 0
) -> pd.DataFrame:
    """Observed route legs: three strategies for every question and language.

    Costs are the provider's own counts for one observation per key (``rep``);
    the router's unit is the question, so a second repetition would only be
    aggregated away again. ``translation_tokens`` is 0 because no translation
    call was ever made: the ``via_english`` cost is therefore a lower bound and
    is reported as one, never padded with an invented overhead.
    """
    require_columns(prompts, PROMPT_COLUMNS, "prompts")
    roles = route_roles(observations)
    block = observations[observations["rep"] == rep]
    if block.empty:
        raise ValueError(f"no observation with rep={rep} to derive routes from")
    native = prompts[["q_id", "prompt_id", "lang"]].copy()
    english = prompts.loc[prompts["lang"] == "en", ["q_id", "prompt_id"]].rename(
        columns={"prompt_id": "en_prompt_id"}
    )
    native = native.merge(english, on="q_id", how="left")
    if native["en_prompt_id"].isna().any():
        missing = sorted(set(native.loc[native["en_prompt_id"].isna(), "q_id"]))
        raise ValueError(f"deriving via_english needs an English prompt for {missing}")

    legs = []
    for strategy in ROUTE_STRATEGIES:
        model = roles[strategy]
        leg = native.copy()
        leg["strategy"] = strategy
        leg["model"] = model
        leg["observed_prompt_id"] = np.where(
            strategy == STRATEGY_VIA_ENGLISH, leg["en_prompt_id"], leg["prompt_id"]
        )
        used = block.loc[
            block["model"] == model,
            ["prompt_id", "input_tokens", "output_tokens", "is_correct"],
        ]
        merged = leg.merge(
            used,
            left_on="observed_prompt_id",
            right_on="prompt_id",
            how="left",
            suffixes=("", "_obs"),
        )
        if merged["input_tokens"].isna().any():
            missing = sorted(
                set(merged.loc[merged["input_tokens"].isna(), "observed_prompt_id"])
            )
            raise ValueError(
                f"{model} has no rep={rep} observation for {missing[:5]} "
                f"({int(merged['input_tokens'].isna().sum())} rows)"
            )
        merged["translation_tokens"] = 0
        merged["total_tokens"] = merged["input_tokens"] + merged["output_tokens"]
        merged["is_correct"] = np.where(merged["is_correct"] == 1, "true", "false")
        merged["source"] = DERIVED_SOURCE
        legs.append(
            merged.drop(
                columns=["prompt_id_obs", "en_prompt_id", "observed_prompt_id"],
                errors="ignore",
            )[list(ROUTE_ROW_COLUMNS)]
        )
    return (
        pd.concat(legs, ignore_index=True)
        .sort_values(["q_id", "lang", "strategy"])
        .reset_index(drop=True)
    )


def real_routes_available() -> bool:
    """True when the real inputs for observed route legs can be resolved.

    A committed ``routes_derived.csv`` on its own does not count: the derived
    legs belong to the real responses and labels they were built from, so a run
    that cannot see those inputs (the hermetic test session, the example mode of
    the dashboard) must stay on the labelled fixture.
    """
    return bool(
        _real_path("routes").exists()
        or (
            _real_path("responses").exists()
            and _real_path("labels").exists()
            and _real_path("prompts").exists()
        )
    )


def derive_routes(out_path: Path | None = None) -> tuple[Path, dict[str, str]]:
    """Write the derived route legs next to the other ARENA artifacts.

    The destination is resolved when the call happens, not when the module is
    imported, so the file that gets written is always the one the caller is told
    about.
    """
    destination = DERIVED_ROUTES_PATH if out_path is None else out_path
    observations = load_observations(RESPONSES_PATH, LABELS_PATH)
    prompts = read_contract_csv(PROMPTS_PATH)
    routes = build_route_rows(observations, prompts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    routes.to_csv(destination, index=False)
    return destination, route_roles(observations)


def ensure_routes() -> tuple[Path, bool, str]:
    """Routes path, whether it is a fixture, and where the legs came from."""
    real = _real_path("routes")
    if real.exists():
        return real, False, "observed"
    if real_routes_available():
        # Rebuild every time: the legs must match the responses on disk, not a
        # copy that an earlier run happened to leave behind.
        path, _roles = derive_routes()
        return path, False, DERIVED_SOURCE
    fixture = FIXTURE_DIR / "routes_fixture.csv"
    if fixture.exists():
        return fixture, True, "fixture"
    raise FileNotFoundError(
        f"neither {real} nor {fixture} exists; run models/fixtures/make_fixtures.py"
    )


def assign_folds(
    groups: Sequence[str], n_splits: int = DEFAULT_SPLITS, seed: int = SEED
) -> np.ndarray:
    """Assign one fold per row so every ``q_id`` stays entirely on one side."""
    unique = pd.unique(pd.Series(list(groups), dtype="string"))
    if len(unique) < n_splits:
        raise ValueError(
            f"need at least {n_splits} distinct q_id groups, found {len(unique)}"
        )
    order = np.random.default_rng(seed).permutation(len(groups))
    grouped = np.asarray(groups, dtype=object)[order]
    folds = np.zeros(len(groups), dtype=int)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, test) in enumerate(
        splitter.split(np.zeros((len(order), 1)), groups=grouped)
    ):
        folds[order[test]] = fold
    return folds


def assign_folds_frame(dataset: pd.DataFrame, folds: Sequence[int]) -> pd.DataFrame:
    """Question-level fold table, asserting that no q_id spans two folds."""
    rows = pd.DataFrame({"q_id": dataset["q_id"].to_numpy(), "fold": list(folds)})
    spread = rows.groupby("q_id")["fold"].nunique()
    broken = sorted(spread[spread > 1].index)
    if broken:
        raise ValueError(f"q_id appears in more than one fold: {broken}")
    return rows.drop_duplicates("q_id").sort_values("q_id").reset_index(drop=True)
