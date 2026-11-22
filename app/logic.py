"""Pure, Streamlit-free logic for the offline dashboard (T-006, T-007, T-008).

Data precedence for every table: real contract CSV in ``data/``, then a demo
fixture in ``data/demo/``, then the labelled fixture in ``models/fixtures/``.
Anything that is not real data turns on the example banner. The module never
calls a tokenizer, an API or the network: predictions come from saved models.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.common.config import DATA_DIR, MODELS_DIR
from src.features import (
    FIXTURE_DIR,
    LANGS,
    META_FEATURES,
    align_matrix,
    build_features,
    design_matrix,
    text_features,
)

DEMO_DIR = DATA_DIR / "demo"
FIGS_DIR = MODELS_DIR / "figs"
BANNER_EXAMPLE = "EXAMPLE DATA, not real results"
FORCE_EXAMPLE_ENV = "LT_FORCE_EXAMPLE"
FORCE_EMPTY_ENV = "LT_FORCE_EMPTY"
TABLES: tuple[str, ...] = ("prompts", "tokens", "labels", "routes")
EXAMPLE_PHRASES = {
    "kk": "Қазақстанның ең биік шыңы қалай аталады және оның биіктігі неше метр?",
    "ru": "Как называется самая высокая вершина Казахстана и какова её высота в метрах?",
    "en": "What is the highest peak of Kazakhstan and how tall is it in metres?",
}
AD_HOC_PROMPT = {
    "prompt_id": "ad-hoc",
    "q_id": "ad-hoc",
    "answer_type": "short_text",
}


@dataclass(frozen=True)
class AppState:
    """Everything the page renders, resolved once per run."""

    sources: dict[str, str]
    tables: dict[str, pd.DataFrame | None]
    manifests: dict[str, dict[str, Any] | None]
    models_dir: Path

    @property
    def is_example(self) -> bool:
        """True when any table or manifest is synthetic or missing."""
        return any(source != "real" for source in self.sources.values())

    @property
    def banner(self) -> str | None:
        """Visible banner text, or None when every input is real."""
        return BANNER_EXAMPLE if self.is_example else None

    def table(self, kind: str) -> pd.DataFrame | None:
        """One loaded table, or None when it does not exist anywhere."""
        return self.tables.get(kind)


def inputs_disabled() -> bool:
    """True when the harness asks for a fully empty offline state."""
    return os.environ.get(FORCE_EMPTY_ENV) == "1"


def candidate(directory: Path, base: str, suffix: str) -> Path | None:
    """Prefer the real artifact and fall back to the fixture-labelled one."""
    for name in (f"{base}.{suffix}", f"{base}_fixture.{suffix}"):
        path = directory / name
        if path.exists():
            return path
    return None


def resolve_table(kind: str) -> tuple[Path | None, str]:
    """Locate one input table and report where it came from."""
    if inputs_disabled():
        return None, "missing"
    if os.environ.get(FORCE_EXAMPLE_ENV) != "1":
        for directory, source in ((DATA_DIR, "real"), (DEMO_DIR, "demo")):
            path = directory / f"{kind}.csv"
            if path.exists():
                return path, source
    fixture = FIXTURE_DIR / f"{kind}_fixture.csv"
    return (fixture, "fixture") if fixture.exists() else (None, "missing")


def read_table(path: Path | None) -> pd.DataFrame | None:
    """Read a CSV as text, returning None when there is nothing to read."""
    if path is None:
        return None
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame if not frame.empty else None


def load_manifest(models_dir: Path, base: str) -> dict[str, Any] | None:
    """Load one metrics manifest, or None when the task has not run yet."""
    path = None if inputs_disabled() else candidate(models_dir, base, "json")
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_state(models_dir: Path = MODELS_DIR) -> AppState:
    """Resolve tables and manifests without touching the network."""
    sources: dict[str, str] = {}
    tables: dict[str, pd.DataFrame | None] = {}
    for kind in TABLES:
        path, source = resolve_table(kind)
        sources[kind] = source
        tables[kind] = read_table(path)
    manifests = {
        base: load_manifest(models_dir, base)
        for base in ("cost_metrics", "risk_metrics", "router_metrics")
    }
    for name, manifest in manifests.items():
        sources[name] = (
            "missing"
            if manifest is None
            else ("fixture" if manifest.get("fixture") else "real")
        )
    return AppState(
        sources=sources, tables=tables, manifests=manifests, models_dir=models_dir
    )


def tokens_per_language(tokens: pd.DataFrame | None) -> pd.DataFrame | None:
    """Mean measured token count per language and tokenizer."""
    if tokens is None or "token_count" not in tokens.columns:
        return None
    frame = tokens.copy()
    frame["token_count"] = pd.to_numeric(frame["token_count"], errors="coerce")
    grouped = (
        frame.groupby(["lang", "tokenizer"])["token_count"]
        .agg(n="size", tokens_mean="mean")
        .round(2)
        .reset_index()
    )
    return grouped.sort_values(["tokenizer", "lang"]).reset_index(drop=True)


def accuracy_per_language(labels: pd.DataFrame | None) -> pd.DataFrame | None:
    """Measured accuracy per language and answering model."""
    if labels is None or "is_correct" not in labels.columns:
        return None
    frame = labels.copy()
    frame["is_correct"] = frame["is_correct"].str.lower().map({"true": 1, "false": 0})
    frame = frame.dropna(subset=["is_correct"])
    if frame.empty:
        return None
    models = ["model"] if "model" in frame.columns else []
    grouped = (
        frame.groupby(["lang", *models])["is_correct"]
        .agg(n="size", accuracy="mean")
        .round(4)
        .reset_index()
    )
    return grouped.sort_values(["lang", *models]).reset_index(drop=True)


def cost_rows(manifest: dict[str, Any] | None) -> pd.DataFrame | None:
    """Model comparison table from the cost manifest."""
    if not manifest or not manifest.get("results"):
        return None
    frame = pd.DataFrame(manifest["results"])
    return (
        frame[frame["lang"] == "all"]
        .sort_values(["tokenizer", "model"])
        .reset_index(drop=True)
    )


def fertility_rows(manifest: dict[str, Any] | None) -> pd.DataFrame | None:
    """Measured fertility diagnostic from the cost manifest."""
    if not manifest or not manifest.get("fertility_diagnostic"):
        return None
    return pd.DataFrame(manifest["fertility_diagnostic"])


def risk_rows(manifest: dict[str, Any] | None) -> pd.DataFrame | None:
    """Held-out risk metrics per model, pooled over languages."""
    if not manifest or not manifest.get("results"):
        return None
    frame = pd.DataFrame(manifest["results"])
    return frame[frame["lang"] == "all"].sort_values("model").reset_index(drop=True)


def empty_state(kind: str) -> str:
    """Readable message for a table or artifact that does not exist yet."""
    return (
        f"No {kind} available offline. Add data/{kind}.csv, data/demo/{kind}.csv or "
        f"models/fixtures/{kind}_fixture.csv; the app makes no network request."
    )


def phrase_frame(phrase: str, lang: str, category: str) -> pd.DataFrame:
    """One contract-shaped prompt row for an ad-hoc phrase."""
    if lang not in LANGS:
        raise ValueError(f"unsupported language: {lang}")
    if not phrase.strip():
        raise ValueError("phrase is empty")
    return pd.DataFrame(
        [{**AD_HOC_PROMPT, "lang": lang, "category": category, "prompt": phrase}]
    )


def phrase_features(phrase: str, lang: str, category: str) -> dict[str, float]:
    """Pre-tokenization features of one phrase, shown next to the prediction."""
    features = text_features(phrase)
    return {
        "lang": lang,
        "category": category,
        "n_chars": features["n_chars"],
        "n_words": features["n_words"],
        "kk_share": round(features["kk_share"], 4),
        "cyrillic_share": round(features["cyrillic_share"], 4),
    }


def load_cost_models(models_dir: Path = MODELS_DIR) -> dict[str, dict[str, Any]]:
    """Saved cost models keyed by ``tokenizer|model``."""
    loaded: dict[str, dict[str, Any]] = {}
    if inputs_disabled():
        return loaded
    for path in sorted(models_dir.glob("cost_*.joblib")):
        payload = joblib.load(path)
        loaded[f"{payload['tokenizer']}|{payload['model_name']}"] = payload
    return loaded


def load_risk_models(models_dir: Path = MODELS_DIR) -> dict[str, dict[str, Any]]:
    """Saved error-risk models keyed by model name."""
    loaded: dict[str, dict[str, Any]] = {}
    if inputs_disabled():
        return loaded
    for name in ("train_rate", "logistic", "lightgbm"):
        path = candidate(models_dir, f"risk_{name}", "joblib")
        if path is not None:
            loaded[name] = joblib.load(path)
    return loaded


def predict_cost(
    phrase: str,
    lang: str,
    category: str,
    models: dict[str, dict[str, Any]],
) -> pd.DataFrame | None:
    """Predicted token count per tokenizer and model for one phrase."""
    if not models:
        return None
    features = build_features(phrase_frame(phrase, lang, category))
    matrix = design_matrix(features)
    rows = []
    for key, payload in sorted(models.items()):
        aligned = align_matrix(matrix, payload["columns"])
        value = float(payload["estimator"].predict(aligned)[0])
        rows.append(
            {
                "tokenizer": payload["tokenizer"],
                "model": payload["model_name"],
                "predicted_tokens": round(max(value, 0.0), 2),
                "fixture": bool(payload.get("fixture")),
            }
        )
    return pd.DataFrame(rows)


def predict_risk(
    phrase: str,
    lang: str,
    category: str,
    answer_model: str,
    models: dict[str, dict[str, Any]],
) -> pd.DataFrame | None:
    """Predicted P(incorrect) per saved risk model for one phrase."""
    if not models:
        return None
    features = build_features(phrase_frame(phrase, lang, category))
    features["model"] = answer_model
    rows = []
    for name, payload in sorted(models.items()):
        matrix = design_matrix(features, meta_columns=tuple(payload["meta_columns"]))
        aligned = align_matrix(matrix, payload["columns"])
        risk = float(payload["estimator"].predict_proba(aligned)[0, 1])
        rows.append(
            {
                "model": name,
                "risk_incorrect": round(min(max(risk, 0.0), 1.0), 4),
                "fixture": bool(payload.get("fixture")),
            }
        )
    return pd.DataFrame(rows)


def figure_paths(figs_dir: Path = FIGS_DIR) -> list[Path]:
    """Saved figures, real ones before fixture-labelled ones."""
    if not figs_dir.exists():
        return []
    return sorted(
        figs_dir.glob("*.png"),
        key=lambda path: (path.stem.endswith("_fixture"), path.name),
    )


def default_phrase(state: AppState, lang: str) -> str:
    """First real prompt in the language, else the bundled example phrase."""
    prompts = state.table("prompts")
    if prompts is not None and {"lang", "prompt"}.issubset(prompts.columns):
        matches = prompts[prompts["lang"] == lang]
        if not matches.empty:
            return str(matches.iloc[0]["prompt"])
    return EXAMPLE_PHRASES[lang]


def categories(state: AppState) -> list[str]:
    """Category choices for the selector."""
    prompts = state.table("prompts")
    if prompts is not None and "category" in prompts.columns:
        return sorted(set(prompts["category"]))
    return ["general"]


def feature_columns() -> tuple[str, ...]:
    """Feature names the page can show as documentation."""
    return META_FEATURES
