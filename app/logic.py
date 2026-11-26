"""Pure, Streamlit-free logic for the offline dashboard (T-006, T-007, T-008).

Data precedence for every table: real contract CSV in ``data/``, then a demo
fixture in ``data/demo/``, then the labelled fixture in ``models/fixtures/``.
Anything that is not real data turns on the example banner. The module never
calls a tokenizer, an API or the network: predictions come from saved models.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.common.config import DATA_DIR, MODELS_DIR, SEED
from src.features import (
    DEFAULT_SPLITS,
    FIXTURE_DIR,
    LANGS,
    META_FEATURES,
    align_matrix,
    assign_folds,
    build_features,
    design_matrix,
    text_features,
)
from src.router import (
    DIRECT,
    KEY,
    STRONGER,
    VIA_ENGLISH,
    baseline_decisions,
    best_cost_model,
    best_risk_model,
    calibrate,
    choose_strategy,
    decode_correctness,
    eligible_keys,
    outcome_summary,
    prepare_routes,
    sweep_thresholds,
    threshold_grid,
)

DEMO_DIR = DATA_DIR / "demo"
FIGS_DIR = MODELS_DIR / "figs"
BANNER_EXAMPLE = "EXAMPLE DATA, not real results"
FORCE_EXAMPLE_ENV = "LT_FORCE_EXAMPLE"
FORCE_EMPTY_ENV = "LT_FORCE_EMPTY"
TABLES: tuple[str, ...] = ("prompts", "tokens", "labels", "routes")
METHOD_LIMITATIONS: list[str] = [
    (
        "Model artifacts are trained on the committed fixtures until T-001 to T-004 "
        "deliver real data."
    ),
    (
        "No tokenizer runs in the browser: without a local T-001 backend the "
        "Playground uses a labelled whitespace fallback, so its chips are not model "
        "tokens."
    ),
    (
        "Native verification share is reported as measured; zero means no Kazakh "
        "reviewer has signed off yet."
    ),
    (
        "Bootstrap intervals resample whole questions (q_id), 1000 draws, seed 42; "
        "small n keeps every number preliminary."
    ),
    (
        "Live machine translation stays disabled without an explicit live flag and a "
        "key, and its output is labelled unverified."
    ),
]


COLD_START_BUDGET_SECONDS = 10.0
TOKENIZE_BUDGET_SECONDS = 1.0
EXAMPLE_PHRASES = {
    "kk": "Қазақстанның ең биік шыңы қалай аталады және оның биіктігі неше метр?",
    "ru": (
        "Как называется самая высокая вершина Казахстана и какова её высота в метрах?"
    ),
    "en": "What is the highest peak of Kazakhstan and how tall is it in metres?",
}
AD_HOC_PROMPT = {
    "prompt_id": "ad-hoc",
    "q_id": "ad-hoc",
    "answer_type": "short",
}


@dataclass(frozen=True)
class AppState:
    """Everything the page renders, resolved once per run."""

    sources: dict[str, str]
    tables: dict[str, pd.DataFrame | None]
    manifests: dict[str, dict[str, Any] | None]
    models_dir: Path
    paths: dict[str, str] = field(default_factory=dict)

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

    def join_prompts(self) -> pd.DataFrame | None:
        """Prompts that share identifiers with the other loaded tables.

        A real ``data/prompts.csv`` cannot be joined with fixture tokens, labels
        or routes: the identifiers differ. When any partner table is synthetic,
        the fixture prompts are used for joins and the banner stays on.
        """
        partners = ("tokens", "labels", "routes")
        if any(self.sources.get(kind) != "real" for kind in partners):
            fixture = FIXTURE_DIR / "prompts_fixture.csv"
            if fixture.exists():
                return read_table(fixture)
        return self.table("prompts")


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
    paths: dict[str, str] = {}
    for kind in TABLES:
        path, source = resolve_table(kind)
        frame = read_table(path)
        if source == "real" and frame_is_synthetic(frame):
            source = "fixture"
        sources[kind] = source if frame is not None else "missing"
        tables[kind] = frame
        paths[kind] = str(path) if path is not None else ""
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
        sources=sources,
        tables=tables,
        manifests=manifests,
        models_dir=models_dir,
        paths=paths,
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


def answer_models(state: AppState) -> list[str]:
    """Answering models the risk panel can score a phrase against."""
    labels = state.table("labels")
    if labels is not None and "model" in labels.columns:
        return sorted(set(labels["model"]))
    manifest = state.manifests.get("risk_metrics") or {}
    known = manifest.get("inputs", {}).get("answer_models", [])
    return list(known) or ["unspecified"]


def empty_state(kind: str) -> str:
    """Readable message for a table or artifact that does not exist yet."""
    return (
        f"No {kind} available offline. Add data/{kind}.csv, data/demo/{kind}.csv or "
        f"models/fixtures/{kind}_fixture.csv; the app makes no network request."
    )


def derived_missing(kind: str) -> str:
    """Readable message for a view built from inputs the app could not find."""
    return (
        f"No {kind} offline: the view is derived from the tokens, labels or routes "
        "tables and the saved models, and at least one of them is missing."
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
    for payload in (models[name] for name in sorted(models)):
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


def load_router_policy(models_dir: Path = MODELS_DIR) -> dict[str, Any] | None:
    """Saved routing policy, or None until T-008 has run."""
    if inputs_disabled():
        return None
    path = candidate(models_dir, "router_policy", "joblib")
    return None if path is None else joblib.load(path)


def _policy_factor(
    policy: dict[str, Any], field: str, strategy: str, lang: str
) -> float:
    """Per-strategy calibration factor for one language, with a documented fallback."""
    for row in policy["statistics"]:
        if row["strategy"] == strategy and row["lang"] == lang:
            return float(row[field])
    return float(
        policy[f"fallback_{'multiplier' if field == 'cost_multiplier' else 'offset'}"][
            strategy
        ]
    )


def router_options(
    phrase: str,
    lang: str,
    category: str,
    policy: dict[str, Any],
    threshold: float | None = None,
    models_dir: Path = MODELS_DIR,
) -> pd.DataFrame | None:
    """Estimated cost and risk of every strategy for one ad-hoc phrase.

    The route language and answering model come from the saved policy; the
    per-strategy multipliers and offsets are the ones fitted during T-008.
    """
    costs = load_cost_models(models_dir)
    risks = load_risk_models(models_dir)
    key = f"{policy['tokenizer']}|{policy['cost_model']}"
    if key not in costs or policy["risk_model"] not in risks:
        return None
    features = build_features(phrase_frame(phrase, lang, category))
    prompt_tokens = float(
        costs[key]["estimator"].predict(
            align_matrix(design_matrix(features), costs[key]["columns"])
        )[0]
    )
    risk_features = features.copy()
    rows = []
    for strategy in policy["strategies"]:
        answer_model = policy["strategy_models"][strategy]
        risk_features["model"] = answer_model
        matrix = design_matrix(
            risk_features,
            meta_columns=tuple(risks[policy["risk_model"]]["meta_columns"]),
        )
        raw = float(
            risks[policy["risk_model"]]["estimator"].predict_proba(
                align_matrix(matrix, risks[policy["risk_model"]]["columns"])
            )[0, 1]
        )
        rows.append(
            {
                "strategy": strategy,
                "answer_model": answer_model,
                "estimated_tokens": round(
                    prompt_tokens
                    * _policy_factor(policy, "cost_multiplier", strategy, lang),
                    2,
                ),
                "risk_incorrect": round(
                    min(
                        max(
                            raw + _policy_factor(policy, "risk_offset", strategy, lang),
                            0.0,
                        ),
                        1.0,
                    ),
                    4,
                ),
            }
        )
    options = pd.DataFrame(rows).rename(
        columns={"estimated_tokens": "cost_hat", "risk_incorrect": "risk_hat"}
    )
    options["chosen"] = options["strategy"] == choose_strategy(
        options, threshold if threshold is not None else float(policy["threshold"])
    )
    return options[["strategy", "answer_model", "cost_hat", "risk_hat", "chosen"]]


def router_rows(manifest: dict[str, Any] | None) -> pd.DataFrame | None:
    """Router sweep and baselines from the T-008 manifest."""
    if not manifest or not manifest.get("curve"):
        return None
    curve = pd.DataFrame(manifest["curve"])[
        ["threshold", "n", "cost_tokens_mean", "accuracy", "expected_accuracy"]
    ]
    return pd.concat([curve, pd.DataFrame(manifest["baselines"])], ignore_index=True)


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


def frame_is_synthetic(frame: pd.DataFrame | None) -> bool:
    """True when a canonical CSV carries fixture provenance inside its rows.

    A file can sit in ``data/`` and still hold fake-client observations, so the
    banner also trusts row-level marks: a ``source`` column saying fixture, or a
    model, model version or tokenizer identifier starting with ``fixture``.
    """
    if frame is None or frame.empty:
        return False
    if (
        "source" in frame.columns
        and frame["source"].astype(str).str.contains("fixture").any()
    ):
        return True
    for column in ("model", "model_version", "tokenizer"):
        if (
            column in frame.columns
            and frame[column].astype(str).str.startswith("fixture").any()
        ):
            return True
    return False


def bootstrap_ci_by_question(
    values: np.ndarray, questions: np.ndarray, resamples: int = 1000, seed: int = SEED
) -> tuple[float, float, float]:
    """Mean and bootstrap 95% percentile interval, resampling whole questions."""
    rng = np.random.default_rng(seed)
    unique = pd.unique(pd.Series(questions))
    positions = {question: np.flatnonzero(questions == question) for question in unique}
    draws = np.empty(resamples)
    for sample in range(resamples):
        picked = rng.choice(unique, size=len(unique), replace=True)
        draws[sample] = values[
            np.concatenate([positions[question] for question in picked])
        ].mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def accuracy_with_ci(
    labels: pd.DataFrame | None, resamples: int = 1000, seed: int = SEED
) -> pd.DataFrame | None:
    """Measured accuracy per language with a bootstrap interval over questions."""
    if labels is None or not {"lang", "q_id", "is_correct"} <= set(labels.columns):
        return None
    frame = labels.copy()
    try:
        decoded = decode_correctness(frame["is_correct"])
    except ValueError:
        return None
    frame["is_correct"] = decoded["is_correct"].astype(float)
    rows = []
    for lang, block in frame.groupby("lang"):
        if lang not in LANGS:
            continue
        values = block["is_correct"].to_numpy(dtype=float)
        mean, low, high = bootstrap_ci_by_question(
            values, block["q_id"].to_numpy(), resamples, seed
        )
        rows.append(
            {
                "lang": lang,
                "n": len(block),
                "n_questions": int(block["q_id"].nunique()),
                "accuracy": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("lang").round(4).reset_index(drop=True)


def _numeric_tokens(tokens: pd.DataFrame) -> pd.DataFrame:
    """Copy of a tokens table with a numeric token_count column."""
    frame = tokens.copy()
    frame["token_count"] = pd.to_numeric(frame["token_count"], errors="coerce")
    return frame.dropna(subset=["token_count"])


def tokens_vs_en_table(tokens: pd.DataFrame | None) -> pd.DataFrame | None:
    """Mean tokens_vs_en per tokenizer and language, paired by question."""
    if tokens is None or not {"q_id", "lang", "tokenizer", "token_count"} <= set(
        tokens.columns
    ):
        return None
    frame = _numeric_tokens(tokens)
    rows = []
    for tokenizer, block in frame.groupby("tokenizer"):
        wide = block.pivot_table(index="q_id", columns="lang", values="token_count")
        if "en" not in wide.columns:
            continue
        base = wide["en"].where(wide["en"] > 0)
        for lang in LANGS:
            if lang not in wide.columns:
                continue
            ratios = (
                (wide[lang] / base)
                if lang != "en"
                else base.notna().astype(float).where(base.notna())
            )
            ratios = ratios.dropna()
            if ratios.empty:
                continue
            rows.append(
                {
                    "tokenizer": tokenizer,
                    "lang": lang,
                    "n": len(ratios),
                    "tokens_vs_en": float(ratios.mean()),
                }
            )
    if not rows:
        return None
    return (
        pd.DataFrame(rows)
        .round(4)
        .sort_values(["tokenizer", "lang"])
        .reset_index(drop=True)
    )


def prompt_ratio_frame(
    tokens: pd.DataFrame | None, tokenizer: str
) -> pd.DataFrame | None:
    """tokens_vs_en per prompt for one tokenizer; missing denominators drop out."""
    if tokens is None or "tokenizer" not in tokens.columns:
        return None
    block = _numeric_tokens(tokens)
    block = block[block["tokenizer"] == tokenizer]
    if block.empty:
        return None
    english = block.loc[block["lang"] == "en", ["q_id", "token_count"]].rename(
        columns={"token_count": "en_tokens"}
    )
    merged = block.merge(english, on="q_id", how="left")
    merged["tokens_vs_en"] = merged["token_count"] / merged["en_tokens"].where(
        merged["en_tokens"] > 0
    )
    return merged.loc[:, ["prompt_id", "tokens_vs_en"]].dropna().reset_index(drop=True)


def playground_frame(
    state: AppState, models_dir: Path = MODELS_DIR
) -> pd.DataFrame | None:
    """Per-prompt table behind the 3D view: ratio, predicted risk, predicted cost."""
    tokens = state.table("tokens")
    prompts = state.join_prompts()
    cost_path = candidate(models_dir, "cost_oof", "csv")
    risk_path = candidate(models_dir, "risk_oof", "csv")
    if tokens is None or prompts is None or cost_path is None or risk_path is None:
        return None
    try:
        cost_model, tokenizer = best_cost_model(state.manifests.get("cost_metrics"))
        risk_model = best_risk_model(state.manifests.get("risk_metrics"))
    except ValueError:
        return None
    costs = pd.read_csv(cost_path)
    costs = costs.loc[
        (costs["model"] == cost_model) & (costs["tokenizer"] == tokenizer),
        ["prompt_id", "prediction"],
    ].rename(columns={"prediction": "cost"})
    risks = pd.read_csv(risk_path)
    risks = risks[risks["model"] == risk_model]
    if risks.empty or "answer_model" not in risks.columns:
        return None
    answer_model = risks["answer_model"].mode().iloc[0]
    risks = risks.loc[risks["answer_model"] == answer_model, ["prompt_id", "risk"]]
    ratios = prompt_ratio_frame(tokens, tokenizer)
    if ratios is None or costs.empty or risks.empty:
        return None
    frame = prompts[["prompt_id", "lang", "category"]].merge(
        ratios, on="prompt_id", how="left"
    )
    frame = frame.merge(costs, on="prompt_id", how="left").merge(
        risks, on="prompt_id", how="left"
    )
    frame = frame.dropna(subset=["tokens_vs_en", "cost", "risk"])
    return frame.reset_index(drop=True) if not frame.empty else None


def router_evaluation(
    state: AppState, models_dir: Path = MODELS_DIR
) -> pd.DataFrame | None:
    """Calibrated route rows for the surface and the curve, or None when unavailable."""
    routes = state.table("routes")
    prompts = state.join_prompts()
    cost_path = candidate(models_dir, "cost_oof", "csv")
    risk_path = candidate(models_dir, "risk_oof", "csv")
    if routes is None or prompts is None or cost_path is None or risk_path is None:
        return None
    try:
        cost_model, tokenizer = best_cost_model(state.manifests.get("cost_metrics"))
        risk_model = best_risk_model(state.manifests.get("risk_metrics"))
        data = prepare_routes(
            routes,
            prompts,
            pd.read_csv(cost_path),
            pd.read_csv(risk_path),
            cost_model,
            tokenizer,
            risk_model,
        )
    except (ValueError, FileNotFoundError):
        return None
    keys = eligible_keys(data)
    data = data[data.set_index(KEY).index.isin(keys.set_index(KEY).index)].reset_index(
        drop=True
    )
    if data.empty:
        return None
    return calibrate(data, assign_folds(list(data["q_id"]), DEFAULT_SPLITS, SEED))


def router_surface_frame(routes: pd.DataFrame | None) -> pd.DataFrame | None:
    """Threshold by language grid of saved tokens and held-out accuracy."""
    if routes is None:
        return None
    _, decisions = sweep_thresholds(routes, threshold_grid())
    direct = routes.loc[routes["strategy"] == "direct", [*KEY, "total_tokens"]].rename(
        columns={"total_tokens": "direct_tokens"}
    )
    rows = []
    for threshold, block in decisions.items():
        merged = block.merge(direct, on=KEY, how="left").dropna(
            subset=["direct_tokens"]
        )
        for lang, part in merged.groupby("lang"):
            rows.append(
                {
                    "threshold": float(threshold),
                    "lang": lang,
                    "n": len(part),
                    "saved_tokens": float(
                        (part["direct_tokens"] - part["total_tokens"]).mean()
                    ),
                    "accuracy": float(part["is_correct"].mean()),
                }
            )
    if not rows:
        return None
    return (
        pd.DataFrame(rows)
        .round(4)
        .sort_values(["lang", "threshold"])
        .reset_index(drop=True)
    )


def observed_tokenizers(tokens: pd.DataFrame | None) -> list[str]:
    """Tokenizers present in the loaded tokens table, sorted for a stable menu."""
    if tokens is None or "tokenizer" not in tokens.columns:
        return []
    return sorted(tokens["tokenizer"].astype(str).unique())


def router_curve_frames(
    routes: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Threshold sweep plus the always-use-one-strategy baselines behind it."""
    if routes is None:
        return None
    curve, _ = sweep_thresholds(routes, threshold_grid())
    rows = [
        outcome_summary(baseline_decisions(routes, strategy), f"always_{strategy}")
        for strategy in (DIRECT, VIA_ENGLISH, STRONGER)
    ]
    return curve.round(4), pd.DataFrame(rows).round(4)


def curve_baselines(baselines: pd.DataFrame) -> pd.DataFrame:
    """The two baselines the Results tab compares the router against."""
    wanted = ("always_direct", "always_via_english")
    return baselines[baselines["policy"].isin(wanted)].reset_index(drop=True)


def route_reason(options: pd.DataFrame, threshold: float) -> str:
    """One-line explanation of the router decision."""
    chosen = options[options["chosen"]]
    if chosen.empty:
        return "no strategy is available for this phrase"
    row = chosen.iloc[0]
    if (options["risk_hat"] <= threshold).any():
        return (
            f"{row['strategy']}: cheapest route with predicted risk "
            f"{row['risk_hat']:.2f} under the threshold {threshold:.2f}, "
            f"about {row['cost_hat']:.0f} tokens"
        )
    return (
        f"{row['strategy']}: nothing stays under the threshold {threshold:.2f}, "
        f"so the safest route wins (predicted risk {row['risk_hat']:.2f})"
    )


def verified_share(prompts: pd.DataFrame | None) -> float | None:
    """Share of prompts a native reviewer marked as verified."""
    if prompts is None or "verified_by_native" not in prompts.columns:
        return None
    values = prompts["verified_by_native"].astype(str).str.strip().str.lower()
    return float(values.isin({"true", "1"}).mean())


def method_tables(state: AppState) -> dict[str, pd.DataFrame]:
    """Provenance, model versions and limitations for the Method tab."""
    sources = pd.DataFrame(
        [
            {
                "input": kind,
                "source": source,
                "path": state.paths.get(kind, ""),
                "rows": 0 if state.table(kind) is None else len(state.table(kind)),
            }
            for kind, source in sorted(state.sources.items())
        ]
    )
    versions = []
    for task, manifest in sorted(state.manifests.items()):
        if not manifest:
            continue
        packages = manifest.get("versions", {})
        versions.append(
            {
                "task": task,
                "generated_at": manifest.get("generated_at", ""),
                "command": manifest.get("command", ""),
                "seed": manifest.get("seed", ""),
                "fixture": bool(manifest.get("fixture")),
                "n_rows": manifest.get("inputs", {}).get("n_rows", ""),
                "python": packages.get("python", ""),
                "scikit-learn": packages.get("scikit-learn", ""),
                "lightgbm": packages.get("lightgbm", ""),
                "shap": packages.get("shap", ""),
            }
        )
    version_frame = pd.DataFrame(versions)
    if not version_frame.empty:
        # Every column stays text so the page can show it without a dtype mix.
        version_frame = version_frame.astype(str)
    limitations = pd.DataFrame({"limitation": METHOD_LIMITATIONS})
    return {
        "sources": sources,
        "versions": version_frame,
        "limitations": limitations,
    }
