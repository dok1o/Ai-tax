"""Tests for the rerun on the real T-004 observations.

Three things are pinned here: the feature names LightGBM accepts, the route legs
derived from the observed responses and labels, and the provenance rules that
decide whether a run reports real numbers or the labelled fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import features
from src.features import (
    DERIVED_ROUTES_PATH,
    ROUTE_ROW_COLUMNS,
    STRATEGY_DIRECT,
    STRATEGY_STRONGER,
    STRATEGY_VIA_ENGLISH,
    build_route_rows,
    design_matrix,
    ensure_routes,
    load_observations,
    real_routes_available,
    route_roles,
)
from src.router import prepare_routes

MODELS_DIR = features.MODELS_DIR
SMALL = "small:1b"
BIG = "big:2b"
KEY = ["model", "prompt_id", "lang", "rep"]
# Columns `load_observations` adds and the contract CSVs do not carry.
DERIVED_ONLY = ["is_correct", "parse_status", "total_tokens"]


def real_prompts() -> pd.DataFrame:
    """Two questions in three languages, contract v2.1 shape."""
    rows = []
    for q_id in ("q_001", "q_002"):
        for lang in ("kk", "ru", "en"):
            rows.append(
                {
                    "prompt_id": f"{q_id}-{lang}",
                    "q_id": q_id,
                    "lang": lang,
                    "category": "math_logic",
                    "prompt": f"{lang} prompt {q_id}",
                    "answer_type": "number",
                    "gold_answer": "36",
                    "verified_by_native": "0",
                }
            )
    return pd.DataFrame(rows)


def real_observations(reps: tuple[int, ...] = (0, 1)) -> pd.DataFrame:
    """Graded observations: the small model is cheap and wrong, the big one is not."""
    prompts = real_prompts()
    rows = []
    for model, (inp, out) in ((SMALL, (10, 5)), (BIG, (20, 20))):
        for rep in reps:
            for _, prompt in prompts.iterrows():
                correct = model == BIG and prompt["lang"] != "kk"
                rows.append(
                    {
                        "model": model,
                        "prompt_id": prompt["prompt_id"],
                        "q_id": prompt["q_id"],
                        "lang": prompt["lang"],
                        "rep": rep,
                        "input_tokens": inp + rep,
                        "output_tokens": out,
                        "usage_available": "true",
                        "is_correct": int(correct),
                        "parse_status": "graded",
                    }
                )
    frame = pd.DataFrame(rows)
    # Shaped the way `load_observations` hands it on: graded, with the total.
    frame["total_tokens"] = frame["input_tokens"] + frame["output_tokens"]
    return frame


def write_contract(directory: Path, name: str, frame: pd.DataFrame) -> Path:
    path = directory / name
    frame.to_csv(path, index=False)
    return path


def test_one_hot_levels_are_sanitized_for_lightgbm() -> None:
    """Regression: real model ids carry a colon, which LightGBM rejects."""
    features_frame = pd.DataFrame(
        [
            {
                **{name: 0.5 for name in features.TEXT_FEATURES},
                "lang": "kk",
                "model": SMALL,
            },
            {
                **{name: 0.25 for name in features.TEXT_FEATURES},
                "lang": "en",
                "model": BIG,
            },
        ]
    )
    matrix = design_matrix(features_frame, meta_columns=("lang", "model"))
    assert f"model={SMALL}" not in matrix.columns
    assert "model=small_1b" in matrix.columns and "model=big_2b" in matrix.columns
    assert "lang=kk" in matrix.columns, "safe names are left alone"
    import lightgbm as lgb

    model = lgb.LGBMClassifier(n_estimators=2, verbose=-1)
    model.fit(matrix.to_numpy(), [0, 1], feature_name=list(matrix.columns))
    assert list(model.booster_.feature_name()) == list(matrix.columns)


def test_two_levels_that_collapse_onto_one_name_are_refused() -> None:
    frame = pd.DataFrame(
        [
            {
                **{name: 0.5 for name in features.TEXT_FEATURES},
                "lang": "kk",
                "model": "a:b",
            },
            {
                **{name: 0.5 for name in features.TEXT_FEATURES},
                "lang": "kk",
                "model": "a_b",
            },
        ]
    )
    with pytest.raises(ValueError, match="sanitized feature name"):
        design_matrix(frame, meta_columns=("lang", "model"))


def test_observations_are_graded_strictly_and_never_imputed(
    tmp_path: Path,
) -> None:
    frame = real_observations()
    frame.loc[0, "parse_status"] = "unparsed"
    frame.loc[0, "is_correct"] = 1
    responses = write_contract(
        tmp_path, "responses.csv", frame.drop(columns=DERIVED_ONLY)
    )
    labels = write_contract(
        tmp_path,
        "labels.csv",
        frame[
            ["model", "prompt_id", "lang", "rep", "q_id", "is_correct", "parse_status"]
        ],
    )
    data = load_observations(responses, labels)
    assert len(data) == len(frame)
    unparsed = data.query("model == @SMALL and prompt_id == 'q_001-kk' and rep == 0")
    assert len(unparsed) == 1
    assert int(unparsed["is_correct"].iloc[0]) == 0, (
        "an unparsed answer counts as wrong"
    )
    assert (data["total_tokens"] == data["input_tokens"] + data["output_tokens"]).all()

    broken = frame.copy()
    broken.loc[1, "usage_available"] = "false"
    responses = write_contract(
        tmp_path, "responses.csv", broken.drop(columns=DERIVED_ONLY)
    )
    with pytest.raises(ValueError, match="without provider usage"):
        load_observations(responses, labels)

    ungraded = frame.iloc[:-1]
    labels = write_contract(
        tmp_path,
        "labels.csv",
        ungraded[
            ["model", "prompt_id", "lang", "rep", "q_id", "is_correct", "parse_status"]
        ],
    )
    responses = write_contract(
        tmp_path, "responses.csv", frame.drop(columns=DERIVED_ONLY)
    )
    with pytest.raises(ValueError, match="grades"):
        load_observations(responses, labels)


def test_roles_are_read_from_cost_and_accuracy() -> None:
    roles = route_roles(real_observations())
    assert roles[STRATEGY_DIRECT] == SMALL, "direct is the cheapest observed model"
    assert roles[STRATEGY_STRONGER] == BIG, "stronger is the most accurate one"
    assert roles[STRATEGY_VIA_ENGLISH] == SMALL
    with pytest.raises(ValueError, match="at least two observed models"):
        route_roles(real_observations().query("model == @SMALL"))


def test_route_legs_use_the_observed_english_prompt_for_via_english() -> None:
    routes = build_route_rows(real_observations(), real_prompts())
    assert list(routes.columns) == list(ROUTE_ROW_COLUMNS)
    assert len(routes) == 6 * 3, "three strategies for every question and language"
    assert set(routes["strategy"]) == {
        STRATEGY_DIRECT,
        STRATEGY_VIA_ENGLISH,
        STRATEGY_STRONGER,
    }
    assert (routes["translation_tokens"] == 0).all()
    assert (
        routes["total_tokens"] == routes["input_tokens"] + routes["output_tokens"]
    ).all()
    assert (routes["source"] == features.DERIVED_SOURCE).all()

    kk = routes[(routes["q_id"] == "q_001") & (routes["lang"] == "kk")].set_index(
        "strategy"
    )
    assert kk.loc[STRATEGY_DIRECT, "prompt_id"] == "q_001-kk"
    assert kk.loc[STRATEGY_DIRECT, "model"] == SMALL
    assert kk.loc[STRATEGY_DIRECT, "total_tokens"] == 15, "rep 0 of the native leg"
    assert kk.loc[STRATEGY_DIRECT, "is_correct"] == "false"
    assert kk.loc[STRATEGY_VIA_ENGLISH, "prompt_id"] == "q_001-kk", (
        "the key stays native"
    )
    assert kk.loc[STRATEGY_VIA_ENGLISH, "total_tokens"] == 15, "billed as the en leg"
    assert kk.loc[STRATEGY_VIA_ENGLISH, "is_correct"] == "false"
    assert kk.loc[STRATEGY_STRONGER, "model"] == BIG
    assert kk.loc[STRATEGY_STRONGER, "total_tokens"] == 40
    assert kk.loc[STRATEGY_STRONGER, "is_correct"] == "false"
    en = routes[(routes["q_id"] == "q_001") & (routes["lang"] == "en")].set_index(
        "strategy"
    )
    assert en.loc[STRATEGY_STRONGER, "is_correct"] == "true"


def test_a_missing_observation_is_an_error_not_a_gap() -> None:
    observations = real_observations(reps=(1,))
    with pytest.raises(ValueError, match="no observation with rep=0"):
        build_route_rows(observations, real_prompts())
    partial = real_observations()
    partial = partial[partial["prompt_id"] != "q_002-en"]
    with pytest.raises(ValueError, match="no rep=0 observation"):
        build_route_rows(partial, real_prompts())
    prompts = real_prompts()
    with pytest.raises(ValueError, match="English prompt"):
        build_route_rows(real_observations(), prompts[prompts["lang"] != "en"])


def test_repeated_observations_do_not_double_the_routes() -> None:
    """Regression: real labels carry two reps, so risk_oof repeats every pair."""
    observations = real_observations()
    routes = build_route_rows(observations, real_prompts())
    prompts = real_prompts()
    cost_oof = pd.DataFrame(
        {
            "model": ["lightgbm"] * 6,
            "tokenizer": ["cl100k_base"] * 6,
            "prompt_id": list(prompts["prompt_id"]),
            "prediction": [20.0] * 6,
        }
    )
    risk_rows = [
        {
            "model": "logistic",
            "prompt_id": row["prompt_id"],
            "answer_model": model,
            "risk": 0.4,
        }
        for model in (SMALL, BIG)
        for _, row in prompts.iterrows()
        for _rep in (0, 1)
    ]
    risk_oof = pd.DataFrame(risk_rows)
    prepared = prepare_routes(
        routes, prompts, cost_oof, risk_oof, "lightgbm", "cl100k_base", "logistic"
    )
    assert len(prepared) == len(routes), "one row per route leg, not per repetition"

    conflicting = risk_oof.copy()
    conflicting.loc[0, "risk"] = 0.9
    with pytest.raises(ValueError, match="disagree within pairs"):
        prepare_routes(
            routes,
            prompts,
            cost_oof,
            conflicting,
            "lightgbm",
            "cl100k_base",
            "logistic",
        )


def test_routes_stay_on_the_fixture_without_the_real_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The committed derived file alone must not make a hidden run look real."""
    for name in ("PROMPTS_PATH", "LABELS_PATH", "RESPONSES_PATH", "ROUTES_PATH"):
        monkeypatch.setattr(features, name, tmp_path / f"absent-{name}.csv")
    assert not real_routes_available()
    path, is_fixture, source = ensure_routes()
    assert is_fixture is True and source == "fixture"
    assert path.name == "routes_fixture.csv"


def test_routes_are_derived_when_the_real_observations_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observations = real_observations()
    prompts_path = write_contract(tmp_path, "prompts.csv", real_prompts())
    responses_path = write_contract(
        tmp_path,
        "responses.csv",
        observations.drop(columns=["is_correct", "parse_status"]),
    )
    labels_path = write_contract(
        tmp_path,
        "labels.csv",
        observations[
            ["model", "prompt_id", "lang", "rep", "q_id", "is_correct", "parse_status"]
        ],
    )
    out = tmp_path / "routes_derived.csv"
    for name, value in (
        ("PROMPTS_PATH", prompts_path),
        ("LABELS_PATH", labels_path),
        ("RESPONSES_PATH", responses_path),
        ("ROUTES_PATH", tmp_path / "absent-routes.csv"),
        ("DERIVED_ROUTES_PATH", out),
    ):
        monkeypatch.setattr(features, name, value)
    assert real_routes_available()
    path, is_fixture, source = ensure_routes()
    assert path == out and is_fixture is False and source == features.DERIVED_SOURCE
    written = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert len(written) == 18
    assert set(written["source"]) == {features.DERIVED_SOURCE}

    observed = write_contract(tmp_path, "routes.csv", written)
    monkeypatch.setattr(features, "ROUTES_PATH", observed)
    path, is_fixture, source = ensure_routes()
    assert path == observed and is_fixture is False and source == "observed"


def test_the_committed_real_artifacts_are_labelled_real() -> None:
    """The rerun on the real data is committed, with its sample sizes."""
    for base, expected in (("cost_metrics", 360), ("risk_metrics", 1080)):
        payload = json.loads((MODELS_DIR / f"{base}.json").read_text(encoding="utf-8"))
        assert payload["fixture"] is False, base
        assert payload["inputs"]["n_rows"] == expected, base
        assert payload["group_by"] == "q_id" and payload["seed"] == 42, base
        assert Path(payload["inputs"]["prompts"]).name == "prompts.csv", base
    router = json.loads(
        (MODELS_DIR / "router_metrics.json").read_text(encoding="utf-8")
    )
    assert router["fixture"] is False
    assert router["inputs"]["routes_source"] == features.DERIVED_SOURCE
    assert router["inputs"]["n_questions"] == 60
    assert router["inputs"]["n_keys"] == 180
    per_fold = {float(value) for value in router["thresholds_per_fold"].values()}
    assert len(router["thresholds_per_fold"]) == 5
    assert per_fold == {float(router["deployment_threshold"])}, "one rule, five folds"
    assert router["selection_rule"]
    derived = pd.read_csv(DERIVED_ROUTES_PATH, dtype=str, keep_default_na=False)
    assert len(derived) == 540
    assert set(derived["source"]) == {features.DERIVED_SOURCE}
    assert derived["q_id"].nunique() == 60


def test_the_state_resolves_every_input_as_real(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement: real files in, "Real data" out — no example badge."""
    from app import logic

    monkeypatch.delenv("LT_FORCE_EXAMPLE", raising=False)
    state = logic.load_state()
    assert state.sources["prompts"] == "real"
    assert state.sources["tokens"] == "real"
    assert state.sources["labels"] == "real"
    assert state.sources["routes"] == "derived", "no data/routes.csv is published"
    for base in ("cost_metrics", "risk_metrics", "router_metrics"):
        assert state.sources[base] == "real", base
    assert state.is_example is False
    assert state.banner is None
    assert len(state.table("labels")) == 1080
    assert len(state.table("tokens")) == 360
    assert logic.verified_share(state.table("prompts")) == 0.0


def test_the_page_shows_the_real_badge_and_the_real_panels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import streamlit
    from streamlit.testing.v1 import AppTest

    from app import tokenizers

    monkeypatch.delenv("LT_FORCE_EXAMPLE", raising=False)
    monkeypatch.delenv("LT_FORCE_EMPTY", raising=False)
    monkeypatch.setattr(
        tokenizers, "load_backends", lambda: tokenizers.BackendReport({}, {})
    )
    streamlit.cache_resource.clear()
    test = AppTest.from_file(
        str(Path(__file__).resolve().parents[2] / "app" / "main.py"),
        default_timeout=240,
    )
    test.run()
    assert not test.exception
    badges = [
        item.value for item in test.markdown if '<span class="badge' in item.value
    ]
    assert len(badges) == 1 and "badge-real" in badges[0]
    assert "Real data" in badges[0]
    metrics = {metric.label: metric.value for metric in test.metric}
    assert float(metrics["Tokens"]) > 0.0
    assert metrics["Route"] in {"direct", "via_english", "stronger_model"}
    assert "chart_router_curve" in test.session_state["rendered_charts"]
    tables = [item.value.to_string() for item in test.dataframe]
    assert any("derived" in table for table in tables), "provenance stays visible"
