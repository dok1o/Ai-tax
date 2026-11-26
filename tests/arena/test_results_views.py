"""Tests for the Results and Method views behind the T-011 page.

Every table the page shows is built by a pure function in ``app.logic``: ratios,
bootstrap intervals, the playground join, the router surface and the provenance
tables. Fixture-backed artifacts are only used where a real join is required.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app import logic
from src.common.config import MODELS_DIR
from src.features import LANGS
from src.router import threshold_grid

TOKENS = pd.DataFrame(
    {
        "prompt_id": ["q0-en", "q0-kk", "q0-ru", "q1-en", "q1-kk", "q2-kk"],
        "q_id": ["q0", "q0", "q0", "q1", "q1", "q2"],
        "lang": ["en", "kk", "ru", "en", "kk", "kk"],
        "tokenizer": ["t1"] * 5 + ["t2"],
        "token_count": ["20", "30", "25", "0", "12", "8"],
    }
)
LABELS = pd.DataFrame(
    {
        "q_id": ["q0", "q0", "q1", "q1"],
        "lang": ["kk"] * 2 + ["en"] * 2,
        "is_correct": ["true", "false", "true", "true"],
    }
)


@pytest.fixture(scope="module")
def calibrated_routes() -> pd.DataFrame:
    """Held-out route rows exactly as the page builds them from the artifacts."""
    routes = logic.router_evaluation(logic.load_state(MODELS_DIR))
    assert routes is not None, "run the T-005/T-007/T-008 fixtures first"
    return routes


def test_tokens_vs_en_pairs_questions_and_drops_zero_denominators() -> None:
    table = logic.tokens_vs_en_table(TOKENS)
    assert table is not None
    per_pair = {
        (row["tokenizer"], row["lang"]): row for row in table.to_dict("records")
    }
    assert per_pair[("t1", "kk")]["tokens_vs_en"] == pytest.approx(1.5)
    assert per_pair[("t1", "ru")]["tokens_vs_en"] == pytest.approx(1.25)
    # q1 has a zero English count, so its Kazakh ratio is unavailable, not infinite.
    assert per_pair[("t1", "kk")]["n"] == 1
    # t2 has no English row at all: the ratio stays unavailable and is dropped.
    assert ("t2", "en") not in per_pair
    assert logic.tokens_vs_en_table(None) is None
    assert logic.tokens_vs_en_table(TOKENS.drop(columns=["q_id"])) is None


def test_prompt_ratio_frame_keeps_one_row_per_prompt() -> None:
    frame = logic.prompt_ratio_frame(TOKENS, "t1")
    assert frame is not None
    assert list(frame["prompt_id"]) == ["q0-en", "q0-kk", "q0-ru"]
    assert list(frame["tokens_vs_en"]) == pytest.approx([1.0, 1.5, 1.25])
    assert logic.prompt_ratio_frame(TOKENS, "absent") is None
    assert logic.prompt_ratio_frame(None, "t1") is None


def test_accuracy_with_ci_brackets_the_point_estimate_per_language() -> None:
    table = logic.accuracy_with_ci(LABELS, resamples=200)
    assert table is not None
    assert list(table["lang"]) == ["en", "kk"]
    row = table[table["lang"] == "kk"].iloc[0]
    assert row["accuracy"] == pytest.approx(0.5)
    assert row["n"] == 2 and row["n_questions"] == 1
    assert row["ci_low"] <= row["accuracy"] <= row["ci_high"]
    assert table[table["lang"] == "en"].iloc[0]["accuracy"] == pytest.approx(1.0)
    assert logic.accuracy_with_ci(None) is None
    broken = LABELS.copy()
    broken["is_correct"] = "maybe"
    assert logic.accuracy_with_ci(broken) is None


def test_bootstrap_interval_is_reproducible_and_question_grouped() -> None:
    values = pd.Series([1.0, 0.0, 1.0, 1.0]).to_numpy()
    questions = pd.Series(["q0", "q0", "q1", "q1"]).to_numpy()
    first = logic.bootstrap_ci_by_question(values, questions, resamples=200)
    second = logic.bootstrap_ci_by_question(values, questions, resamples=200)
    assert first == second
    mean, low, high = first
    assert mean == pytest.approx(0.75)
    assert low <= mean <= high


def test_router_surface_frame_covers_the_whole_threshold_grid(
    calibrated_routes: pd.DataFrame,
) -> None:
    frame = logic.router_surface_frame(calibrated_routes)
    assert frame is not None
    assert sorted(frame["threshold"].unique()) == pytest.approx(threshold_grid())
    assert set(frame["lang"]) <= set(LANGS)
    assert (frame["n"] > 0).all()
    assert frame["accuracy"].between(0.0, 1.0).all()
    assert frame["saved_tokens"].notna().all()
    # Every threshold has one row per language, so the surface grid is complete.
    counts = frame.groupby("threshold")["lang"].nunique()
    assert (counts == counts.iloc[0]).all()
    assert logic.router_surface_frame(None) is None


def test_router_curve_frames_name_the_two_baselines_the_card_asks_for(
    calibrated_routes: pd.DataFrame,
) -> None:
    frames = logic.router_curve_frames(calibrated_routes)
    assert frames is not None
    curve, baselines = frames
    assert {"threshold", "n", "cost_tokens_mean", "accuracy"} <= set(curve.columns)
    assert len(curve) == len(threshold_grid())
    assert list(baselines["policy"]) == [
        "always_direct",
        "always_via_english",
        "always_stronger_model",
    ]
    assert list(logic.curve_baselines(baselines)["policy"]) == [
        "always_direct",
        "always_via_english",
    ]
    # The always-via-English baseline must cost more tokens than always-direct.
    costs = baselines.set_index("policy")["cost_tokens_mean"]
    assert costs["always_via_english"] > costs["always_direct"]
    assert logic.router_curve_frames(None) is None


def test_route_reason_names_the_strategy_and_the_threshold() -> None:
    options = pd.DataFrame(
        {
            "strategy": ["direct", "via_english"],
            "risk_hat": [0.30, 0.10],
            "cost_hat": [90.0, 170.0],
            "chosen": [True, False],
        }
    )
    reason = logic.route_reason(options, 0.95)
    assert reason.startswith("direct:")
    assert "0.95" in reason and "90" in reason
    strict = options.copy()
    strict["chosen"] = [False, True]
    assert logic.route_reason(strict, 0.05).startswith("via_english:")
    empty = options.iloc[0:0].assign(chosen=[])
    assert "no strategy" in logic.route_reason(empty, 0.95)


def test_verified_share_reads_the_contract_column() -> None:
    prompts = pd.DataFrame({"verified_by_native": ["true", "false", "TRUE", "0"]})
    assert logic.verified_share(prompts) == pytest.approx(0.5)
    assert logic.verified_share(pd.DataFrame({"prompt_id": ["a"]})) is None
    assert logic.verified_share(None) is None


def test_frame_is_synthetic_reads_row_level_provenance() -> None:
    assert logic.frame_is_synthetic(pd.DataFrame({"source": ["fixture"]}))
    assert logic.frame_is_synthetic(pd.DataFrame({"model": ["fixture-lm"]}))
    assert logic.frame_is_synthetic(pd.DataFrame({"tokenizer": ["fixture/tok"]}))
    assert not logic.frame_is_synthetic(pd.DataFrame({"model": ["qwen"]}))
    assert not logic.frame_is_synthetic(pd.DataFrame())
    assert not logic.frame_is_synthetic(None)


def test_method_tables_report_sources_versions_and_limitations() -> None:
    state = logic.load_state(MODELS_DIR)
    tables = logic.method_tables(state)
    sources = tables["sources"].set_index("input")
    assert {"prompts", "tokens", "labels", "routes"} <= set(sources.index)
    assert {"cost_metrics", "risk_metrics", "router_metrics"} <= set(sources.index)
    assert sources.loc["tokens", "source"] in {"fixture", "real", "missing", "demo"}
    assert int(sources.loc["tokens", "rows"]) >= 0
    versions = tables["versions"]
    assert not versions.empty
    assert {"task", "generated_at", "python", "fixture"} <= set(versions.columns)
    assert all(isinstance(value, str) for value in versions["generated_at"])
    assert len(tables["limitations"]) == len(logic.METHOD_LIMITATIONS)


def test_observed_tokenizers_lists_the_tokens_table() -> None:
    assert logic.observed_tokenizers(TOKENS) == ["t1", "t2"]
    assert logic.observed_tokenizers(None) == []
    assert logic.observed_tokenizers(TOKENS.drop(columns=["tokenizer"])) == []


def test_derived_missing_explains_what_the_view_needs() -> None:
    message = logic.derived_missing("router evaluation")
    assert "router evaluation" in message
    assert "saved models" in message
    assert "network" not in message


def test_join_prompts_falls_back_to_the_fixture_identifiers() -> None:
    state = logic.load_state(MODELS_DIR)
    prompts = state.join_prompts()
    assert prompts is not None
    if state.sources["tokens"] != "real":
        tokens = state.table("tokens")
        assert tokens is not None
        assert set(prompts["prompt_id"]) >= set(tokens["prompt_id"])


def test_playground_and_evaluation_frames_come_from_the_saved_artifacts(
    monkeypatch,
) -> None:
    monkeypatch.setenv(logic.FORCE_EXAMPLE_ENV, "1")
    state = logic.load_state(MODELS_DIR)
    frame = logic.playground_frame(state)
    assert frame is not None
    assert {"prompt_id", "lang", "category", "tokens_vs_en", "cost", "risk"} <= set(
        frame.columns
    )
    assert set(frame["lang"]) <= set(LANGS)
    assert frame["risk"].between(0.0, 1.0).all()
    assert (frame["cost"] > 0).all()
    assert not frame.isna().any().any()
    routes = logic.router_evaluation(state)
    assert routes is not None and {"risk_hat", "cost_hat"} <= set(routes.columns)
    surface = logic.router_surface_frame(routes)
    assert surface is not None and not surface.empty
