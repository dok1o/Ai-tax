"""Unit tests for the pure app logic and Streamlit smoke tests (T-006)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

from app import logic, text
from src.common.config import MODELS_DIR

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app" / "main.py"


def test_resolve_table_honours_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(logic.FORCE_EMPTY_ENV, "1")
    assert logic.resolve_table("prompts") == (None, "missing")
    monkeypatch.delenv(logic.FORCE_EMPTY_ENV)
    monkeypatch.setenv(logic.FORCE_EXAMPLE_ENV, "1")
    path, source = logic.resolve_table("prompts")
    assert source == "fixture"
    assert path is not None and path.name.endswith("_fixture.csv")
    assert path.exists()


def test_load_state_labels_fixture_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(logic.FORCE_EXAMPLE_ENV, "1")
    state = logic.load_state(MODELS_DIR)
    assert state.is_example
    assert state.banner == logic.BANNER_EXAMPLE
    prompts = state.table("prompts")
    assert prompts is not None and len(prompts) == 180
    assert state.sources["cost_metrics"] in {"fixture", "real", "missing"}


def test_tokens_per_language_is_hand_checkable() -> None:
    tokens = pd.DataFrame(
        {
            "lang": ["kk", "kk", "en", "en"],
            "tokenizer": ["t1"] * 4,
            "token_count": ["10", "20", "4", "6"],
        }
    )
    table = logic.tokens_per_language(tokens)
    assert table is not None
    assert table.to_dict("records") == [
        {"lang": "en", "tokenizer": "t1", "n": 2, "tokens_mean": 5.0},
        {"lang": "kk", "tokenizer": "t1", "n": 2, "tokens_mean": 15.0},
    ]
    assert logic.tokens_per_language(None) is None


def test_accuracy_per_language_is_hand_checkable() -> None:
    labels = pd.DataFrame(
        {
            "lang": ["kk", "kk", "en", "en"],
            "model": ["m1"] * 4,
            "is_correct": ["true", "false", "true", "true"],
        }
    )
    table = logic.accuracy_per_language(labels)
    assert table is not None
    assert table.to_dict("records") == [
        {"lang": "en", "model": "m1", "n": 2, "accuracy": 1.0},
        {"lang": "kk", "model": "m1", "n": 2, "accuracy": 0.5},
    ]
    assert logic.accuracy_per_language(None) is None


def test_phrase_validation_and_features() -> None:
    with pytest.raises(ValueError, match="phrase is empty"):
        logic.phrase_frame("   ", "kk", "general")
    with pytest.raises(ValueError, match="unsupported language"):
        logic.phrase_frame("сәлем", "de", "general")
    features = logic.phrase_features(logic.EXAMPLE_PHRASES["kk"], "kk", "geography")
    assert features["kk_share"] > 0.0
    assert features["cyrillic_share"] == 1.0
    assert features["n_words"] > 0


def baseline_predictions(table: pd.DataFrame) -> pd.Series:
    """Per-language mean baseline predictions, indexed by tokenizer."""
    baseline = table[table["model"] == "lang_mean"]
    return baseline.set_index("tokenizer")["predicted_tokens"]


def test_predict_cost_uses_saved_models() -> None:
    models = logic.load_cost_models(MODELS_DIR)
    assert models, "run `python -m src.train cost` first"
    table = logic.predict_cost(logic.EXAMPLE_PHRASES["kk"], "kk", "geography", models)
    assert table is not None and len(table) == len(models)
    assert (table["predicted_tokens"] > 0).all()
    assert np.isfinite(table["predicted_tokens"].to_numpy()).all()
    english_table = logic.predict_cost(
        logic.EXAMPLE_PHRASES["en"], "en", "geography", models
    )
    assert english_table is not None
    kazakh_mean = baseline_predictions(table)
    english_mean = baseline_predictions(english_table)
    assert (kazakh_mean > english_mean).all(), (
        "the per-language mean baseline must rank Kazakh above English"
    )
    assert np.isfinite(english_table["predicted_tokens"].to_numpy()).all()
    assert logic.predict_cost("текст", "kk", "general", {}) is None


def test_empty_state_message_names_the_input() -> None:
    message = logic.empty_state("labels")
    assert "labels" in message and "network" in message


def run_app(monkeypatch: pytest.MonkeyPatch, force_empty: bool) -> AppTest:
    """Run the page in Streamlit's test harness with cleared caches."""
    streamlit.cache_resource.clear()
    if force_empty:
        monkeypatch.setenv(logic.FORCE_EMPTY_ENV, "1")
    else:
        monkeypatch.setenv(logic.FORCE_EXAMPLE_ENV, "1")
    test = AppTest.from_file(str(APP), default_timeout=120)
    test.run()
    return test


def test_app_starts_with_fixture_data(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded_before = set(sys.modules)
    test = run_app(monkeypatch, force_empty=False)
    assert not test.exception
    assert logic.BANNER_EXAMPLE in [item.value for item in test.error]
    assert test.title[0].value == "Language Tax in LLMs"
    assert len(test.dataframe) >= 1
    assert {"tiktoken", "transformers"}.isdisjoint(set(sys.modules) - loaded_before)


def test_app_starts_without_any_data(monkeypatch: pytest.MonkeyPatch) -> None:
    test = run_app(monkeypatch, force_empty=True)
    assert not test.exception
    messages = [item.value for item in test.info]
    assert any("No tokens available offline" in message for message in messages)
    assert any("No prompt predictions offline" in message for message in messages)
    assert any("No router evaluation offline" in message for message in messages)
    assert logic.BANNER_EXAMPLE in [item.value for item in test.error]


def test_risk_panel_is_enabled_with_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    test = run_app(monkeypatch, force_empty=False)
    assert not test.exception
    assert "Answering model" in [box.label for box in test.selectbox]


def test_missing_models_render_disabled_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=True)
    assert not test.exception
    assert "Answering model" not in [box.label for box in test.selectbox]
    placeholders = {box.label: box.disabled for box in test.number_input}
    assert placeholders["Predicted tokens"] is True
    assert placeholders["Predicted P(incorrect)"] is True
    strategies = {box.label: box for box in test.text_input}
    assert strategies["Chosen strategy"].disabled is True
    assert strategies["Chosen strategy"].value == "pending T-008"


def test_router_options_mark_exactly_one_choice() -> None:
    policy = logic.load_router_policy(MODELS_DIR)
    assert policy is not None, "run `python -m src.router` first"
    for lang in ("kk", "en"):
        options = logic.router_options(
            logic.EXAMPLE_PHRASES[lang], lang, "geography", policy
        )
        assert options is not None
        assert options["strategy"].tolist() == list(policy["strategies"])
        assert int(options["chosen"].sum()) == 1
        assert (options["cost_hat"] > 0).all()
        assert options["risk_hat"].between(0.0, 1.0).all()
    strict = logic.router_options(
        logic.EXAMPLE_PHRASES["kk"], "kk", "geography", policy, threshold=0.01
    )
    assert strict is not None
    assert strict.loc[strict["chosen"], "risk_hat"].iloc[0] == strict["risk_hat"].min()


def test_router_panel_is_enabled_with_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=False)
    assert not test.exception
    assert "Risk threshold" in [slider.label for slider in test.slider]
    metrics = {metric.label: metric.value for metric in test.metric}
    assert metrics["Chosen strategy"] in {"direct", "via_english", "stronger_model"}
    assert float(metrics["Predicted tokens (best T-005 model)"]) > 0.0
    reasons = [item.value for item in test.markdown]
    assert any("threshold" in reason for reason in reasons)


def test_router_panel_is_a_placeholder_without_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=True)
    assert not test.exception
    assert not test.slider
    assert any("pending T-008" in item.value for item in test.text_input)


def test_page_has_the_four_card_tabs(monkeypatch: pytest.MonkeyPatch) -> None:
    test = run_app(monkeypatch, force_empty=False)
    assert not test.exception
    assert len(test.tabs) == 4
    assert [area.label for area in test.text_area] == ["kk text", "ru text", "en text"]
    assert "Tokenizer" in [box.label for box in test.sidebar.selectbox]


def test_playground_shows_chips_counts_and_unavailable_ratios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=False)
    assert not test.exception
    chip_blocks = [item.value for item in test.markdown if "span title=" in item.value]
    assert len(chip_blocks) == 3
    assert 'title="id 0 | whitespace-fallback"' in chip_blocks[0]
    assert text.LANG_COLORS["kk"] in chip_blocks[0]
    assert text.LANG_COLORS["en"] in chip_blocks[2]
    metrics = [metric.value for metric in test.metric]
    assert metrics[0].isdigit()
    test.text_area[2].set_value("").run()
    assert not test.exception
    metrics = {metric.label: metric.value for metric in test.metric}
    assert metrics["tokens_vs_en"] == text.UNAVAILABLE
    # The empty English area loses its chips and shows the unavailable note.
    chip_blocks = [item.value for item in test.markdown if "line-height" in item.value]
    assert len(chip_blocks) == 2
    empty_blocks = [
        item.value for item in test.markdown if "empty phrase" in item.value
    ]
    assert empty_blocks and text.UNAVAILABLE in empty_blocks[0]


def test_machine_translation_stays_off_without_both_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=False)
    buttons = {button.label: button.disabled for button in test.button}
    assert buttons["Machine translate"] is True
    monkeypatch.setenv(text.LIVE_ENV, "1")
    monkeypatch.delenv(text.KEY_ENV, raising=False)
    test = run_app(monkeypatch, force_empty=False)
    assert test.button[0].disabled is True
    monkeypatch.delenv(text.LIVE_ENV, raising=False)
    monkeypatch.setenv(text.KEY_ENV, "secret")
    test = run_app(monkeypatch, force_empty=False)
    assert test.button[0].disabled is True


def test_machine_translation_runs_only_with_both_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(text.LIVE_ENV, "1")
    monkeypatch.setenv(text.KEY_ENV, "secret")
    calls: list[tuple[str, str]] = []

    def fake_translate(phrase: str, target: str, _environ=None, _opener=None) -> str:
        calls.append((phrase, target))
        return f"mock {target}"

    monkeypatch.setattr(text, "translate_text", fake_translate)
    test = run_app(monkeypatch, force_empty=False)
    assert test.button[0].disabled is False
    test.button[0].click().run()
    assert not test.exception
    assert [target for _, target in calls] == ["ru", "en"]
    assert calls[0][0] == logic.EXAMPLE_PHRASES["kk"]
    assert test.session_state["text_ru"] == "mock ru"
    assert [area.value for area in test.text_area][1:] == ["mock ru", "mock en"]
    assert any(text.TRANSLATE_LABEL in item.value for item in test.caption), (
        "machine output must be labelled"
    )


def test_sidebar_toggle_switches_both_3d_views_to_2d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=False)
    assert "chart_prompts_3d" in test.session_state["rendered_charts"]
    assert "chart_surface_3d" in test.session_state["rendered_charts"]
    test.sidebar.radio[0].set_value("2D").run()
    assert not test.exception
    rendered = test.session_state["rendered_charts"]
    assert "chart_prompts_2d" in rendered
    assert "chart_surface_2d" in rendered
    assert "chart_prompts_3d" not in rendered


def test_recorded_timings_stay_inside_the_card_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=False)
    assert not test.exception
    assert (
        float(test.session_state["cold_start_seconds"])
        < logic.COLD_START_BUDGET_SECONDS
    )
    assert float(test.session_state["tokenize_seconds"]) < logic.TOKENIZE_BUDGET_SECONDS
    captions = [item.value for item in test.sidebar.caption]
    assert any("cold start to interactive results" in item for item in captions)
    method_notes = [item.value for item in test.markdown]
    assert any("cold offline start" in item for item in method_notes)
    assert any("single-phrase tokenization" in item for item in method_notes)
