"""Unit tests for the pure app logic and Streamlit smoke tests (T-006)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

from app import logic
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
    test = run_app(monkeypatch, force_empty=False)
    assert not test.exception
    assert logic.BANNER_EXAMPLE in [item.value for item in test.error]
    assert test.title[0].value == "Language Tax in LLMs"
    assert len(test.dataframe) >= 1
    assert "tiktoken" not in sys.modules and "transformers" not in sys.modules


def test_app_starts_without_any_data(monkeypatch: pytest.MonkeyPatch) -> None:
    test = run_app(monkeypatch, force_empty=True)
    assert not test.exception
    messages = [item.value for item in test.info]
    assert any("No tokens available offline" in message for message in messages)
    assert any("No cost model available offline" in message for message in messages)


def test_risk_panel_is_enabled_with_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    test = run_app(monkeypatch, force_empty=False)
    assert not test.exception
    assert "Answering model" in [box.label for box in test.selectbox]


def test_risk_panel_is_a_placeholder_without_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=True)
    assert not test.exception
    assert "Answering model" not in [box.label for box in test.selectbox]
    assert len(test.number_input) == 1
