"""Pure app logic plus offline Streamlit runs of the redesigned page (T-016)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

from app import charts, logic, text, tokenizers
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


FAKE_ID = "fake/cl100k"
ABSENT = tokenizers.BackendReport(
    backends={},
    problems={
        "cl100k_base": "the cl100k_base BPE file is not in /tmp/data-gym-cache",
        "Qwen/Qwen2.5-0.5B@5796c1d": "not in the Hugging Face cache; the app "
        "never downloads, so this tokenizer stays disabled",
    },
)


def fake_report() -> tokenizers.BackendReport:
    """A backend that needs no network: one id per whitespace word."""

    def encode(phrase: str) -> list[tuple[int, str]]:
        return [(700 + index, piece) for index, piece in enumerate(phrase.split())]

    return tokenizers.BackendReport(
        backends={FAKE_ID: tokenizers.Backend(FAKE_ID, encode)}, problems={}
    )


def run_app(
    monkeypatch: pytest.MonkeyPatch,
    force_empty: bool,
    report: tokenizers.BackendReport | None = None,
    diagnostic: bool = False,
) -> AppTest:
    """Run the page in Streamlit's test harness with cleared caches."""
    streamlit.cache_resource.clear()
    if force_empty:
        monkeypatch.setenv(logic.FORCE_EMPTY_ENV, "1")
    else:
        monkeypatch.setenv(logic.FORCE_EXAMPLE_ENV, "1")
    monkeypatch.setattr(
        tokenizers, "load_backends", lambda: ABSENT if report is None else report
    )
    test = AppTest.from_file(str(APP), default_timeout=180)
    test.run()
    if diagnostic:
        word_split(test).set_value(True).run()
    return test


def word_split(test: AppTest):
    """The sidebar switch for the labelled word-split diagnostic."""
    return next(
        toggle
        for toggle in test.sidebar.toggle
        if toggle.label.startswith("Word split")
    )


def chip_blocks(test: AppTest) -> list[str]:
    return [item.value for item in test.markdown if 'class="chip"' in item.value]


def all_text(test: AppTest) -> str:
    parts = [item.value for item in test.markdown]
    parts += [item.value for item in test.caption]
    parts += [item.value for item in test.warning]
    parts += [item.value for item in test.info]
    parts += [item.value for item in test.error]
    return "\n".join(parts)


def test_page_shows_one_slim_badge_instead_of_two_banners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=False)
    assert not test.exception
    assert not test.error, "the duplicated banner box must be gone"
    badges = [
        item.value for item in test.markdown if '<span class="badge' in item.value
    ]
    assert len(badges) == 1, "one slim badge, not the two old boxes"
    assert "badge-example" in badges[0] and "Example data" in badges[0]
    assert logic.BANNER_EXAMPLE not in all_text(test)


def test_badge_switches_to_real_data_for_real_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataclasses

    real = dataclasses.replace(
        logic.load_state(MODELS_DIR),
        sources={key: "real" for key in logic.load_state(MODELS_DIR).sources},
    )
    monkeypatch.setattr(logic, "load_state", lambda _models_dir=MODELS_DIR: real)
    test = run_app(monkeypatch, force_empty=False)
    assert not test.exception
    badges = [
        item.value for item in test.markdown if '<span class="badge' in item.value
    ]
    assert len(badges) == 1 and "badge-real" in badges[0]
    assert "Real data" in badges[0]
    assert charts.FIXTURE_TAG not in all_text(test)


def test_page_has_the_four_card_tabs(monkeypatch: pytest.MonkeyPatch) -> None:
    test = run_app(monkeypatch, force_empty=False, report=fake_report())
    assert not test.exception
    assert len(test.tabs) == 4
    assert [area.label for area in test.text_area] == ["kk text", "ru text", "en text"]
    labels = [box.label for box in test.sidebar.selectbox]
    assert labels == ["Playground tokenizer", "Chart tokenizer"]


def test_real_backend_chips_carry_token_ids_and_no_word_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=False, report=fake_report())
    assert not test.exception
    assert chip_blocks(test), "a loaded backend must tokenize the three phrases"
    assert 'title="position 0 | token id 700 | fake/cl100k"' in chip_blocks(test)[0]
    assert tokenizers.WORD_SPLIT_LABEL not in all_text(test)
    assert not test.warning
    box = test.sidebar.selectbox[0]
    assert box.options == [FAKE_ID] and box.disabled is False
    cards = [item.value for item in test.markdown if 'class="metric-big"' in item.value]
    assert len(cards) == 3
    assert text.LANG_COLORS["kk"] in cards[0]
    assert text.LANG_COLORS["ru"] in cards[1]
    assert text.LANG_COLORS["en"] in cards[2]
    assert "x1.00" in cards[2], "English is its own denominator"
    assert "words" not in test.session_state["rendered_charts"]


def test_missing_backends_are_disabled_and_nothing_is_tokenized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=False)
    assert not test.exception
    box = test.sidebar.selectbox[0]
    assert box.disabled is True and box.options == ["none available offline"]
    disabled = {toggle.label for toggle in test.sidebar.toggle if toggle.disabled}
    assert disabled == set(ABSENT.problems)
    warnings = [item.value for item in test.sidebar.warning]
    assert any("unavailable offline" in item for item in warnings)
    assert any("never" in item.value for item in test.sidebar.caption)
    assert "not in" in all_text(test)
    assert not chip_blocks(test), "no backend means no chips, not word chips"
    assert any("nothing is tokenized" in item.value for item in test.info)
    placeholders = [
        item.value
        for item in test.markdown
        if '<span class="chip-empty">' in item.value
    ]
    assert len(placeholders) == 3, "each language card shows the unavailable notice"


def test_word_split_diagnostic_is_labelled_words_not_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=False, diagnostic=True)
    assert not test.exception
    assert chip_blocks(test)
    assert f"position 0 | {tokenizers.WORD_SPLIT_LABEL}" in chip_blocks(test)[0]
    assert all("token id" not in block for block in chip_blocks(test))
    assert any(tokenizers.WORD_SPLIT_LABEL in item.value for item in test.warning), (
        "the diagnostic must announce itself"
    )
    corners = [
        item.value
        for item in test.markdown
        if '<span class="count-corner">' in item.value
    ]
    assert corners and all("words" in item for item in corners)


def test_model_row_shows_cost_risk_and_route_with_a_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=False, report=fake_report())
    assert not test.exception
    metrics = {metric.label: metric.value for metric in test.metric}
    assert float(metrics["Tokens"]) > 0.0
    assert 0.0 <= float(metrics["P(incorrect)"]) <= 1.0
    assert metrics["Route"] in {"direct", "via_english", "stronger_model"}
    assert "Risk threshold" in [slider.label for slider in test.slider]
    assert "Answering model" in [box.label for box in test.selectbox]
    reasons = [item.value for item in test.markdown]
    assert any("threshold" in item for item in reasons)


def test_missing_models_render_disabled_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=True, report=fake_report())
    assert not test.exception
    assert "Answering model" not in [box.label for box in test.selectbox]
    placeholders = {box.label: box.disabled for box in test.number_input}
    assert placeholders["Predicted tokens"] is True
    assert placeholders["Predicted P(incorrect)"] is True
    strategies = {box.label: box for box in test.text_input}
    assert strategies["Chosen strategy"].disabled is True
    assert strategies["Chosen strategy"].value == "pending T-008"
    messages = [item.value for item in test.info]
    assert any("No tokens available offline" in item for item in messages)
    assert any("No prompt predictions offline" in item for item in messages)
    assert any("No router evaluation offline" in item for item in messages)
    labels = {metric.label for metric in test.metric}
    assert "Route" not in labels and "Tokens" not in labels


def test_view_toggle_switches_both_3d_views_to_2d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=False, report=fake_report())
    assert "chart_prompts_3d" in test.session_state["rendered_charts"]
    assert "chart_surface_3d" in test.session_state["rendered_charts"]
    assert "chart_phrase_bars" in test.session_state["rendered_charts"]
    test.radio[0].set_value("2D").run()
    assert not test.exception
    rendered = test.session_state["rendered_charts"]
    assert "chart_prompts_2d" in rendered and "chart_surface_2d" in rendered
    assert "chart_prompts_3d" not in rendered


def test_machine_translation_stays_off_without_both_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for environ in (
        {},
        {text.LIVE_ENV: "1"},
        {text.KEY_ENV: "secret"},
        {text.LIVE_ENV: "0", text.KEY_ENV: "secret"},
    ):
        monkeypatch.delenv(text.LIVE_ENV, raising=False)
        monkeypatch.delenv(text.KEY_ENV, raising=False)
        monkeypatch.setenv(text.LIVE_ENV, environ.get(text.LIVE_ENV, ""))
        if text.KEY_ENV in environ:
            monkeypatch.setenv(text.KEY_ENV, environ[text.KEY_ENV])
        test = run_app(monkeypatch, force_empty=False, report=fake_report())
        buttons = {button.label: button.disabled for button in test.button}
        assert buttons["Machine translate"] is True, environ


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
    test = run_app(monkeypatch, force_empty=False, report=fake_report())
    assert test.button[0].disabled is False
    test.button[0].click().run()
    assert not test.exception
    assert [target for _, target in calls] == ["ru", "en"]
    assert calls[0][0] == logic.EXAMPLE_PHRASES["kk"]
    assert test.session_state["text_ru"] == "mock ru"
    assert [area.value for area in test.text_area][1:] == ["mock ru", "mock en"]
    assert any(text.TRANSLATE_LABEL in item.value for item in test.caption)


def test_sidebar_timing_caption_reports_the_run_that_measured_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caption is drawn after the phrases are tokenized, not before them."""

    def slow_encode(phrase: str) -> list[tuple[int, str]]:
        time.sleep(0.005)
        return [(700 + index, piece) for index, piece in enumerate(phrase.split())]

    report = tokenizers.BackendReport(
        backends={FAKE_ID: tokenizers.Backend(FAKE_ID, slow_encode)}, problems={}
    )
    test = run_app(monkeypatch, force_empty=False, report=report)
    assert not test.exception
    measured = float(test.session_state["tokenize_seconds"])
    assert measured >= 0.005, "the fake backend must be slow enough to show up"
    captions = [item.value for item in test.sidebar.caption]
    assert any(f"{measured:.4f} s" in item for item in captions), captions


def test_recorded_timings_stay_inside_the_card_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=False, report=fake_report())
    assert not test.exception
    cold = float(test.session_state["cold_start_seconds"])
    phrase = float(test.session_state["tokenize_seconds"])
    assert cold < logic.COLD_START_BUDGET_SECONDS
    assert phrase < logic.TOKENIZE_BUDGET_SECONDS
    assert any("cold start" in item.value for item in test.sidebar.caption), (
        "the sidebar must record both timings"
    )
    method_notes = [item.value for item in test.markdown]
    assert any("cold offline start" in item for item in method_notes)
    assert any("single-phrase tokenization" in item for item in method_notes)


def test_method_tab_reports_the_tokenizer_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = run_app(monkeypatch, force_empty=False)
    assert not test.exception
    tables = [item.value.to_string() for item in test.dataframe]
    assert any("disabled" in table for table in tables)
    assert "Tokenizers" in [item.value for item in test.subheader]


def test_plotly_template_matches_the_page_theme() -> None:
    charts.apply_theme()
    import plotly.io as pio

    template = pio.templates[pio.templates.default]
    assert template.layout.paper_bgcolor == charts.PAGE["panel"]
    assert template.layout.font.color == charts.PAGE["text"]
    assert list(template.layout.colorway) == [
        charts.LANG_COLORS[lang] for lang in charts.LANG_ORDER
    ]
