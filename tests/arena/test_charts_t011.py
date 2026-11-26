"""Figure tests for the T-011 charts: trace types, encodings, labels and n.

The builders are pure, so every assertion runs without Streamlit and without
touching the network.
"""

from __future__ import annotations

import pandas as pd
import plotly.io as pio
import pytest

from app import charts

TOKENS = pd.DataFrame(
    {
        "tokenizer": ["t1"] * 3 + ["t2"] * 3,
        "lang": ["en", "kk", "ru"] * 2,
        "n": [60] * 6,
        "tokens_mean": [37.5, 57.3, 45.1, 40.0, 66.0, 50.0],
    }
)
RATIOS = pd.DataFrame(
    {
        "tokenizer": ["t1"] * 3,
        "lang": ["en", "kk", "ru"],
        "n": [60] * 3,
        "tokens_vs_en": [1.0, 1.528, 1.203],
    }
)
PROMPTS = pd.DataFrame(
    {
        "prompt_id": ["q0-kk", "q0-ru", "q0-en", "q1-kk"],
        "lang": ["kk", "ru", "en", "kk"],
        "category": ["math", "math", "geo", "geo"],
        "tokens_vs_en": [1.5, 1.2, 1.0, 1.4],
        "risk": [0.4, 0.3, 0.2, 0.5],
        "cost": [60.0, 48.0, 40.0, 56.0],
    }
)
SURFACE = pd.DataFrame(
    {
        "threshold": [0.05, 0.95] * 3,
        "lang": ["en"] * 2 + ["kk"] * 2 + ["ru"] * 2,
        "n": [60] * 6,
        "saved_tokens": [-0.5, -2.0, 1.0, 3.0, 0.5, 2.0],
        "accuracy": [0.90, 0.85, 0.60, 0.55, 0.75, 0.70],
    }
)
ACCURACY = pd.DataFrame(
    {
        "lang": ["en", "kk", "ru"],
        "n": [120] * 3,
        "n_questions": [60] * 3,
        "accuracy": [0.83, 0.57, 0.73],
        "ci_low": [0.75, 0.48, 0.64],
        "ci_high": [0.89, 0.66, 0.81],
    }
)
CURVE = pd.DataFrame(
    {
        "threshold": [0.05, 0.95],
        "n": [180, 180],
        "cost_tokens_mean": [106.9, 96.1],
        "accuracy": [0.78, 0.73],
    }
)
BASELINES = pd.DataFrame(
    {
        "policy": ["always_direct", "always_via_english"],
        "n": [180, 180],
        "cost_tokens_mean": [95.8, 170.7],
        "accuracy": [0.63, 0.75],
    }
)


def test_titles_carry_n_and_label_fixture_data() -> None:
    assert charts.title_with_n("Tokens", 180, True) == (
        f"Tokens (n=180){charts.FIXTURE_TAG}"
    )
    assert charts.title_with_n("Tokens", 180, False) == "Tokens (n=180)"
    assert charts.FIXTURE_TAG not in charts.title_with_n("Tokens", None, False)
    assert charts.title_with_n("Tokens", None, False) == "Tokens"


def test_token_chart_is_a_bar_per_language_with_units() -> None:
    figure = charts.tokens_by_language(TOKENS, "t1", True)
    assert [trace.type for trace in figure.data] == ["bar"]
    trace = figure.data[0]
    assert list(trace.x) == ["en", "kk", "ru"]
    assert list(trace.y) == [37.5, 57.3, 45.1]
    assert list(trace.marker.color) == [
        charts.LANG_COLORS["en"],
        charts.LANG_COLORS["kk"],
        charts.LANG_COLORS["ru"],
    ]
    assert "(n=180)" in figure.layout.title.text
    assert charts.FIXTURE_TAG in figure.layout.title.text
    assert "t1" in figure.layout.title.text
    assert figure.layout.yaxis.title.text == "mean tokens per prompt [tokens]"
    other = charts.tokens_by_language(TOKENS, "t2", False)
    assert charts.FIXTURE_TAG not in other.layout.title.text
    assert list(other.data[0].y) == [40.0, 66.0, 50.0]


def test_ratio_chart_plots_tokens_vs_en() -> None:
    figure = charts.token_ratios(RATIOS, "t1", False)
    assert [trace.type for trace in figure.data] == ["bar"]
    assert list(figure.data[0].y) == [1.0, 1.528, 1.203]
    assert "en = 1.0" in figure.layout.yaxis.title.text
    assert "(n=180)" in figure.layout.title.text


def test_scatter3d_encodes_language_colour_category_symbol_and_prompt_id() -> None:
    figure = charts.scatter3d(PROMPTS, True)
    assert [trace.type for trace in figure.data] == ["scatter3d"]
    trace = figure.data[0]
    assert list(trace.x) == list(PROMPTS["tokens_vs_en"])
    assert list(trace.y) == list(PROMPTS["risk"])
    assert list(trace.z) == list(PROMPTS["cost"])
    assert list(trace.hovertext) == list(PROMPTS["prompt_id"])
    assert list(trace.marker.color) == [0, 1, 2, 0]
    assert list(trace.marker.colorbar.ticktext) == ["kk", "ru", "en"]
    assert set(trace.marker.symbol) <= set(charts.SYMBOLS)
    assert trace.marker.symbol[0] != trace.marker.symbol[2]
    scene = figure.layout.scene
    assert scene.xaxis.title.text == "tokens_vs_en [ratio to English]"
    assert scene.yaxis.title.text == "predicted P(incorrect) [probability]"
    assert scene.zaxis.title.text == "predicted cost [tokens]"
    assert "(n=4)" in figure.layout.title.text
    assert charts.FIXTURE_TAG in figure.layout.title.text
    assert "math" in figure.layout.annotations[0].text


def test_scatter3d_2d_toggle_keeps_one_trace_per_language() -> None:
    figure = charts.scatter3d_2d(PROMPTS, False)
    assert [trace.type for trace in figure.data] == ["scatter"] * 3
    assert [trace.name for trace in figure.data] == ["kk", "ru", "en"]
    assert figure.data[0].marker.color == charts.LANG_COLORS["kk"]
    assert list(figure.data[0].x) == [0.4, 0.5]
    assert list(figure.data[0].y) == [60.0, 56.0]
    assert list(figure.data[0].hovertext) == ["q0-kk", "q1-kk"]
    assert charts.FIXTURE_TAG not in figure.layout.title.text
    assert figure.layout.xaxis.title.text == "predicted P(incorrect) [probability]"


def test_router_surface_maps_threshold_saved_cost_and_accuracy() -> None:
    figure = charts.router_surface(SURFACE, True)
    assert [trace.type for trace in figure.data] == ["surface"]
    trace = figure.data[0]
    assert [list(row) for row in trace.x] == [[0.05, 0.95]] * 3
    assert trace.y.shape == (3, 2)
    assert trace.z.shape == (3, 2)
    assert trace.z.tolist()[0] == [0.90, 0.85]
    assert trace.hovertemplate == charts.SURFACE_HOVER
    scene = figure.layout.scene
    assert scene.xaxis.title.text == "risk threshold [P(incorrect)]"
    assert scene.yaxis.title.text == "saved tokens vs always_direct [tokens]"
    assert scene.zaxis.title.text == "held-out accuracy [share correct]"
    assert "(n=360)" in figure.layout.title.text


def test_router_surface_2d_toggle_has_one_line_per_language() -> None:
    figure = charts.router_surface_2d(SURFACE, False)
    assert [trace.type for trace in figure.data] == ["scatter"] * 3
    assert [trace.name for trace in figure.data] == ["kk", "ru", "en"]
    assert list(figure.data[0].x) == [0.05, 0.95]
    assert list(figure.data[0].y) == [0.60, 0.55]
    assert figure.layout.yaxis.title.text == "held-out accuracy [share correct]"


def test_accuracy_chart_carries_asymmetric_bootstrap_bars() -> None:
    figure = charts.accuracy_with_ci(ACCURACY, True)
    assert [trace.type for trace in figure.data] == ["bar"]
    bars = figure.data[0].error_y
    assert bars.type == "data"
    assert bars.symmetric is False
    assert pytest.approx(list(bars.array)) == [0.06, 0.09, 0.08]
    assert pytest.approx(list(bars.arrayminus)) == [0.08, 0.09, 0.09]
    assert list(figure.data[0].y) == [0.83, 0.57, 0.73]
    assert figure.layout.yaxis.title.text == "share of correct answers [0..1]"
    assert "(n=360)" in figure.layout.title.text


def test_router_curve_is_plotted_against_the_two_baselines() -> None:
    figure = charts.router_curve(CURVE, BASELINES, True)
    assert [trace.type for trace in figure.data] == ["scatter"] * 3
    assert figure.data[0].name == "router sweep"
    assert list(figure.data[0].x) == [0.78, 0.73]
    assert list(figure.data[0].y) == [106.9, 96.1]
    assert [trace.name for trace in figure.data[1:]] == [
        "always_direct",
        "always_via_english",
    ]
    assert figure.data[1].marker.symbol == "x"
    assert figure.layout.xaxis.title.text == "held-out accuracy [share correct]"
    assert figure.layout.yaxis.title.text == "mean tokens per question [tokens]"
    assert "(n=180)" in figure.layout.title.text


PHRASE_COUNTS = {"kk": 12, "ru": 9, "en": 6}
PHRASE_RATIOS = {"kk": 2.0, "ru": 1.5, "en": 1.0}


def test_language_colours_are_shared_by_chips_bars_and_3d() -> None:
    """T-016 asks for one palette: kk teal, ru amber, en blue, everywhere."""
    from app import text

    assert charts.LANG_COLORS == text.LANG_COLORS
    assert charts.LANG_COLORS["kk"].upper() == charts.PAGE["accent"].upper()
    scatter = charts.scatter3d(PROMPTS, True)
    stops = dict(scatter.data[0].marker.colorscale)
    assert stops[0.0] == charts.LANG_COLORS["kk"]


def test_phrase_bars_show_one_tokenizer_and_the_language_colours() -> None:
    figure = charts.phrase_bars(PHRASE_COUNTS, PHRASE_RATIOS, "cl100k_base")
    assert [trace.type for trace in figure.data] == ["bar"]
    assert list(figure.data[0].x) == ["kk", "ru", "en"]
    assert list(figure.data[0].y) == [12, 9, 6]
    assert list(figure.data[0].marker.color) == [
        charts.LANG_COLORS[lang] for lang in ("kk", "ru", "en")
    ]
    assert list(figure.data[0].text) == ["x2.00", "x1.50", "x1.00"]
    assert figure.layout.title.text == "Your phrases, cl100k_base (n=3)"
    assert figure.layout.yaxis.title.text == "tokens per phrase [tokens]"
    assert figure.layout.xaxis.title.text == "language"


def test_phrase_bars_label_the_word_split_diagnostic() -> None:
    figure = charts.phrase_bars(
        PHRASE_COUNTS, {"kk": None, "ru": None, "en": 1.0}, "word-split", words=True
    )
    assert list(figure.data[0].text) == ["", "", "x1.00"]
    assert figure.layout.title.text == (
        "Your phrases, word-split — words, not tokens (n=3)"
    )
    assert figure.layout.yaxis.title.text == "words per phrase [words]"


def test_theme_template_paints_the_page_colours() -> None:
    template = charts.theme_template()
    assert template.layout.paper_bgcolor == charts.PAGE["panel"]
    assert template.layout.plot_bgcolor == charts.PAGE["panel"]
    assert template.layout.font.color == charts.PAGE["text"]
    assert template.layout.xaxis.gridcolor == charts.PAGE["grid"]
    assert list(template.layout.colorway) == [
        charts.LANG_COLORS[lang] for lang in charts.LANG_ORDER
    ]
    charts.apply_theme()
    assert pio.templates.default == "arena_dark"
    assert pio.templates["arena_dark"] is not None
