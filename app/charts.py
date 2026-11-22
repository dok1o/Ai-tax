"""Plotly figure builders for the dashboard (T-011).

Every builder is a pure function that returns a ``plotly.graph_objects.Figure``
so tests can assert trace types and encodings without Streamlit. Axes carry
units, every title carries the sample size n, and fixture-derived data is
labelled in the title as well as by the page banner.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

FIXTURE_TAG = " — FIXTURE data, not a result"
SURFACE_HOVER = (
    "threshold %{x}<br>saved %{y:.1f} tokens<br>accuracy %{z:.3f}<extra></extra>"
)
LANG_ORDER = ("kk", "ru", "en")
LANG_COLORS = {"kk": "#00AFB9", "ru": "#E8A33D", "en": "#4C6EF5"}


def title_with_n(base: str, n: int | None, fixture: bool) -> str:
    """Chart title with the sample size and the fixture label."""
    suffix = "" if n is None else f" (n={n})"
    return f"{base}{suffix}{FIXTURE_TAG if fixture else ''}"


def _lang_colors(langs: Sequence[str]) -> list[str]:
    return [LANG_COLORS.get(str(lang), "#868E96") for lang in langs]


def tokens_by_language(table: pd.DataFrame, tokenizer: str, fixture: bool) -> go.Figure:
    """Mean measured token count per language for one tokenizer."""
    frame = table[table["tokenizer"] == tokenizer]
    n = int(frame["n"].sum()) if "n" in frame and not frame.empty else None
    figure = go.Figure(
        go.Bar(
            x=frame["lang"],
            y=frame["tokens_mean"],
            marker_color=_lang_colors(frame["lang"]),
            name="mean tokens",
            hovertemplate="lang %{x}<br>%{y:.2f} tokens<extra></extra>",
        )
    )
    figure.update_layout(
        title=title_with_n(f"Prompt tokens per language, {tokenizer}", n, fixture),
        xaxis_title="language",
        yaxis_title="mean tokens per prompt [tokens]",
    )
    return figure


def token_ratios(table: pd.DataFrame, tokenizer: str, fixture: bool) -> go.Figure:
    """tokens_vs_en per language: the same question in Kazakh against English."""
    frame = table[table["tokenizer"] == tokenizer]
    n = int(frame["n"].sum()) if "n" in frame and not frame.empty else None
    figure = go.Figure(
        go.Bar(
            x=frame["lang"],
            y=frame["tokens_vs_en"],
            marker_color=_lang_colors(frame["lang"]),
            name="tokens_vs_en",
            hovertemplate="lang %{x}<br>ratio %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title_with_n(f"tokens_vs_en per language, {tokenizer}", n, fixture),
        xaxis_title="language",
        yaxis_title="tokens / English tokens [ratio, en = 1.0]",
    )
    return figure


# The eight symbols scatter3d accepts; 2D scatter takes the same set.
SYMBOLS = (
    "circle",
    "square",
    "diamond",
    "cross",
    "x",
    "circle-open",
    "square-open",
    "diamond-open",
)


def _lang_codes(frame: pd.DataFrame) -> tuple[list[int], list[str], list[float]]:
    """Numeric language codes plus the tick labels and colorscale stops they need."""
    langs = [lang for lang in LANG_ORDER if lang in set(frame["lang"])]
    extra = sorted(set(frame["lang"]) - set(langs))
    langs += extra
    positions = [index / max(len(langs) - 1, 1) for index in range(len(langs))]
    codes = frame["lang"].map({lang: index for index, lang in enumerate(langs)})
    return codes.astype(int).tolist(), langs, positions


def _symbols(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """One marker symbol per category, plus the legend labels."""
    categories = sorted(frame["category"].astype(str).unique())
    mapping = {
        category: SYMBOLS[index % len(SYMBOLS)]
        for index, category in enumerate(categories)
    }
    return frame["category"].astype(str).map(mapping).tolist(), categories


def scatter3d(frame: pd.DataFrame, fixture: bool) -> go.Figure:
    """Predicted cost against predicted risk against the English token ratio."""
    codes, langs, positions = _lang_codes(frame)
    symbols, categories = _symbols(frame)
    figure = go.Figure(
        go.Scatter3d(
            x=frame["tokens_vs_en"],
            y=frame["risk"],
            z=frame["cost"],
            mode="markers",
            marker={
                "size": 5,
                "color": codes,
                "colorscale": [
                    [position, LANG_COLORS.get(lang, "#888888")]
                    for position, lang in zip(positions, langs, strict=False)
                ],
                "showscale": True,
                "colorbar": {
                    "title": {"text": "language"},
                    "tickvals": list(range(len(langs))),
                    "ticktext": langs,
                },
                "symbol": symbols,
                "line": {"width": 0},
            },
            hovertext=frame["prompt_id"],
            hoverinfo="text+x+y+z",
            name="prompts",
        )
    )
    figure.update_layout(
        title=title_with_n(
            "Prompts: token ratio, predicted risk, predicted cost", len(frame), fixture
        ),
        scene={
            "xaxis": {"title": "tokens_vs_en [ratio to English]"},
            "yaxis": {"title": "predicted P(incorrect) [probability]"},
            "zaxis": {"title": "predicted cost [tokens]"},
        },
        annotations=[
            {
                "text": "symbols: " + ", ".join(categories),
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.06,
                "xanchor": "left",
                "font": {"size": 10},
            }
        ],
    )
    return figure


def scatter3d_2d(frame: pd.DataFrame, fixture: bool) -> go.Figure:
    """2D toggle for the 3D prompt view: risk against cost, coloured by language."""
    _, categories = _symbols(frame)
    symbol_of = {
        category: SYMBOLS[index % len(SYMBOLS)]
        for index, category in enumerate(categories)
    }
    figure = go.Figure()
    for lang in [lang for lang in LANG_ORDER if lang in set(frame["lang"])] + sorted(
        set(frame["lang"]) - set(LANG_ORDER)
    ):
        block = frame[frame["lang"] == lang]
        if block.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=block["risk"],
                y=block["cost"],
                mode="markers",
                marker={
                    "color": LANG_COLORS.get(lang, "#888888"),
                    "symbol": block["category"].astype(str).map(symbol_of).tolist(),
                    "size": 8,
                },
                hovertext=block["prompt_id"],
                hoverinfo="text+x+y",
                name=str(lang),
            )
        )
    figure.update_layout(
        title=title_with_n(
            "Prompts in 2D: predicted risk against cost", len(frame), fixture
        ),
        xaxis_title="predicted P(incorrect) [probability]",
        yaxis_title="predicted cost [tokens]",
    )
    return figure


def router_surface(frame: pd.DataFrame, fixture: bool) -> go.Figure:
    """Observed router evaluation: threshold, saved tokens and accuracy."""
    langs = list(frame["lang"].unique())
    thresholds = sorted(frame["threshold"].unique())
    grid = frame.pivot_table(index="lang", columns="threshold", values="saved_tokens")
    accuracy = frame.pivot_table(index="lang", columns="threshold", values="accuracy")
    figure = go.Figure(
        go.Surface(
            x=[thresholds] * len(langs),
            y=grid.reindex(index=langs, columns=thresholds).to_numpy(),
            z=accuracy.reindex(index=langs, columns=thresholds).to_numpy(),
            hovertemplate=SURFACE_HOVER,
            name="router",
        )
    )
    figure.update_layout(
        title=title_with_n(
            "Router surface: threshold, saved tokens, accuracy",
            int(frame["n"].sum()),
            fixture,
        ),
        scene={
            "xaxis": {"title": "risk threshold [P(incorrect)]"},
            "yaxis": {"title": "saved tokens vs always_direct [tokens]"},
            "zaxis": {"title": "held-out accuracy [share correct]"},
        },
    )
    return figure


def router_surface_2d(frame: pd.DataFrame, fixture: bool) -> go.Figure:
    """2D toggle for the surface: accuracy against threshold, one line per language."""
    figure = go.Figure()
    for lang in LANG_ORDER:
        block = frame[frame["lang"] == lang].sort_values("threshold")
        if block.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=block["threshold"],
                y=block["accuracy"],
                mode="lines+markers",
                name=str(lang),
                marker={"color": LANG_COLORS.get(str(lang), "#868E96")},
                hovertemplate="threshold %{x:.2f}<br>accuracy %{y:.3f}<extra>"
                + str(lang)
                + "</extra>",
            )
        )
    figure.update_layout(
        title=title_with_n(
            "Router accuracy against threshold", int(frame["n"].sum()), fixture
        ),
        xaxis_title="risk threshold [P(incorrect)]",
        yaxis_title="held-out accuracy [share correct]",
    )
    return figure


def accuracy_with_ci(table: pd.DataFrame, fixture: bool) -> go.Figure:
    """Measured accuracy per language with bootstrap 95% intervals."""
    lower = (table["accuracy"] - table["ci_low"]).abs()
    upper = (table["ci_high"] - table["accuracy"]).abs()
    figure = go.Figure(
        go.Bar(
            x=table["lang"],
            y=table["accuracy"],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": upper,
                "arrayminus": lower,
            },
            marker_color=_lang_colors(table["lang"]),
            name="accuracy",
            hovertemplate="lang %{x}<br>%{y:.3f} correct<extra></extra>",
        )
    )
    figure.update_layout(
        title=title_with_n(
            "Accuracy per language, bootstrap 95% CI", int(table["n"].sum()), fixture
        ),
        xaxis_title="language",
        yaxis_title="share of correct answers [0..1]",
        yaxis_range=[0, 1],
    )
    return figure


def router_curve(
    curve: pd.DataFrame, baselines: pd.DataFrame, fixture: bool
) -> go.Figure:
    """Cost-quality curve against the always-direct and always-via-English baselines."""
    figure = go.Figure(
        go.Scatter(
            x=curve["accuracy"],
            y=curve["cost_tokens_mean"],
            mode="lines+markers",
            name="router sweep",
            hovertemplate="accuracy %{x:.3f}<br>%{y:.1f} tokens<extra>router</extra>",
        )
    )
    for _, row in baselines.iterrows():
        figure.add_trace(
            go.Scatter(
                x=[row["accuracy"]],
                y=[row["cost_tokens_mean"]],
                mode="markers",
                marker={"size": 11, "symbol": "x"},
                name=str(row["policy"]),
                hovertemplate="accuracy %{x:.3f}<br>%{y:.1f} tokens<extra></extra>",
            )
        )
    figure.update_layout(
        title=title_with_n(
            "Router cost-quality curve",
            int(curve["n"].iloc[0]) if len(curve) else None,
            fixture,
        ),
        xaxis_title="held-out accuracy [share correct]",
        yaxis_title="mean tokens per question [tokens]",
    )
    return figure
