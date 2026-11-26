"""Offline trilingual dashboard (T-011, redesigned in T-016).

Start it with ``python run.py dashboard`` or
``python -m streamlit run app/main.py``. Every number comes from files on disk:
no API call, no tokenizer download, no network request. Tokenizer backends are
the real ones from T-001 when their caches exist offline; when they do not, the
page says so and disables them instead of pretending that words are tokens.

The dark theme is painted here (the CSS block plus ``charts.apply_theme``) so it
does not depend on a config file; ``app/.streamlit/config.toml`` holds the same
theme for Streamlit's own widgets and explains where Streamlit looks for it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

START_TIME = time.perf_counter()

import pandas as pd
import streamlit as st

from app import charts, logic, text, token_backends
from src.features import LANGS

TABS = ("Playground", "3D views", "Results", "Method")
TEXT_KEY = "text_{lang}"
BADGE_EXAMPLE = "Example data"
BADGE_REAL = "Real data"
LANG_NAMES = {"kk": "Kazakh", "ru": "Russian", "en": "English"}
COST_MODEL_HELP = "Pending T-005: no saved token-cost model was found offline."
RISK_MODEL_HELP = "Pending T-007: no saved error-risk model was found offline."
ROUTER_HELP = "Pending T-008: direct / via_english / stronger_model."
TRANSLATE_HELP = (
    "Off unless the operator starts the app with the live flag (LT_LIVE=1) and a "
    "translation key (LT_TRANSLATE_KEY). Nothing is sent without a button press."
)
DIAGNOSTIC_HELP = (
    "Whitespace word splitting. It is a diagnostic for tests and layout work: the "
    "numbers it produces are words, not tokens, and are labelled that way."
)
CSS = """
<style>
html, body, .stApp, [data-testid="stAppViewContainer"] { font-size: 17px; }
.stApp { background-color: #0E1116; color: #E6EDF3; }
h1 { font-size: 2.5rem; letter-spacing: -0.02em; color: #F5F8FA; }
h2, h3 { color: #F5F8FA; }
p, li, label, .stCaption { color: #D5DDE5; }
.badge { display: inline-block; padding: .25rem .8rem; border-radius: 999px;
         font-weight: 600; font-size: .95rem; letter-spacing: .01em; }
.badge-example { background: #2E2410; color: #F2C14E; border: 1px solid #E8A33D; }
.badge-real { background: #0E2C2E; color: #58E1E8; border: 1px solid #00AFB9; }
.card-head { display: flex; align-items: center; gap: .45rem; font-size: 1.05rem; }
.lang-dot { width: .75rem; height: .75rem; border-radius: 50%; display: inline-block; }
.count-corner { margin-left: auto; font-weight: 700; font-size: .95rem;
                padding: .1rem .6rem; border-radius: 999px;
                background: #0E1116; border: 1px solid #2B323B; color: #E6EDF3; }
.metric-big { font-size: 2.3rem; font-weight: 700; line-height: 1.15; }
.metric-sub { color: #9DA7B3; font-size: 1rem; }
.multiplier { margin-left: auto; font-weight: 700; font-size: 1.15rem; }
.chips { line-height: 2.3; }
.chip { display: inline-block; margin: 2px; padding: 2px 9px; border-radius: 10px;
        border: 1px solid; background: #0E1116; font-size: .95rem; }
.chip-empty { color: #9DA7B3; }
.chip-bytes { font-family: monospace; font-size: .8rem; border-style: dashed; }
[data-testid="stMetric"] { background: #161B22; border: 1px solid #2B323B;
                          border-radius: 12px; padding: 10px 14px; }
[data-testid="stMetricLabel"] { color: #9DA7B3; }
[data-testid="stMetricValue"] { color: #F5F8FA; }
</style>
"""


@st.cache_resource(show_spinner=False)
def cached_state() -> logic.AppState:
    """Resolve inputs once per server process."""
    return logic.load_state()


@st.cache_resource(show_spinner=False)
def cached_backends() -> token_backends.BackendReport:
    """Real tokenizer backends, discovered once and never downloaded."""
    return token_backends.load_backends()


@st.cache_resource(show_spinner=False)
def cached_cost_models() -> dict:
    """Saved token-cost models (T-005)."""
    return logic.load_cost_models()


@st.cache_resource(show_spinner=False)
def cached_risk_models() -> dict:
    """Saved error-risk models (T-007)."""
    return logic.load_risk_models()


@st.cache_resource(show_spinner=False)
def cached_router_policy() -> dict | None:
    """Saved routing policy (T-008)."""
    return logic.load_router_policy()


@st.cache_resource(show_spinner=False)
def cached_playground_frame() -> pd.DataFrame | None:
    """Per-prompt ratio, predicted risk and predicted cost behind the 3D view."""
    return logic.playground_frame(cached_state())


@st.cache_resource(show_spinner=False)
def cached_evaluation() -> pd.DataFrame | None:
    """Calibrated held-out route rows behind the surface and the curve."""
    return logic.router_evaluation(cached_state())


@st.cache_resource(show_spinner=False)
def cached_surface_frame() -> pd.DataFrame | None:
    """Threshold by language grid of saved tokens and accuracy."""
    return logic.router_surface_frame(cached_evaluation())


@st.cache_resource(show_spinner=False)
def cached_curve_frames() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Router sweep plus the always-use-one-strategy baselines."""
    return logic.router_curve_frames(cached_evaluation())


@st.cache_resource(show_spinner=False)
def cached_accuracy() -> pd.DataFrame | None:
    """Accuracy per language with bootstrap intervals over questions."""
    return logic.accuracy_with_ci(cached_state().table("labels"))


def note_chart(key: str) -> None:
    """Record which chart variants this run rendered, for tests and debugging."""
    rendered = st.session_state.setdefault("chart_run", [])
    if key not in rendered:
        rendered.append(key)


def finish_charts() -> None:
    """Keep only the charts of the last run, so a toggle leaves no stale keys."""
    st.session_state["rendered_charts"] = st.session_state.pop("chart_run", [])


def record_cold_start() -> float:
    """Seconds from importing this page to the first fully rendered run."""
    if "cold_start_seconds" not in st.session_state:
        st.session_state["cold_start_seconds"] = time.perf_counter() - START_TIME
    return float(st.session_state["cold_start_seconds"])


def playground_phrases() -> dict[str, str]:
    """Text of every language area, with the bundled examples filled in first."""
    for lang in LANGS:
        st.session_state.setdefault(
            TEXT_KEY.format(lang=lang), logic.EXAMPLE_PHRASES[lang]
        )
    return {lang: str(st.session_state[TEXT_KEY.format(lang=lang)]) for lang in LANGS}


def measure_phrases(
    phrases: dict[str, str], backend: token_backends.Backend | None, diagnostic: bool
) -> tuple[dict[str, list[text.Chip]], dict[str, int | None], dict[str, float | None]]:
    """Chips, token counts and tokens_vs_en for the three phrases, one timing."""
    slowest = 0.0
    chips: dict[str, list[text.Chip]] = {}
    counts: dict[str, int | None] = {}
    for lang in LANGS:
        started = time.perf_counter()
        pieces = text.token_chips(phrases[lang], lang, backend, diagnostic)
        slowest = max(slowest, time.perf_counter() - started)
        chips[lang] = pieces
        counts[lang] = len(pieces) if pieces else None
    st.session_state["tokenize_seconds"] = slowest
    st.session_state["tokenized_phrases"] = len(LANGS)
    return chips, counts, text.tokens_vs_en(counts)


def summary_card(lang: str, count: int | None, ratio: float | None) -> None:
    """One big language card: token count and the tokens_vs_en multiplier."""
    color = text.LANG_COLORS[lang]
    unit = "tokens" if count is not None else ""
    st.markdown(
        f'<div class="card-head"><span class="lang-dot" style="background:{color}">'
        f"</span><b>{LANG_NAMES[lang]} ({lang})</b>"
        f'<span class="multiplier" style="color:{color}">'
        f"{text.format_multiplier(ratio)}</span></div>"
        f'<div class="metric-big" style="color:{color}">'
        f"{count if count is not None else text.UNAVAILABLE} <span "
        f'class="metric-sub">{unit}</span></div>'
        f'<div class="metric-sub">tokens_vs_en against the English phrase</div>',
        unsafe_allow_html=True,
    )


def chip_card(
    lang: str,
    phrase: str,
    chips: list[text.Chip],
    count: int | None,
    ratio: float | None,
    words: bool,
) -> None:
    """Bordered card with the text area, the count in the corner and the chips."""
    color = text.LANG_COLORS[lang]
    unit = "words" if words else "tokens"
    with st.container(border=True):
        st.markdown(
            f'<div class="card-head"><span class="lang-dot" style="background:{color}">'
            f"</span><b>{lang}</b>"
            f'<span class="count-corner">{count if count is not None else "-"} '
            f"{unit}</span></div>",
            unsafe_allow_html=True,
        )
        st.text_area(f"{lang} text", key=TEXT_KEY.format(lang=lang), height=96)
        st.markdown(text.chips_html(chips), unsafe_allow_html=True)
        if any(chip.raw for chip in chips):
            st.caption(text.BYTE_CAPTION)
        st.caption(
            f"tokens_vs_en {text.format_ratio(ratio)} | fertility "
            f"{text.format_ratio(text.fertility(count, text.word_count(phrase)))} | "
            f"script hint {text.detect_lang(phrase)}"
        )


def translate_into(focus: str) -> None:
    """Fill the other two language areas on an explicit button press."""
    source = str(
        st.session_state.get(TEXT_KEY.format(lang=focus), logic.EXAMPLE_PHRASES[focus])
    )
    for target in [lang for lang in LANGS if lang != focus]:
        try:
            st.session_state[TEXT_KEY.format(lang=target)] = text.translate_text(
                source, target
            )
        except (PermissionError, ValueError, OSError) as error:
            st.warning(f"{target}: {error}")
            continue
        st.caption(f"{target}: {text.TRANSLATE_LABEL}")


def render_sidebar(
    state: logic.AppState, report: token_backends.BackendReport
) -> tuple[token_backends.Backend | None, bool, str | None]:
    """Tokenizer choice, dataset tokenizer, provenance and the recorded timings."""
    st.sidebar.header("Tokenizers")
    identifiers = report.identifiers
    selected = None
    if identifiers:
        choice = st.sidebar.selectbox(
            "Playground tokenizer", identifiers, index=0, key="tokenizer_choice"
        )
        selected = report.get(str(choice))
    else:
        st.sidebar.selectbox(
            "Playground tokenizer", ["none available offline"], disabled=True
        )
    for name, reason in sorted(report.problems.items()):
        st.sidebar.toggle(name, value=False, disabled=True)
        st.sidebar.caption(reason)
    diagnostic = st.sidebar.toggle(
        f"Word split ({token_backends.WORD_SPLIT_LABEL})",
        value=False,
        help=DIAGNOSTIC_HELP,
        key="diagnostic_split",
    )
    if report.problems:
        st.sidebar.warning(
            f"{len(report.problems)} of "
            f"{len(report.problems) + len(identifiers)} tokenizers are unavailable "
            "offline and stay disabled; nothing is downloaded."
        )

    st.sidebar.header("Dataset")
    observed = logic.observed_tokenizers(state.table("tokens"))
    chart_choice = None
    if observed:
        chart_choice = st.sidebar.selectbox(
            "Chart tokenizer", observed, index=0, key="chart_tokenizer_choice"
        )
    else:
        st.sidebar.selectbox("Chart tokenizer", ["none"], disabled=True)

    st.sidebar.header("Inputs")
    for kind, source in sorted(state.sources.items()):
        st.sidebar.write(f"- `{kind}`: **{source}**")
    st.sidebar.caption("Read from disk only: no API call, no download.")
    record_cold_start()
    return (
        selected,
        bool(diagnostic),
        None if chart_choice is None else str(chart_choice),
    )


def report_timings() -> None:
    """Both recorded budgets in the sidebar, after this run measured them."""
    st.sidebar.caption(
        f"cold start {record_cold_start():.2f} s "
        f"(budget {logic.COLD_START_BUDGET_SECONDS:.0f} s) | slowest phrase "
        f"{float(st.session_state.get('tokenize_seconds', 0.0)):.4f} s "
        f"(budget {logic.TOKENIZE_BUDGET_SECONDS:.0f} s), st.cache_resource"
    )


def render_models(phrase: str, lang: str, category: str, state: logic.AppState) -> None:
    """Predicted cost, error risk and the router decision, side by side."""
    cost_column, risk_column, route_column = st.columns(3)
    costs = logic.predict_cost(phrase, lang, category, cached_cost_models())
    with cost_column:
        st.markdown("**Predicted cost (T-005)**")
        if costs is None:
            st.number_input(
                "Predicted tokens", value=0.0, disabled=True, help=COST_MODEL_HELP
            )
            st.caption("Placeholder — enable after T-005 writes models/cost_*.joblib.")
        else:
            best = float(costs["predicted_tokens"].min())
            try:
                model, tokenizer = logic.best_cost_model(
                    state.manifests.get("cost_metrics")
                )
                picked = costs[
                    (costs["model"] == model) & (costs["tokenizer"] == tokenizer)
                ]
                if not picked.empty:
                    best = float(picked.iloc[0]["predicted_tokens"])
            except ValueError:
                model, tokenizer = "cheapest saved model", costs.iloc[0]["tokenizer"]
            st.metric("Tokens", round(best, 1))
            st.caption(f"{model} · {tokenizer}")
            if bool(costs["fixture"].any()):
                st.caption("fixture-trained model, not a measurement")

    risk_models = cached_risk_models()
    with risk_column:
        st.markdown("**Error risk (T-007)**")
        if not risk_models:
            st.number_input(
                "Predicted P(incorrect)",
                value=0.0,
                disabled=True,
                help=RISK_MODEL_HELP,
            )
            st.caption("Placeholder — enable after T-007 writes models/risk_*.joblib.")
        else:
            answer_model = st.selectbox(
                "Answering model", logic.answer_models(state), index=0
            )
            table = logic.predict_risk(
                phrase, lang, category, answer_model, risk_models
            )
            best_risk = float(table["risk_incorrect"].max())
            st.metric("P(incorrect)", f"{best_risk:.2f}")
            st.caption(
                f"worst of {len(table)} saved risk models · q_id-grouped folds"
                + (" · fixture-trained" if bool(table["fixture"].any()) else "")
            )

    policy = cached_router_policy()
    with route_column:
        st.markdown("**Route (T-008)**")
        if policy is None:
            st.text_input(
                "Chosen strategy",
                value="pending T-008",
                disabled=True,
                help=ROUTER_HELP,
            )
            return
        threshold = st.slider(
            "Risk threshold", 0.05, 0.95, float(policy["threshold"]), 0.05
        )
        options = logic.router_options(phrase, lang, category, policy, threshold)
        if options is None:
            st.info(logic.empty_state("cost or risk model"))
            return
        chosen = options.loc[options["chosen"], "strategy"].tolist()
        st.metric("Route", chosen[0] if chosen else "none")
        st.write(logic.route_reason(options, threshold))


def render_playground(
    state: logic.AppState,
    backend: token_backends.Backend | None,
    diagnostic: bool,
) -> None:
    """Summary cards, the bar chart beside them, chip cards and the model row."""
    phrases = playground_phrases()
    chips, counts, ratios = measure_phrases(phrases, backend, diagnostic)
    words = backend is None
    identifier = token_backends.WORD_SPLIT_ID if words else backend.identifier
    if words and not diagnostic:
        st.info(
            "No real tokenizer is available offline, so nothing is tokenized. Enable "
            "the word-split diagnostic in the sidebar to see "
            f"{token_backends.WORD_SPLIT_LABEL}."
        )
    elif words:
        st.warning(
            f"Word-split diagnostic active: every count below is "
            f"{token_backends.WORD_SPLIT_LABEL} and no chip carries a model token id."
        )

    top = st.columns([1, 1, 1, 1.7])
    for lang, column in zip(LANGS, top[:3], strict=False):
        with column:
            summary_card(lang, counts[lang], ratios[lang])
    with top[3]:
        st.plotly_chart(
            charts.phrase_bars(counts, ratios, identifier, words=words),
            key="chart_phrase_bars",
            width="stretch",
        )
        note_chart("chart_phrase_bars")

    controls = st.columns([1, 1, 1.4])
    live = text.live_translation_enabled()
    with controls[0]:
        focus = st.selectbox("Score language", list(LANGS), index=0)
    with controls[1]:
        category = st.selectbox("Category", logic.categories(state), index=0)
    with controls[2]:
        pressed = st.button(
            "Machine translate",
            disabled=not live,
            help=TRANSLATE_HELP if not live else "Fills the other two areas on press.",
        )
        if pressed:
            translate_into(focus)
        if not live:
            st.caption(TRANSLATE_HELP)

    card_columns = st.columns(len(LANGS))
    for column, lang in zip(card_columns, LANGS, strict=True):
        with column:
            chip_card(
                lang, phrases[lang], chips[lang], counts[lang], ratios[lang], words
            )
    render_models(phrases[focus], focus, category, state)


def render_3d(state: logic.AppState) -> None:
    """The two 3D views with their 2D toggle, axis units and n in every title."""
    example = state.is_example
    view = st.radio("View", ("3D", "2D"), horizontal=True, key="view_3d")
    prompts = cached_playground_frame()
    if prompts is None:
        st.info(logic.derived_missing("prompt predictions"))
    elif view == "3D":
        st.plotly_chart(
            charts.scatter3d(prompts, example), key="chart_prompts_3d", width="stretch"
        )
        note_chart("chart_prompts_3d")
    else:
        st.plotly_chart(
            charts.scatter3d_2d(prompts, example),
            key="chart_prompts_2d",
            width="stretch",
        )
        note_chart("chart_prompts_2d")

    surface = cached_surface_frame()
    if surface is None:
        st.info(logic.derived_missing("router evaluation"))
    elif view == "3D":
        st.plotly_chart(
            charts.router_surface(surface, example),
            key="chart_surface_3d",
            width="stretch",
        )
        note_chart("chart_surface_3d")
    else:
        st.plotly_chart(
            charts.router_surface_2d(surface, example),
            key="chart_surface_2d",
            width="stretch",
        )
        note_chart("chart_surface_2d")


def render_charts(state: logic.AppState, chart_tokenizer: str | None) -> None:
    """Measured token counts and ratios for the dataset tokenizer."""
    example = state.is_example
    tokens = state.table("tokens")
    per_language = logic.tokens_per_language(tokens)
    if per_language is None or chart_tokenizer is None:
        st.info(logic.empty_state("tokens"))
        return
    st.plotly_chart(
        charts.tokens_by_language(per_language, chart_tokenizer, example),
        key="chart_tokens_by_language",
        width="stretch",
    )
    note_chart("chart_tokens_by_language")
    ratios = logic.tokens_vs_en_table(tokens)
    if ratios is None:
        st.info(logic.derived_missing("token ratios"))
        return
    st.plotly_chart(
        charts.token_ratios(ratios, chart_tokenizer, example),
        key="chart_ratios_in_charts_tab",
        width="stretch",
    )
    note_chart("chart_ratios_in_charts_tab")


def render_results(state: logic.AppState, chart_tokenizer: str | None) -> None:
    """Accuracy with CIs, token ratios and the router curve against two baselines."""
    example = state.is_example
    st.subheader("Measured tokens per language")
    render_charts(state, chart_tokenizer)

    st.subheader("Accuracy per language")
    accuracy = cached_accuracy()
    if accuracy is None:
        st.info(logic.empty_state("labels"))
    else:
        st.plotly_chart(
            charts.accuracy_with_ci(accuracy, example),
            key="chart_accuracy_ci",
            width="stretch",
        )
        note_chart("chart_accuracy_ci")
        st.dataframe(accuracy, hide_index=True)
        st.caption(
            "Bootstrap 95% intervals resample whole questions (q_id), 1000 draws, "
            "seed 42."
        )

    st.subheader("Token ratios")
    ratios = logic.tokens_vs_en_table(state.table("tokens"))
    if ratios is None:
        st.info(logic.derived_missing("token ratios"))
    else:
        st.dataframe(ratios, hide_index=True)

    st.subheader("Router curve against the baselines")
    frames = cached_curve_frames()
    if frames is None:
        st.info(logic.derived_missing("router evaluation"))
        return
    curve, baselines = frames
    st.plotly_chart(
        charts.router_curve(curve, logic.curve_baselines(baselines), example),
        key="chart_router_curve",
        width="stretch",
    )
    note_chart("chart_router_curve")
    st.dataframe(
        curve[["threshold", "n", "cost_tokens_mean", "accuracy"]], hide_index=True
    )
    st.dataframe(
        baselines[["policy", "n", "cost_tokens_mean", "accuracy"]], hide_index=True
    )


def render_method(state: logic.AppState, report: token_backends.BackendReport) -> None:
    """Provenance, tokenizer status, versions, n, verified share, limitations."""
    tables = logic.method_tables(state)
    st.subheader("Tokenizers")
    st.dataframe(
        pd.DataFrame(
            [
                {"tokenizer": name, "status": "loaded", "detail": "offline cache"}
                for name in report.identifiers
            ]
            + [
                {"tokenizer": name, "status": "disabled", "detail": reason}
                for name, reason in sorted(report.problems.items())
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Data sources")
    st.dataframe(tables["sources"], hide_index=True, width="stretch")
    st.subheader("Model versions and query dates")
    if tables["versions"].empty:
        st.info(logic.empty_state("metrics manifest"))
    else:
        st.dataframe(tables["versions"], hide_index=True, width="stretch")
    st.subheader("Sample sizes")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "input": kind,
                    "n_rows": 0
                    if state.table(kind) is None
                    else len(state.table(kind)),
                    "path": state.paths.get(kind, ""),
                }
                for kind in logic.TABLES
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    share = logic.verified_share(state.table("prompts"))
    st.metric(
        "Native-verified share of prompts",
        text.UNAVAILABLE if share is None else f"{share:.3f}",
    )
    st.caption(
        "Query dates are the manifest timestamps above; no collection date is "
        "recorded inside the CSV contracts yet (T-003)."
    )
    st.subheader("Limitations")
    for note in tables["limitations"]["limitation"]:
        st.write(f"- {note}")
    st.subheader("Recorded timings")
    st.write(
        f"- cold offline start to interactive results: {record_cold_start():.2f} s "
        f"(budget {logic.COLD_START_BUDGET_SECONDS:.0f} s)"
    )
    st.write(
        f"- slowest single-phrase tokenization this run: "
        f"{float(st.session_state.get('tokenize_seconds', 0.0)):.4f} s (budget "
        f"{logic.TOKENIZE_BUDGET_SECONDS:.0f} s)"
    )


def render() -> None:
    """Build the whole page."""
    st.set_page_config(page_title="Language Tax in LLMs", layout="wide")
    charts.apply_theme()
    st.markdown(CSS, unsafe_allow_html=True)
    state = cached_state()
    report = cached_backends()
    title, badge = st.columns([4, 1])
    with title:
        st.title("Language Tax in LLMs")
    with badge:
        kind = "example" if state.is_example else "real"
        label = BADGE_EXAMPLE if state.is_example else BADGE_REAL
        st.markdown(
            f'<div style="text-align:right;padding-top:1.4rem">'
            f'<span class="badge badge-{kind}">{label}</span></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "Kazakh, Russian and English: the same question costs different amounts of "
        "tokens and gets different answers. Offline, saved results only."
    )
    st.session_state["chart_run"] = []
    backend, diagnostic, chart_tokenizer = render_sidebar(state, report)
    playground, views, results, method = st.tabs(list(TABS))
    with playground:
        render_playground(state, backend, diagnostic)
    with views:
        render_3d(state)
    with results:
        render_results(state, chart_tokenizer)
    with method:
        render_method(state, report)
    finish_charts()
    report_timings()


render()
