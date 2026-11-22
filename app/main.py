"""Offline trilingual dashboard (T-011): Playground, Charts, Results, Method.

Start it with ``python run.py dashboard`` or
``python -m streamlit run app/main.py``. Every number comes from files on disk:
no API call, no tokenizer download, no network request. Machine translation
stays switched off unless the operator opens both live gates, and anything
fixture-derived turns on the example banner.
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

from app import charts, logic, text
from src.features import LANGS

TABS = ("Playground", "Charts", "Results", "Method")
TEXT_KEY = "text_{lang}"
COST_MODEL_HELP = "Pending T-005: no saved token-cost model was found offline."
RISK_MODEL_HELP = "Pending T-007: no saved error-risk model was found offline."
ROUTER_HELP = "Pending T-008: direct / via_english / stronger_model."
TRANSLATE_HELP = (
    "Off unless the operator starts the app with the live flag (LT_LIVE=1) and a "
    "translation key (LT_TRANSLATE_KEY). Nothing is sent without a button press."
)


@st.cache_resource(show_spinner=False)
def cached_state() -> logic.AppState:
    """Resolve inputs once per server process."""
    return logic.load_state()


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
def cached_backends() -> dict:
    """Local tokenizer backends, discovered once and reused by every rerun."""
    return text.local_tokenizers()


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


def tokenizer_menu(state: logic.AppState) -> list[str]:
    """Sidebar tokenizer choices: measured tokenizers first, local backends after."""
    observed = logic.observed_tokenizers(state.table("tokens"))
    local = text.available_backends()
    return observed + [name for name in local if name not in observed]


def chart_tokenizer(state: logic.AppState, selected: str) -> str | None:
    """Tokenizer used by the charts: the selection when it has measured rows."""
    observed = logic.observed_tokenizers(state.table("tokens"))
    if selected in observed:
        return selected
    return observed[0] if observed else None


def playground_tokenizer(selected: str) -> str:
    """Tokenizer used for the chips: the selection when it runs offline."""
    return selected if selected in cached_backends() else text.FALLBACK_TOKENIZER


def timed_chips(
    phrase: str, lang: str, tokenizer: str
) -> tuple[list[text.Chip], float]:
    """Chips for one phrase plus the seconds the tokenization took."""
    started = time.perf_counter()
    chips = text.token_chips(phrase, lang, tokenizer, cached_backends())
    return chips, time.perf_counter() - started


def record_cold_start() -> float:
    """Seconds from importing this page to the first fully rendered run."""
    if "cold_start_seconds" not in st.session_state:
        st.session_state["cold_start_seconds"] = time.perf_counter() - START_TIME
    return float(st.session_state["cold_start_seconds"])


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


def render_sidebar(state: logic.AppState) -> tuple[str, str]:
    """Provenance, tokenizer choice, 3D/2D toggle and the recorded timings."""
    st.sidebar.header("Inputs")
    if state.banner:
        st.sidebar.warning(state.banner)
    for kind, source in sorted(state.sources.items()):
        st.sidebar.write(f"- `{kind}`: **{source}**")
    st.sidebar.caption(
        "Everything is read from disk; the app never calls an API or downloads a "
        "tokenizer."
    )
    st.sidebar.header("Tokenizer")
    menu = tokenizer_menu(state)
    selected = st.sidebar.selectbox(
        "Tokenizer",
        menu or [text.FALLBACK_TOKENIZER],
        index=0,
        help="Charts use it when the tokens table has rows for it; the Playground "
        "uses it only when it runs offline, otherwise the labelled fallback does.",
    )
    view = st.sidebar.radio("3D views", ("3D", "2D"), horizontal=True)
    cold = record_cold_start()
    st.sidebar.caption(
        f"cold start to interactive results: {cold:.2f} s (budget "
        f"{logic.COLD_START_BUDGET_SECONDS:.0f} s); inputs, models and tokenizer "
        "backends are cached with st.cache_resource. Both timings are repeated in "
        "the Method tab."
    )
    return str(selected), str(view)


def playground_phrases() -> dict[str, str]:
    """Text of every language area, with the bundled examples filled in first."""
    for lang in LANGS:
        st.session_state.setdefault(
            TEXT_KEY.format(lang=lang), logic.EXAMPLE_PHRASES[lang]
        )
    return {lang: str(st.session_state[TEXT_KEY.format(lang=lang)]) for lang in LANGS}


def render_language_column(
    lang: str, phrases: dict[str, str], tokenizer: str, translated: bool
) -> None:
    """One independent text area with its chips, counts and ratios."""
    phrase = st.text_area(f"{lang} text", key=TEXT_KEY.format(lang=lang), height=110)
    if translated:
        st.caption(text.TRANSLATE_LABEL)
    chips, seconds = timed_chips(phrase, lang, tokenizer)
    st.session_state["tokenize_seconds"] = max(
        float(st.session_state.get("tokenize_seconds", 0.0)), seconds
    )
    st.markdown(text.chips_html(chips), unsafe_allow_html=True)
    counts = {
        other: (
            len(text.tokenize(phrases[other], tokenizer, cached_backends()))
            if phrases[other].strip()
            else None
        )
        for other in LANGS
    }
    counts[lang] = len(chips) or None
    ratio_value = text.tokens_vs_en(counts)[lang]
    st.metric("tokens", len(chips) if chips else text.UNAVAILABLE)
    st.metric("tokens_vs_en", text.format_ratio(ratio_value))
    st.metric(
        "fertility (tokens/word)",
        text.format_ratio(text.fertility(len(chips) or None, text.word_count(phrase))),
    )
    st.caption(
        f"script hint {text.detect_lang(phrase)} (heuristic) | Kazakh-specific "
        f"letters {text.kazakh_letter_share(phrase):.1%} | tokenizer {tokenizer}"
    )


def render_models(phrase: str, lang: str, category: str, state: logic.AppState) -> None:
    """Cost (T-005), error risk (T-007) and route (T-008) for the scored phrase."""
    st.subheader("Models on this phrase")
    cost_models = cached_cost_models()
    costs = logic.predict_cost(phrase, lang, category, cost_models)
    if costs is None:
        st.number_input(
            "Predicted tokens", value=0.0, disabled=True, help=COST_MODEL_HELP
        )
        st.caption("Placeholder — enable after T-005 writes models/cost_*.joblib.")
    else:
        chosen = None
        try:
            model, tokenizer = logic.best_cost_model(
                state.manifests.get("cost_metrics")
            )
            picked = costs[
                (costs["model"] == model) & (costs["tokenizer"] == tokenizer)
            ]
            chosen = None if picked.empty else float(picked.iloc[0]["predicted_tokens"])
        except ValueError:
            chosen = float(costs["predicted_tokens"].min())
        st.metric("Predicted tokens (best T-005 model)", round(chosen, 2))
        st.dataframe(costs, hide_index=True)
        if bool(costs["fixture"].any()):
            st.caption("Cost predictions come from models trained on fixture data.")

    risk_models = cached_risk_models()
    if not risk_models:
        st.number_input(
            "Predicted P(incorrect)", value=0.0, disabled=True, help=RISK_MODEL_HELP
        )
        st.caption("Placeholder — enable after T-007 writes models/risk_*.joblib.")
    else:
        answer_model = st.selectbox(
            "Answering model", logic.answer_models(state), index=0
        )
        risks = logic.predict_risk(phrase, lang, category, answer_model, risk_models)
        st.dataframe(risks, hide_index=True)
        st.caption(
            "P(incorrect) from models trained with q_id-grouped folds; "
            "fixture-trained models are preliminary, not a result."
        )

    policy = cached_router_policy()
    if policy is None:
        st.text_input(
            "Chosen strategy", value="pending T-008", disabled=True, help=ROUTER_HELP
        )
        return
    threshold = st.slider(
        "Risk threshold", 0.05, 0.95, float(policy["threshold"]), 0.05
    )
    options = logic.router_options(phrase, lang, category, policy, threshold)
    if options is None:
        st.info(logic.empty_state("cost or risk model"))
        return
    chosen_rows = options.loc[options["chosen"], "strategy"].tolist()
    st.metric("Chosen strategy", chosen_rows[0] if chosen_rows else "none")
    st.write(logic.route_reason(options, threshold))
    st.dataframe(options, hide_index=True)
    st.caption(
        "Cheapest strategy whose predicted risk stays under the threshold; "
        "estimates come from the saved T-005 and T-007 models plus the T-008 "
        "calibration."
    )


def render_playground(state: logic.AppState, tokenizer: str) -> None:
    """Three independent language areas, chips, ratios and the three models."""
    st.subheader("Playground")
    st.caption(
        "Start in any language and fill the others by hand. Chips, counts and "
        "ratios are derived from your text and the offline tokenizer only."
    )
    live = text.live_translation_enabled()
    phrases = playground_phrases()
    focus = st.selectbox("Score language", list(LANGS), index=0)
    category = st.selectbox("Category", logic.categories(state), index=0)
    pressed = st.button(
        "Machine translate",
        disabled=not live,
        help=TRANSLATE_HELP if not live else "Fills the other two areas on press.",
    )
    if pressed:
        translate_into(focus)
    if not live:
        st.caption(TRANSLATE_HELP)
    backend = playground_tokenizer(tokenizer)
    if backend != tokenizer:
        st.caption(
            f"{tokenizer} has no offline backend here, so the chips use the labelled "
            f"`{backend}` and are not model tokens."
        )
    columns = st.columns(len(LANGS))
    for column, lang in zip(columns, LANGS, strict=True):
        with column:
            render_language_column(lang, phrases, backend, pressed)
    render_models(phrases[focus], focus, category, state)


def render_charts(state: logic.AppState, tokenizer: str, view: str) -> None:
    """Token counts, the 3D prompt view and the router surface, each with a 2D twin."""
    example = state.is_example
    tokens = state.table("tokens")
    per_language = logic.tokens_per_language(tokens)
    measured = chart_tokenizer(state, tokenizer)
    if per_language is None or measured is None:
        st.info(logic.empty_state("tokens"))
    else:
        st.plotly_chart(
            charts.tokens_by_language(per_language, measured, example),
            key="chart_tokens_by_language",
        )
        note_chart("chart_tokens_by_language")
        st.dataframe(per_language, hide_index=True)
        ratios = logic.tokens_vs_en_table(tokens)
        if ratios is None:
            st.info(logic.derived_missing("token ratios"))
        else:
            st.plotly_chart(
                charts.token_ratios(ratios, measured, example),
                key="chart_ratios_in_charts_tab",
            )
            note_chart("chart_ratios_in_charts_tab")

    prompts = cached_playground_frame()
    if prompts is None:
        st.info(logic.derived_missing("prompt predictions"))
    elif view == "3D":
        st.plotly_chart(charts.scatter3d(prompts, example), key="chart_prompts_3d")
        note_chart("chart_prompts_3d")
    else:
        st.plotly_chart(charts.scatter3d_2d(prompts, example), key="chart_prompts_2d")
        note_chart("chart_prompts_2d")

    surface = cached_surface_frame()
    if surface is None:
        st.info(logic.derived_missing("router evaluation"))
    elif view == "3D":
        st.plotly_chart(charts.router_surface(surface, example), key="chart_surface_3d")
        note_chart("chart_surface_3d")
    else:
        st.plotly_chart(
            charts.router_surface_2d(surface, example), key="chart_surface_2d"
        )
        note_chart("chart_surface_2d")


def render_results(state: logic.AppState, tokenizer: str) -> None:
    """Accuracy with CIs, token ratios and the router curve against two baselines."""
    example = state.is_example
    accuracy = cached_accuracy()
    st.subheader("Accuracy per language")
    if accuracy is None:
        st.info(logic.empty_state("labels"))
    else:
        st.plotly_chart(
            charts.accuracy_with_ci(accuracy, example), key="chart_accuracy_ci"
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
        measured = chart_tokenizer(state, tokenizer)
        if measured is not None:
            st.plotly_chart(
                charts.token_ratios(ratios, measured, example),
                key="chart_results_token_ratios",
            )
            note_chart("chart_results_token_ratios")
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
    )
    note_chart("chart_router_curve")
    st.dataframe(
        curve[["threshold", "n", "cost_tokens_mean", "accuracy"]], hide_index=True
    )
    st.dataframe(
        baselines[["policy", "n", "cost_tokens_mean", "accuracy"]], hide_index=True
    )


def render_method(state: logic.AppState) -> None:
    """Data sources, model versions, query dates, n, verified share, limitations."""
    tables = logic.method_tables(state)
    st.subheader("Data sources")
    st.dataframe(tables["sources"], hide_index=True)
    st.subheader("Model versions and query dates")
    if tables["versions"].empty:
        st.info(logic.derived_missing("metrics manifest"))
    else:
        st.dataframe(tables["versions"], hide_index=True)
    st.subheader("Sample sizes")
    counts = pd.DataFrame(
        [
            {
                "input": kind,
                "n_rows": 0 if state.table(kind) is None else len(state.table(kind)),
                "path": state.paths.get(kind, ""),
            }
            for kind in logic.TABLES
        ]
    )
    st.dataframe(counts, hide_index=True)
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
        f"- cold offline start to interactive results: "
        f"{record_cold_start():.2f} s (budget "
        f"{logic.COLD_START_BUDGET_SECONDS:.0f} s)"
    )
    st.write(
        f"- slowest single-phrase tokenization this run: "
        f"{float(st.session_state.get('tokenize_seconds', 0.0)):.4f} s (budget "
        f"{logic.TOKENIZE_BUDGET_SECONDS:.0f} s)"
    )


def render() -> None:
    """Build the whole page."""
    st.set_page_config(page_title="Language Tax in LLMs", layout="wide")
    state = cached_state()
    st.title("Language Tax in LLMs")
    st.caption(
        "Kazakh, Russian and English: the same question costs different amounts of "
        "tokens and gets different answers. Offline demo, saved results only."
    )
    if state.banner:
        st.error(state.banner)
    tokenizer, view = render_sidebar(state)
    st.session_state["chart_run"] = []
    playground, charts_tab, results, method = st.tabs(list(TABS))
    with playground:
        render_playground(state, tokenizer)
    with charts_tab:
        render_charts(state, tokenizer, view)
    with results:
        render_results(state, tokenizer)
    with method:
        render_method(state)
    finish_charts()
    record_cold_start()


render()
