"""Offline dashboard page (T-006) with cost, risk and router panels.

Start it with ``make dashboard`` or ``python -m streamlit run app/main.py``.
It reads local CSV files and saved models only: no API call, no tokenizer
download, no network request. Synthetic inputs turn on a visible banner.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app import logic
from src.features import LANGS


@st.cache_resource(show_spinner=False)
def cached_state() -> logic.AppState:
    """Resolve inputs once per server session."""
    return logic.load_state()


@st.cache_resource(show_spinner=False)
def cached_cost_models() -> dict:
    """Saved token-cost models."""
    return logic.load_cost_models()


@st.cache_resource(show_spinner=False)
def cached_risk_models() -> dict:
    """Saved error-risk models."""
    return logic.load_risk_models()


def render_sidebar(state: logic.AppState) -> None:
    """Show where every input came from."""
    st.sidebar.header("Inputs")
    if state.banner:
        st.sidebar.warning(state.banner)
    for kind, source in sorted(state.sources.items()):
        st.sidebar.write(f"- `{kind}`: **{source}**")
    prompts = state.table("prompts")
    tokens = state.table("tokens")
    labels = state.table("labels")
    st.sidebar.caption(
        f"prompts: {0 if prompts is None else len(prompts)} rows, "
        f"tokens: {0 if tokens is None else len(tokens)} rows, "
        f"labels: {0 if labels is None else len(labels)} rows. "
        "Everything is read from disk; the app never calls an API."
    )


def render_predictor(state: logic.AppState) -> tuple[str, str, str]:
    """Phrase input with the predicted token cost."""
    st.subheader("Predict the token cost of a phrase")
    lang = st.selectbox("Language", list(LANGS), index=0)
    category = st.selectbox("Category", logic.categories(state), index=0)
    phrase = st.text_area("Phrase", value=logic.default_phrase(state, lang), height=100)
    features = logic.phrase_features(phrase, lang, category)
    st.caption(
        " | ".join(f"{name}: {value}" for name, value in features.items())
        + "  (features are computed before any tokenizer runs)"
    )
    costs = logic.predict_cost(phrase, lang, category, cached_cost_models())
    if costs is None:
        st.info(logic.empty_state("cost model"))
    else:
        st.dataframe(costs, hide_index=True)
        if bool(costs["fixture"].any()):
            st.caption("Predictions come from models trained on fixture data.")
    return phrase, lang, category


def render_risk(phrase: str, lang: str, category: str, state: logic.AppState) -> None:
    """Error-risk panel (T-007): placeholder until risk artifacts exist."""
    st.subheader("Risk of a wrong answer")
    models = cached_risk_models()
    if not models:
        st.number_input(
            "Predicted P(incorrect)",
            value=0.0,
            disabled=True,
            help="Pending T-007: the error-risk model has not been trained yet.",
        )
        st.caption("Placeholder — enable after T-007 writes models/risk_*.joblib.")
        return
    answer_model = st.selectbox("Answering model", logic.answer_models(state), index=0)
    risks = logic.predict_risk(phrase, lang, category, answer_model, models)
    st.dataframe(risks, hide_index=True)
    st.caption(
        "P(incorrect) from models trained with q_id-grouped folds. "
        "Fixture-trained models are preliminary, not a result."
    )


def render_router() -> None:
    """Router panel: a disabled placeholder until T-008 artifacts exist."""
    st.subheader("Router decision")
    st.text_input(
        "Chosen strategy",
        value="pending T-008",
        disabled=True,
        help="Pending T-008: direct / via_english / stronger_model.",
    )


def render_charts(state: logic.AppState) -> None:
    """Tabs with the measured summaries read from the CSV tables."""
    tokens_tab, accuracy_tab, models_tab, figures_tab = st.tabs(
        ["Tokens per language", "Accuracy per language", "Model comparison", "Figures"]
    )
    with tokens_tab:
        table = logic.tokens_per_language(state.table("tokens"))
        if table is None:
            st.info(logic.empty_state("tokens"))
        else:
            st.dataframe(table, hide_index=True)
            st.bar_chart(
                table.pivot(index="lang", columns="tokenizer", values="tokens_mean")
            )
    with accuracy_tab:
        table = logic.accuracy_per_language(state.table("labels"))
        if table is None:
            st.info(logic.empty_state("labels"))
        else:
            st.dataframe(table, hide_index=True)
            st.bar_chart(table.pivot(index="lang", columns="model", values="accuracy"))
    with models_tab:
        costs = logic.cost_rows(state.manifests.get("cost_metrics"))
        fertility = logic.fertility_rows(state.manifests.get("cost_metrics"))
        if costs is None:
            st.info(logic.empty_state("cost metrics"))
        else:
            st.dataframe(costs, hide_index=True)
            st.bar_chart(
                costs.pivot(index="tokenizer", columns="model", values="mae_mean")
            )
        risks = logic.risk_rows(state.manifests.get("risk_metrics"))
        if risks is not None:
            st.caption("Error-risk models (T-007), held out by q_id.")
            st.dataframe(risks, hide_index=True)
        if fertility is not None:
            st.caption(
                "Measured fertility (tokens per word) — diagnostic, never a feature."
            )
            st.dataframe(fertility, hide_index=True)
    with figures_tab:
        figures = logic.figure_paths()
        if not figures:
            st.info(logic.empty_state("figures"))
        for path in figures:
            st.image(str(path), caption=path.name)


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
    render_sidebar(state)
    phrase, lang, category = render_predictor(state)
    render_risk(phrase, lang, category, state)
    render_router()
    render_charts(state)


render()
