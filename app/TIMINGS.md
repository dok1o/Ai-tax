# Recorded timings

Acceptance budgets, unchanged from `tasks/T-011.md` and restated for T-016: cold
offline start to interactive results under 10 s, and one phrase tokenized in under
1 s with the resources cached through `st.cache_resource`. Both are measured below
and both are reported by the running page (sidebar caption and the Method tab, from
`st.session_state`).

## Environment

- Sandbox container, offline: no network egress, no tokenizer download.
- Python 3.11.2, streamlit 1.67.0, plotly 7.1.0, pandas 3.1.1, scikit-learn 1.9.1,
  lightgbm 4.7.0.
- Inputs: `data/prompts.csv` (real, 180 rows) plus the committed fixture tables and
  fixture-trained artifacts in `models/`, so the page shows the "Example data" badge.
- Tokenizer caches: **absent** here. There is no tiktoken BPE file for `cl100k_base`
  in `TIKTOKEN_CACHE_DIR`/`DATA_GYM_CACHE_DIR` and no
  `models--Qwen--Qwen2.5-0.5B` snapshot in the Hugging Face cache, so
  `app.tokenizers.load_backends()` returns zero backends and two reasons. Every
  "backend available" number below is therefore measured with a **fake offline
  backend** (one id per whitespace word) injected into `load_backends`, which walks
  the same `Backend.pieces` → `text.token_chips` code path as a real tokenizer. No
  number below is a claim about tiktoken or Qwen speed.
- Measured on 2026-11-22. The card names a demo laptop; this container is not that
  laptop, so the numbers are the sandbox measurement, not a laptop claim.

## Cold offline start to interactive results

`st.cache_resource.clear()` before every run, fresh interpreter per trial, page run
to its first fully rendered state in Streamlit's `AppTest` harness (7 charts and all
four tabs rendered):

| Trial | Page cold start, no backend available | Page cold start, fake backend | Wall from process start → first render |
| --- | --- | --- | --- |
| 1 | 1.36 s | 1.28 s | 2.80 / 2.83 s |
| 2 | 1.31 s | 1.33 s | 2.71 / 2.90 s |
| 3 | 1.34 s | 1.30 s | 2.77 / 2.83 s |

Budget 10 s: met with a factor of about seven. The wall column includes importing
the Streamlit test harness itself, which a real `streamlit run` pays once at server
start. `app.tokenizers.load_backends()` costs 0.0014 s when the caches are missing
(it only stats two directories) and is cached by `cached_backends` in any case.

Reproduce:

```bash
LT_FORCE_EXAMPLE=1 .venv/bin/python - <<'PY'
import time
from streamlit.testing.v1 import AppTest
started = time.perf_counter()
test = AppTest.from_file("app/main.py", default_timeout=180)
test.run()
print("wall %.2f s" % (time.perf_counter() - started))
print("recorded cold start %.2f s" % test.session_state["cold_start_seconds"])
print("slowest phrase %.6f s" % test.session_state["tokenize_seconds"])
PY
```

`tests/arena/test_app.py::test_recorded_timings_stay_inside_the_card_budgets`
asserts both budgets on every run of `python run.py check`.

## One phrase tokenized

Slowest single phrase per page run, from `st.session_state["tokenize_seconds"]`
(the maximum of the three language areas, each timed around `app.text.token_chips`):

| Trial | No backend available (nothing tokenized) | Fake backend (11–14 chips) |
| --- | --- | --- |
| 1 | 0.000004 s | 0.000032 s |
| 2 | 0.000005 s | 0.000037 s |
| 3 | 0.000005 s | 0.000034 s |

Direct call, no Streamlit: `text.token_chips(EXAMPLE_PHRASES["kk"], "kk", backend)`
takes 0.000011 s for 11 chips. Budget 1 s: met by four orders of magnitude. A real
tokenizer adds the model's own encode plus one decode per id; that cost is not
measurable in this container and has to be recorded again once a cache exists.

Caching: the tokenizer registry (`cached_backends`), the resolved inputs
(`cached_state`), the saved models (`cached_cost_models`, `cached_risk_models`,
`cached_router_policy`) and the derived evaluation frames (`cached_evaluation`,
`cached_surface_frame`, `cached_curve_frames`, `cached_accuracy`,
`cached_playground_frame`) are all `st.cache_resource`, so a rerun re-renders
without reloading anything.

## What changed for T-016

The T-011 build tokenized with a labelled `whitespace-fallback` backend whenever
`src.tokens` was not importable, so a Playground count could silently be a word
count. That backend no longer exists. The page now loads only real offline backends
(`cl100k_base` through tiktoken, `Qwen/Qwen2.5-0.5B@<pinned revision>` through
`AutoTokenizer.from_pretrained(..., local_files_only=True)`, both identifiers read
from `src/tokens.py` without importing `transformers`), and:

- a missing cache disables that tokenizer in the sidebar and prints the reason, so
  nothing is tokenized and no count is shown;
- word splitting survives only as the explicitly labelled diagnostic
  "Word split (words, not tokens)", which announces itself in a warning, drops every
  token id from the chips and renames the chart unit to words.

## Theme

The dark palette lives in `app/charts.PAGE` and `app/text.LANG_COLORS`, is applied to
Plotly by `charts.apply_theme()` (template `arena_dark`, registered as the default)
and to the page by the CSS block in `app/main.py`. `app/.streamlit/config.toml`
carries the matching Streamlit theme; see the header of that file for why Streamlit
only reads it when the server starts from `app/`, and for the environment-variable
form that works from the repository root.
