# T-011 recorded timings

Acceptance in `tasks/T-011.md`: cold offline start to interactive results under
10 s, and one phrase tokenized in under 1 s with the resources cached through
`st.cache_resource`. Both are measured below and both are also reported by the
running page (sidebar caption and the Method tab, from `st.session_state`).

## Environment

- Sandbox container, offline: no network egress, no tokenizer download.
- Python 3.11.2, streamlit 1.67.0, plotly 7.1.0, pandas 3.1.1, scikit-learn 1.9.1,
  lightgbm 4.7.0.
- Inputs: `data/prompts.csv` (real, 180 rows) plus the committed fixture tables and
  fixture-trained artifacts in `models/`, so the page shows the example banner.
- Measured on 2026-11-22. The card names a demo laptop; this container is not that
  laptop, so the numbers are the sandbox measurement, not a laptop claim.

## Cold offline start to interactive results

`st.cache_resource.clear()` before every run, fresh interpreter per trial, page run
to its first fully rendered state in Streamlit's `AppTest` harness:

| Trial | Page cold start (import of `app/main.py` → first interactive render) | Wall from process start → first interactive render |
| --- | --- | --- |
| 1 | 3.51 s | 6.36 s |
| 2 | 2.44 s | 6.12 s |
| 3 | 2.64 s | 5.87 s |

Budget 10 s: met. The wall column includes importing the Streamlit test harness
itself (about 0.7 s), which a real `streamlit run` pays once at server start.

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
print("slowest phrase %.5f s" % test.session_state["tokenize_seconds"])
PY
```

`tests/arena/test_app.py::test_recorded_timings_stay_inside_the_card_budgets`
asserts both budgets on every run of `python run.py check`.

## One phrase tokenized

Slowest single phrase per page run, from `st.session_state["tokenize_seconds"]`
(the maximum of the three language areas, each timed around `app.text.token_chips`):

| Trial | Slowest phrase | Chips |
| --- | --- | --- |
| 1 | 0.00020 s | 12–15 |
| 2 | 0.00012 s | 12–15 |
| 3 | 0.00011 s | 12–15 |

Direct call, no Streamlit: `app.text.token_chips(EXAMPLE_PHRASES["kk"], "kk")`
takes 0.000087 s for 12 chips. Budget 1 s: met by four orders of magnitude.

Caching: the tokenizer backend registry (`cached_backends`), the resolved inputs
(`cached_state`), the saved models (`cached_cost_models`, `cached_risk_models`,
`cached_router_policy`) and the derived evaluation frames (`cached_evaluation`,
`cached_surface_frame`, `cached_curve_frames`, `cached_accuracy`,
`cached_playground_frame`) are all `st.cache_resource`, so a rerun re-renders
without reloading anything.

Caveat: no T-001 tokenizer backend is importable offline yet, so the Playground
tokenizes with the labelled `whitespace-fallback` backend. Those chips are not
model tokens; when `src.tokens` publishes `TOKENIZER_IDS` and `encode_pieces`, the
same code path picks them up and the timing has to be recorded again.
