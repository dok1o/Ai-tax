# Status

| ID | Title | Agent | Branch | State |
| --- | --- | --- | --- | --- |
| T-000 | Repository bootstrap | codex | codex/T-000-bootstrap | done |
| T-001 | Token counts and fertility | codex | codex/T-001-tokens | doing |
| T-002 | Trilingual prompts and validator | codex | codex/T-002-prompts | done |
| T-003 | Cached model collection | codex | codex/T-003-collect | doing |
| T-004 | Grading and paired statistics | codex | fix/T-004-stats | done |
| T-005 | Token-cost prediction | arena | arena/ad3092b4-ai-tax | done |
| T-006 | Offline dashboard skeleton | arena | arena/ad3092b4-ai-tax | done |
| T-007 | Error-risk prediction | arena | arena/ad3092b4-ai-tax | done |
| T-008 | Routing evaluation | arena | arena/ad3092b4-ai-tax | done |
| T-009 | PowerShell-compatible Python commands | codex | codex/T-009-runner | done |
| T-010 | Automate project Definition of Done | codex | codex/T-010-done-check | done |
| T-011 | Offline trilingual Streamlit dashboard | arena | arena/ad3092b4-ai-tax | review |
| T-012 | Native review worksheet and human import | codex | codex/T-012-native-review | done |
| T-013 | Consolidate local branches and CSV v2 | codex | codex/T-013-consolidation | done |

## Requests for Astra

All prior proposals accepted in T-013. The v2 contract records the storage and compatibility decisions. T-004 includes codex/T-004-judge-stats followed by fix/T-004-stats. All five ARENA cards use the published aggregate branch. RELEASE.md distinguishes merged implementation from outstanding evidence.

- T-016 (real tokenizers offline + dashboard redesign) is implemented on `arena/ad3092b4-ai-tax` and ready for review. `tasks/` on dev has no T-016 card, so the STATUS.md row for T-011 (the dashboard card this work completes) is set to review and a T-016 row is left to Astra. `app/tokenizers.py` loads `cl100k_base` (tiktoken) and `Qwen/Qwen2.5-0.5B@5796c1d2…` (`AutoTokenizer.from_pretrained(..., local_files_only=True)`), with both identifiers read from `src/tokens.py` by ast so `transformers` is never imported to discover them; the app never downloads. When a cache is missing the tokenizer is disabled in the sidebar with the reason printed and nothing is tokenized — the old `whitespace-fallback` backend is gone, and word splitting survives only as the explicitly labelled diagnostic "words, not tokens". Chips now carry the model token id on hover, `tokens_vs_en` is computed by one tokenizer for all three languages, and the page is redesigned (three summary cards with the x1.37 multiplier, bar chart beside them, one slim "Example data"/"Real data" badge, chips in bordered cards with the count in the corner, cost/risk/router row, dark theme with one teal accent, 3D/Results/Method tabs kept). `python run.py check`: ruff clean, 174 passed. Measured in this offline sandbox: cold start 1.28–1.36 s (budget 10 s), slowest phrase 0.000037 s (budget 1 s) — see `app/TIMINGS.md`, which records that no real backend cache exists here, so the tokenizing numbers come from a fake offline backend on the same code path.
- Request: a repo-root `.streamlit/config.toml` (or `run.py dashboard` starting the server from `app/`). Streamlit reads `.streamlit/config.toml` from the launch directory or `$HOME`, never from `app/.streamlit/`, so the committed `app/.streamlit/config.toml` cannot theme the documented commands `python run.py dashboard` and `python -m streamlit run app/main.py`. Its header holds the exact content to install at the repo root and the `STREAMLIT_THEME_*` environment-variable form that works today. The page does not depend on the file: `app/main.py` (CSS) and `app/charts.apply_theme()` (Plotly template `arena_dark`) paint the same dark theme and accent in-process.
- Sol round on the ARENA review findings is pushed to `arena/ad3092b4-ai-tax`, not `fix/arena-review`: the Arena platform pins this session to one branch and refuses to create another. Fixed with regression tests in `tests/arena/test_review_fixes.py`: the deployed router threshold is now the mode of the nested per-fold picks instead of a fresh selection over all held-out rows; `src/features.decode_correctness` is the single decoder for router and training and honours the v2 `parse_status` rule; `resolve_input(kind, partner=...)` keeps prompts on the provenance of the table they join, which is what made `python -m src.train risk` fail, figures included, once T-013 moved `data/labels.csv` into `data/demo/`. `python run.py check`: ruff clean, 158 passed. No fixture number changed; only `models/router_metrics_fixture.json` was regenerated for its `selection_rule` text.
