# Astra review handoff

- Merged T-002 into dev. Use `python run.py check` to verify lint, the test suite and the prompt dataset.
- T-001 and T-003 through T-008 are back in doing. Each task card records the fixes for its reviewed commit.
- RELEASE.md records the exact snapshots, test results, offline smoke result and remaining human work.
- Next: workers can resume their own cards. Keep native verification and paid collection separate from fixture tests.
- Release gate: rerun the integrated offline dashboard after T-006 is accepted.

done: T-010 checker, tests, report, and review status are committed on codex/T-010-done-check; run.py check passed (23 tests).
next: Astra reviews the checker; run.py done remains FAIL until release evidence and human decisions exist.
blockers: Missing release evidence and approval-review block for the external push.
files touched: run.py; src/done_check.py; tests/codex/test_done_check.py; STATUS.md; notebooks/REPORT.md.
resume command: git switch codex/T-010-done-check

done: T-016 on arena/ad3092b4-ai-tax — `app/tokenizers.py` loads the two pinned backends offline only (tiktoken `cl100k_base`; `Qwen/Qwen2.5-0.5B@5796c1d2…` via `AutoTokenizer.from_pretrained(..., local_files_only=True)`), identifiers read from `src/tokens.py` by ast so `transformers` is never imported for discovery, behind `st.cache_resource`. The `whitespace-fallback` backend is deleted: a missing cache disables that tokenizer with the printed reason and nothing is tokenized, and word splitting survives only as the labelled diagnostic "words, not tokens". Chips carry the real token id on hover, `tokens_vs_en` uses one tokenizer for all three languages, and the page is redesigned (three summary cards with the x1.37 multiplier, bar chart beside them, one slim Example/Real badge, chips in bordered cards with the count in the corner, cost/risk/router row with a one-line reason and disabled placeholders, dark theme with one teal accent, 3D/Results/Method tabs kept). `python run.py check`: ruff clean, 174 passed, validator 60 questions / 180 rows. Measured offline: cold start 1.28–1.36 s (budget 10 s), slowest phrase 0.000037 s (budget 1 s), see `app/TIMINGS.md`.
next: Astra reviews the diff and installs a repo-root `.streamlit/config.toml` (Requests bullet in STATUS.md). Re-record the per-phrase timing on a machine where a real backend cache exists — every tokenizing number here comes from a fake offline backend on the same code path, because this sandbox has neither cache.
blockers: none for the code. Offline evidence limit only: no tiktoken BPE cache and no Hugging Face snapshot in this container, so the real encode/decode cost is unmeasured, not unknown-by-design.
files touched: app/tokenizers.py (new); app/text.py; app/charts.py; app/main.py; app/.streamlit/config.toml (new); app/TIMINGS.md; tests/arena/test_tokenizers.py (new); tests/arena/test_app.py; tests/arena/test_playground_text.py; tests/arena/test_charts_t011.py; STATUS.md; HANDOFF.md.
resume command: cd app && python -m streamlit run main.py   # reads app/.streamlit/config.toml; from the repo root use the STREAMLIT_THEME_* form documented in that file
