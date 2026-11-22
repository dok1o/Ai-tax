# T-005 handoff

- Done: pre-tokenization feature builder (`src/features.py`), GroupKFold-by-`q_id` cost training and model comparison (`src/train.py`), split/leakage tests (`tests/test_cost_split.py`, 7 cases), output contract documented in `models/token_cost/README.md`.
- Verified: `make check PYTHON=.venv/bin/python` passes (ruff format, ruff check, 11 tests). End-to-end run on a temporary synthetic fixture wrote metrics.csv, oof_predictions.csv, folds.csv, run.json and four refit `.joblib` files; `python -m src.train` exits 2 with a clear message while T-001/T-002 inputs are absent.
- Numbers: none committed. No real prompts or token counts exist yet, so `models/token_cost/` holds only its README.
- Next: rerun `python -m src.train --seed 42 --n-splits 5` as soon as `data/prompts.csv` and `data/tokens.csv` land, then read `metrics.csv` (linear vs lightgbm, by lang and tokenizer) for the poster and T-008 routing costs.
- Blockers: T-001 and T-002 outputs; Makefile has no `cost` target (Astra-owned).
- Branch note: the session was pinned by the platform to `arena/ad3092b4-ai-tax`, not the card name `arena/T-005-token-predictor` or the STATUS.md name `arena/T-005-cost`. Content is identical to the card; Astra decides the merge/rename.
- Files touched: `src/features.py`, `src/train.py`, `tests/test_cost_split.py`, `models/token_cost/README.md`, `STATUS.md` (own row plus requests), `HANDOFF.md`.
- Dependencies: none added; `joblib` ships with scikit-learn. Environment built with `python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt`.
- Resume: `git switch arena/ad3092b4-ai-tax && make check PYTHON=.venv/bin/python && .venv/bin/python -m pytest tests/test_cost_split.py -q`.
- Reuse for T-007/T-008: `src.features.build_features` / `design_matrix` and `src.train.assign_folds` are target-agnostic; `refit_artifacts` payloads carry the column order a router needs.
