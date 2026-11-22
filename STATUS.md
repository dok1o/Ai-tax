# Status

| ID | Title | Agent | Branch | State |
| --- | --- | --- | --- | --- |
| T-000 | Repository bootstrap | codex | codex/T-000-bootstrap | done |
| T-001 | Token counts and fertility | codex | codex/T-001-tokens | todo |
| T-002 | Trilingual prompts and validator | codex | codex/T-002-prompts | todo |
| T-003 | Cached model collection | codex | codex/T-003-collect | todo |
| T-004 | Grading and paired statistics | codex | codex/T-004-stats | todo |
| T-005 | Token-cost prediction | arena | arena/ad3092b4-ai-tax | review |
| T-006 | Offline dashboard skeleton | arena | arena/T-006-dashboard | todo |
| T-007 | Error-risk prediction | arena | arena/T-007-risk | todo |
| T-008 | Routing evaluation | arena | arena/T-008-router | todo |

## Requests for Astra

- T-005 branch name: the platform pinned this session to `arena/ad3092b4-ai-tax` (cut from `e47a220` of dev), so the card name `arena/T-005-token-predictor` and the table name `arena/T-005-cost` could not be used. Please merge or rename on your side; content matches the card.
- T-005 Makefile: add a `cost` target running `$(PYTHON) -m src.train --seed 42 --n-splits 5` (Makefile is Astra-only). It exits 2 until T-001/T-002 write `data/tokens.csv` and `data/prompts.csv`.
- T-005 .gitignore: generated artifacts in `models/token_cost/` (`*.joblib`, `metrics.csv`, `oof_predictions.csv`, `folds.csv`, `run.json`) should be ignored while `models/token_cost/README.md` stays tracked.
