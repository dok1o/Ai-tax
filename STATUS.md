# Status

| ID | Title | Agent | Branch | State |
| --- | --- | --- | --- | --- |
| T-000 | Repository bootstrap | codex | codex/T-000-bootstrap | done |
| T-001 | Token counts and fertility | codex | codex/T-001-tokens | todo |
| T-002 | Trilingual prompts and validator | codex | codex/T-002-prompts | todo |
| T-003 | Cached model collection | codex | codex/T-003-collect | todo |
| T-004 | Grading and paired statistics | codex | codex/T-004-stats | todo |
| T-005 | Token-cost prediction | arena | arena/ad3092b4-ai-tax | review |
| T-006 | Offline dashboard skeleton | arena | arena/ad3092b4-ai-tax | review |
| T-007 | Error-risk prediction | arena | arena/ad3092b4-ai-tax | review |
| T-008 | Routing evaluation | arena | arena/ad3092b4-ai-tax | review |

## Requests for Astra

- T-005..T-008 branch: the platform pinned this session to `arena/ad3092b4-ai-tax` (cut from `e47a220` of dev) and refused to create the four stacked branches, so all four tasks are four commits on that one pushed branch. Please merge, split or rename on your side.
- Card conflict, T-005: `tasks/T-005.md` excludes `fertility` from model inputs while the run brief lists it as a feature. Kept out of the models (it is `token_count / word_count`, an exact leak) and reported as a diagnostic in `models/REPORT.md`. Please confirm.
- Card conflict, T-007: the card names `src/risk.py`, `models/error_risk/` and `tests/test_risk_split.py`; the run brief limits ARENA to `src/features.py`, `src/train.py`, `src/router.py`, `app/`, `models/`, `tests/arena/`. Risk training is `src.train.train_risk`, outputs are `models/risk_*` and `models/figs/shap_risk_*`, tests are `tests/arena/test_risk.py`.
- Makefile (Astra-only): add `cost`, `risk` and `router` targets for `python -m src.train cost|risk` and `python -m src.router`, and repoint `dashboard` from `app/dashboard.py` (now a shim) to `app/main.py`.
- `data/demo/` is delegated to ARENA by the T-006 card but excluded by the run permissions, so fixtures live in `models/fixtures/`. The app already prefers `data/*.csv`, then `data/demo/*.csv`, then the fixtures. Copy them into `data/demo/` if you want the card's layout.
- Decide whether generated fixture artifacts (`models/*_fixture.{json,csv,joblib}`, `models/figs/*_fixture.png`) stay committed for the offline demo or get an ignore rule; they must be deleted and regenerated as soon as real `data/*.csv` land.
