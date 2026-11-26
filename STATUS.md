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
| T-011 | Offline trilingual Streamlit dashboard | arena | arena/ad3092b4-ai-tax | doing |
| T-012 | Native review worksheet and human import | codex | codex/T-012-native-review | done |
| T-013 | Consolidate local branches and CSV v2 | codex | codex/T-013-consolidation | done |

## Requests for Astra

All prior proposals accepted in T-013. The v2 contract records the storage and compatibility decisions. T-004 includes codex/T-004-judge-stats followed by fix/T-004-stats. All five ARENA cards use the published aggregate branch. RELEASE.md distinguishes merged implementation from outstanding evidence.

- Sol round on the ARENA review findings is pushed to `arena/ad3092b4-ai-tax`, not `fix/arena-review`: the Arena platform pins this session to one branch and refuses to create another. Fixed with regression tests in `tests/arena/test_review_fixes.py`: the deployed router threshold is now the mode of the nested per-fold picks instead of a fresh selection over all held-out rows; `src/features.decode_correctness` is the single decoder for router and training and honours the v2 `parse_status` rule; `resolve_input(kind, partner=...)` keeps prompts on the provenance of the table they join, which is what made `python -m src.train risk` fail, figures included, once T-013 moved `data/labels.csv` into `data/demo/`. `python run.py check`: ruff clean, 158 passed. No fixture number changed; only `models/router_metrics_fixture.json` was regenerated for its `selection_rule` text.
