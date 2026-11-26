# Status

| ID | Title | Agent | Branch | State |
| --- | --- | --- | --- | --- |
| T-000 | Repository bootstrap | codex | codex/T-000-bootstrap | done |
| T-001 | Token counts and fertility | codex | codex/T-001-tokens | doing |
| T-002 | Trilingual prompts and validator | codex | codex/T-002-prompts | done |
| T-003 | Cached model collection | codex | codex/T-003-collect | doing |
| T-004 | Real Ollama grades and paired statistics | codex | codex/T-004-real-run | done |
| T-005 | Token-cost prediction | arena | arena/ad3092b4-ai-tax | done |
| T-006 | Offline dashboard skeleton | arena | arena/ad3092b4-ai-tax | done |
| T-007 | Error-risk prediction | arena | arena/ad3092b4-ai-tax | done |
| T-008 | Routing evaluation | arena | arena/ad3092b4-ai-tax | done |
| T-009 | PowerShell-compatible Python commands | codex | codex/T-009-runner | done |
| T-010 | Automate project Definition of Done | codex | codex/T-010-done-check | done |
| T-011 | Offline trilingual Streamlit dashboard | arena | arena/ad3092b4-ai-tax | doing |
| T-012 | Native review worksheet and human import | codex | codex/T-012-native-review | done |
| T-013 | Consolidate local branches and CSV v2 | codex | codex/T-013-consolidation | done |
| T-014 | Import final v2 prompt dataset | codex | codex/T-014-dataset | done |
| T-015 | Ollama and Gemini model clients | codex | codex/T-015-clients | done |
| T-016 | Offline tokenizers and dashboard redesign | arena | arena/ad3092b4-ai-tax | blocked |
| T-017 | Prompt text corrections and token regeneration | codex | codex/T-017-text-fixes | done |

## Requests for Astra

- Claude Code acts as Astra for the rest of this sprint by human decision. Schema v2.1, T-004 and T-017 are merged into dev.
- Codex (T-010 owner): `src/done_check.py` accepts only `accuracy` and `mcnemar_exact`. Make it read `accuracy_strict` and the other v2.1 metric names before `python run.py done` is used.
- ARENA (Sol): T-016 stays blocked at `df87b23`. Two T-016 tests assume a cold tokenizer cache; make them hermetic (fail on a warm machine cache) and set the T-016 row to review. The network guard itself works.
