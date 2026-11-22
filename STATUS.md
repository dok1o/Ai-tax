# Status

| ID | Title | Agent | Branch | State |
| --- | --- | --- | --- | --- |
| T-000 | Repository bootstrap | codex | codex/T-000-bootstrap | done |
| T-001 | Token counts and fertility | codex | codex/T-001-tokens | review |
| T-002 | Trilingual prompts and validator | codex | codex/T-002-prompts | todo |
| T-003 | Cached model collection | codex | codex/T-003-collect | todo |
| T-004 | Grading and paired statistics | codex | codex/T-004-stats | todo |
| T-005 | Token-cost prediction | arena | arena/T-005-cost | todo |
| T-006 | Offline dashboard skeleton | arena | arena/T-006-dashboard | todo |
| T-007 | Error-risk prediction | arena | arena/T-007-risk | todo |
| T-008 | Routing evaluation | arena | arena/T-008-router | todo |

## Requests for Astra

- T-001 tokens_vs_en is absent from tokens.csv v1; proposed v2: add the paired English-relative ratio column.
- T-002 answer_type values differ from v1; proposed v2: align the enum with mcq/number/short or document the v1 mapping.
- T-002 task column list omits required prompt_id; proposed v2/task update: clarify prompt_id remains required.
- T-004 unparsed status is absent from labels.csv v1; proposed v2: add a parse_status field.
- T-004 metrics.csv has no v1 schema; proposed v2: define its columns and types.
- T-002 validator path differs between card and run request; proposed v2: standardize on data/validate_prompts.py under current path permissions.
- T-002 requests a src/common topic list outside this run's paths; proposed v2: allow it or keep the list under data/.
- T-001 branch predates run.py from T-009; proposed integration: include the runner and forward --input.
