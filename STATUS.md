# Status

| ID | Title | Agent | Branch | State |
| --- | --- | --- | --- | --- |
| T-000 | Repository bootstrap | codex | codex/T-000-bootstrap | done |
| T-001 | Token counts and fertility | codex | codex/T-001-tokens | doing |
| T-002 | Trilingual prompts and validator | codex | codex/T-002-prompts | done |
| T-003 | Cached model collection | codex | codex/T-003-collect | doing |
| T-004 | Grading and paired statistics | codex | codex/T-004-judge-stats | doing |
| T-005 | Token-cost prediction | arena | arena/T-005-cost-predictor | doing |
| T-006 | Offline dashboard skeleton | arena | arena/T-006-app-skeleton | doing |
| T-007 | Error-risk prediction | arena | arena/T-007-error-predictor | doing |
| T-008 | Routing evaluation | arena | arena/T-008-router | doing |
| T-009 | PowerShell-compatible Python commands | codex | codex/T-009-runner | done |
| T-010 | Automate project Definition of Done | codex | codex/T-010-done-check | todo |
| T-011 | Offline trilingual Streamlit dashboard | arena | arena/T-011-ui | todo |
| T-012 | Native review worksheet and human import | codex | codex/T-012-native-review | todo |

## Requests for Astra

- T-012 needs a versioned prompts.csv contract that accepts numeric verification marks alongside existing boolean text. Coordinate the schema and validator before enabling the human import.

- T-001 tokens_vs_en is absent from tokens.csv v1; proposed v2: add a paired English-relative ratio column.
- T-002 task answer_type values differ from v1; proposed v2: align the enum with mcq/number/short or document the v1 mapping.
- T-002 task column list omits required prompt_id; proposed v2/task update: clarify prompt_id remains required.
- T-002 asks equal gold answers across languages while v1 localizes short_text; proposed v2: add canonical_answer or permit localized golds.
- T-002 asks verified_by_native=0/1 and no empty cells while v1 uses lowercase booleans and permits nullable; proposed v2: clarify serialization.
- T-004 unparsed status is absent from labels.csv v1; proposed v2: add a parse_status field.
- T-004 metrics.csv has no v1 schema; proposed v2: define its columns and types.
- T-002 validator path differs between card and run request; proposed run update: standardize on data/validate_prompts.py under current path permissions.
- T-002 requests a src/common topic list outside this run's paths; proposed run update: permit it or keep the list under data/.
- T-001 run.py tokens does not pass --input; proposed v2: add argument forwarding to the shared runner.
- T-001 branch predates run.py from T-009; proposed integration: include the runner before using run.py commands on that branch.
- T-003 requests a representation for missing provider usage. Preserve returned answers in the raw cache and keep incomplete rows out of v1 responses.csv pending a contract decision.
- T-004 requests a metrics.csv contract. The four existing v1 CSV schemas remain in force during this review.
- ARENA submitted its four tasks on one branch. RELEASE.md identifies the reviewed tips, and each task card records the changes required for resubmission.
