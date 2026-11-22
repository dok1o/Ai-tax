# Status

| ID | Title | Agent | Branch | State |
| --- | --- | --- | --- | --- |
| T-000 | Repository bootstrap | codex | codex/T-000-bootstrap | done |
| T-001 | Token counts and fertility | codex | codex/T-001-tokens | review |
| T-002 | Trilingual prompts and validator | codex | codex/T-002-prompts | review |
| T-003 | Cached model collection | codex | codex/T-003-collect | review |
| T-004 | Grading and paired statistics | codex | codex/T-004-judge-stats | review |
| T-005 | Token-cost prediction | arena | arena/T-005-cost | todo |
| T-006 | Offline dashboard skeleton | arena | arena/T-006-dashboard | todo |
| T-007 | Error-risk prediction | arena | arena/T-007-risk | todo |
| T-008 | Routing evaluation | arena | arena/T-008-router | todo |
| T-009 | PowerShell-compatible Python commands | codex | codex/T-009-runner | done |

## Requests for Astra

- T-001 tokens_vs_en is absent from tokens.csv v1; proposed v2: add a paired English-relative ratio column.
- T-002/T-004 request answer_type mcq/number/short while v1 permits integer/decimal/boolean/short_text; proposed v2: align the enum or document aliases.
- T-002 task column list omits required prompt_id; proposed v2/task update: clarify prompt_id remains required.
- T-002 asks equal gold answers across languages while v1 localizes short_text; proposed v2: add canonical_answer or permit localized golds.
- T-002 asks verified_by_native=0/1 and no empty cells while v1 uses lowercase booleans and permits nullable; proposed v2: clarify serialization.
- T-003 provider usage may be unavailable while responses v1 requires nonnegative integers; proposed v2: make usage nullable with an explicit availability field.
- T-004 unparsed status is absent from labels.csv v1; proposed v2: add a parse_status field.
- T-004 metrics.csv has no v1 schema; proposed v2: define its columns and types.
- T-002 validator path differs between card and run request; proposed run update: standardize on data/validate_prompts.py under current path permissions.
- T-002 requests a src/common topic list outside this run's paths; proposed run update: permit it or keep the list under data/.
- T-001 run.py tokens does not pass --input; proposed v2: add argument forwarding to the shared runner.
- T-001 branch predates run.py from T-009; proposed integration: include the runner before using run.py commands on that branch.
- T-004 STATUS branch differs from run assignment; proposed task update: use codex/T-004-judge-stats consistently.
