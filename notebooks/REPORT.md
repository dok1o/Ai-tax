# Autonomous Batch Report

## Task status

| Task | Branch / commit | Verification | Pushed | State |
| --- | --- | --- | --- | --- |
| T-001 | `codex/T-001-tokens` / `5b146f2` | 10 tests passed; Ruff passed. `run.py` is absent on this branch. | No; automatic approval review rejected push. | review |
| T-002 | `codex/T-002-prompts` / `f66a762` (includes `7caa2a8`) | `run.py check`: 16 passed; validator: 60 questions, 180 rows, 6 categories. | No; automatic approval review rejected push. | review |
| T-003 | `codex/T-003-collect` / `e2e5956` | `run.py check`: 9 passed; fake collector wrote 360 fixture responses. | No; automatic approval review rejected push. | review |
| T-004 | `codex/T-004-judge-stats` / `5baf005` | `run.py check`: 16 passed; 360 labels, 0 unparsed, 12 metric rows. | No; automatic approval review rejected push. | review |

## Key outputs

- `data/tokens.csv` retains the SCHEMA v1 columns. `data/tokens_ratio.csv` stores `tokens_vs_en` separately; its 24 ratios were derived from the saved sample token counts. A fresh tokenization attempt could not fetch the missing `cl100k_base` encoding file because network name resolution was unavailable.
- `data/prompts_draft.csv` contains 60 example questions with `verified_by_native=0`; `data/prompts.csv` contains the 180 v1 rows with `verified_by_native=false`. All 60 questions have kk, ru and en rows across six categories. No native verification is claimed.
- `data/responses.csv` contains 360 fake-client rows. Model IDs begin with `fixture/`; append-only cache and attempt logs are in ignored local `data/raw/` files and are not committed.
- `data/labels.csv`, `data/label_status.csv` and `data/metrics.csv` are fixture outputs. The v1 labels header stays unchanged; unparsed status is explicit in the extra status file. The 12 metrics rows include seeded question-cluster bootstrap intervals, exact paired McNemar p-values and Holm adjustment.

## Real, fixture and missing

- Real: saved tokenizer rows for the sample phrases are in `data/tokens.csv`; the ratio file is calculated from those saved counts. They were not freshly recomputed in this batch.
- Fixture: every response, label and metric came from `FakeClient`; no live model/API call was made. The prompt set is unverified example content.
- Missing: native-speaker review, real model observations, and a fresh offline tokenizer recomputation. No `.env` file was read.

## PowerShell verification commands

Run after assembling the four branches on `dev`:

```powershell
& '.\.venv\Scripts\python.exe' run.py check
& '.\.venv\Scripts\python.exe' -m data.validate_prompts
& '.\.venv\Scripts\python.exe' -m src.tokens --input data/demo/sample_phrases.csv
& '.\.venv\Scripts\python.exe' -m src.collect --input data/prompts.csv
& '.\.venv\Scripts\python.exe' -m src.judge --prompts data/prompts.csv --responses data/responses.csv
& '.\.venv\Scripts\python.exe' -m src.stats --labels data/labels.csv
```

## Push commands blocked by automatic approval review

The review rejected pushes because it could not verify the external GitHub destination `origin`. No retry or alternate push method was used.

```powershell
git push origin codex/T-001-tokens
git push origin codex/T-002-prompts
git push origin codex/T-003-collect
git push origin codex/T-004-judge-stats
```

## Astra review order

Review and merge T-001, T-002, T-003, then T-004. Each branch was cut independently from `dev`; Astra may need to reconcile the task status rows and repeated Requests entries while merging. Review the Requests list for the proposed v2 contract changes before any schema revision.


## T-010 — Definition of Done checker

- Targeted command: `\.venv\Scripts\python.exe run.py test tests/codex/test_done_check.py` — 7 passed; Ruff passed for task files.
- Full command: `\.venv\Scripts\python.exe run.py check` — Ruff passed, 23 tests passed, prompt validator passed (60 questions / 180 rows; native verification 0.000).
- Gate command: `\.venv\Scripts\python.exe run.py done` — expected exit 1; all ten release gates currently fail on missing evidence or human decisions. DATA reports 0/60 verified kk rows and pending D1/D2; MEASUREMENT lacks `data/tokens.csv` and has pending M1; STATISTICS lacks `data/metrics.csv`; MODELS and ROUTER lack saved artifacts; APP lacks offline timing/sign-off; REPRODUCIBILITY lacks `notebooks/rebuild_comparison.csv`; INTEGRITY and PRESENTATION lack human evidence; MERGE has no release tag.
- The checker is read-only and emits one PASS/FAIL row per gate. No live calls or data edits were made.
- Next: Astra reviews this branch; then resolve release evidence and human sign-offs before expecting `run.py done` to pass.
- Push: not retried because automatic approval review previously rejected this external origin as unverified. After approval, run `git push -u origin codex/T-010-done-check`.
