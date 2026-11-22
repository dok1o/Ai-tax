# Task verification report

## T-012 — Native review worksheet

- Targeted command: `\.venv\Scripts\python.exe run.py test tests/codex/test_native_review.py` — 6 passed; Ruff passed for the task files.
- Full command: `\.venv\Scripts\python.exe run.py check` — Ruff passed, 22 tests passed, prompt validator passed (60 questions / 180 rows; native verification 0.000).
- Export command: `\.venv\Scripts\python.exe -m src.native_review export` wrote 60 rows to the extra `data/native_review.csv` worksheet.
- The human-only import command was not run. `data/prompts.csv` and its existing `verified_by_native` values were not changed. Import logic keeps SCHEMA v1 and serializes new marks as `true`; the numeric 1/0 mismatch and proposed v2 update are recorded under Requests in STATUS.md.
- Next: a Kazakh native speaker completes the worksheet; Astra reviews this branch. Push was not retried after the prior auto-review rejection; after approval, run `git push -u origin codex/T-012-native-review`.
