# Task verification report

## T-010 — Definition of Done checker

- Targeted command: `\.venv\Scripts\python.exe run.py test tests/codex/test_done_check.py` — 4 passed.
- Full command: `\.venv\Scripts\python.exe run.py check` — Ruff passed, 20 tests passed, prompt validator passed (60 questions / 180 rows; native verification 0.000).
- Gate command: `\.venv\Scripts\python.exe run.py done` — expected exit 1; all ten release gates currently fail on missing evidence or human decisions. DATA reports 0/60 verified kk rows and pending D1/D2; MEASUREMENT lacks `data/tokens.csv` and has pending M1; STATISTICS lacks `data/metrics.csv`; MODELS and ROUTER lack saved artifacts; APP, REPRODUCIBILITY, INTEGRITY and PRESENTATION lack release evidence; MERGE has no release tag.
- The checker is read-only and emits one PASS/FAIL row per gate. No live calls or data edits were made.
- Next: Astra reviews this branch; then resolve release evidence and human sign-offs before expecting `run.py done` to pass.
- Push: pending; if blocked, run `git push -u origin codex/T-010-done-check` after approval.
