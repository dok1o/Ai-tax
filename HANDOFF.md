# T-004 Handoff

- Done: real-only judge and metrics generated from `data/responses.csv` (1,080 rows; no fixture models).
- Done: numeric final-result and Cyrillic multiple-choice parsing regressions covered; paired tests now use per-question repetition aggregation and per-model Holm correction.
- Done: `notebooks/REPORT.md` updated with cell counts, unparsed rates, accuracy intervals, paired tests, sanity sample, and data issues.
- Verified: targeted judge/statistics tests passed (29); Ruff passed.
- Pending: full `python run.py check` passed Ruff, then pytest stalled after seven test dots and was interrupted after about 10 minutes; no assertion failure was reported.
- Blocked: push was rejected by auto-review because `origin` was not verified as the authorized destination. Do not retry until the human confirms the destination.
- Files: `data/responses.csv`, `data/labels.csv`, `data/label_status.csv`, `data/metrics.csv`, `src/judge.py`, `src/stats.py`, `tests/codex/test_judge.py`, `tests/codex/test_stats.py`, `notebooks/REPORT.md`, `STATUS.md`.
- Resume: investigate the slow full-suite test; after the human confirms `origin`, retry `git push -u origin codex/T-004-real-run`.
