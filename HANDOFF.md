# T-004 Handoff

- Done: regraded 1,080 saved responses; no API calls; one clear Cyrillic MCQ marker is now parsed, while multiple/incomplete option lists remain unparsed.
- Done: metrics include `accuracy_strict` and `accuracy_parsed`, both with q_id bootstrap CIs; report updated from generated CSVs.
- Verified: 33 targeted judge/statistics tests passed; Ruff passed.
- Current data: 122/1,080 responses unparsed; llama3.2:3b kk 64/120 (53.3%).
- Note: full `run.py check` was not rerun; an earlier attempt stalled after seven test dots.
- Files: `data/labels.csv`, `data/label_status.csv`, `data/metrics.csv`, `src/judge.py`, `src/stats.py`, `tests/codex/test_judge.py`, `tests/codex/test_stats.py`, `notebooks/REPORT.md`.
- Next: Astra reviews the local T-004 commit. No push attempted; auto-review previously rejected `origin` verification.
- Resume: `git diff dev...codex/T-004-real-run`
