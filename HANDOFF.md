# Astra review handoff

- Merged T-002 into dev. Use `python run.py check` to verify lint, the test suite and the prompt dataset.
- T-001 and T-003 through T-008 are back in doing. Each task card records the fixes for its reviewed commit.
- RELEASE.md records the exact snapshots, test results, offline smoke result and remaining human work.
- Next: workers can resume their own cards. Keep native verification and paid collection separate from fixture tests.
- Release gate: rerun the integrated offline dashboard after T-006 is accepted.

done: T-010 checker, tests, report, and review status are committed on codex/T-010-done-check; run.py check passed (23 tests).
next: Astra reviews the checker; run.py done remains FAIL until release evidence and human decisions exist.
blockers: Missing release evidence and approval-review block for the external push.
files touched: run.py; src/done_check.py; tests/codex/test_done_check.py; STATUS.md; notebooks/REPORT.md.
resume command: git switch codex/T-010-done-check
