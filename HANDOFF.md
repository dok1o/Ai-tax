# Astra review handoff

- Merged T-002 into dev. Use `python run.py check` to verify lint, the test suite and the prompt dataset.
- T-001 and T-003 through T-008 are back in doing. Each task card records the fixes for its reviewed commit.
- RELEASE.md records the exact snapshots, test results, offline smoke result and remaining human work.
- Next: workers can resume their own cards. Keep native verification and paid collection separate from fixture tests.
- Release gate: rerun the integrated offline dashboard after T-006 is accepted.
