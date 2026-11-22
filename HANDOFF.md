# T-009 handoff

- Done: the project uses `python run.py` for checks, tests, token counts and the offline dashboard.
- Next: start T-002 for prompts. T-001 can develop against test inputs while prompts are prepared. T-006 can start with offline examples.
- Blockers: none for these cards. Collection needs configured models, local credentials and human approval before paid calls.
- Files touched: root command runner, AGENTS.md, STATUS.md, HANDOFF.md and task cards T-001 through T-008.
- Resume T-002 from dev: `git switch dev && git pull --ff-only && git switch -c codex/T-002-prompts`.
- Verify with Python 3.11: `python -m pip install -r requirements.txt`, then `python run.py check`.
