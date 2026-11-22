# T-000 handoff

- Done: project layout, shared paths and seed, CSV contract v1, environment template, Make targets and eight task cards.
- Next: start T-002 for prompts. T-001 can develop against test inputs while prompts are prepared. T-006 can start with offline examples.
- Blockers: none for these cards. Collection needs configured models, local credentials and human approval before paid calls.
- Files touched: AGENTS.md (canonical filename), root setup and tracking files, data/SCHEMA.md, src/common/config.py, tests/test_schema.py, tasks/ and empty pipeline placeholders.
- Resume T-002 from dev: `git switch dev && git pull --ff-only && git switch -c codex/T-002-prompts`.
- Verify with Python 3.11 and GNU Make: `python -m pip install -r requirements.txt`, then `make check`.
