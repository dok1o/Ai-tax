# T-001 handoff

- Done: Added token counting for `cl100k_base` and revision-pinned `google/byt5-small`; output uses the contract columns and fertility formula.
- Done: Added targeted tests for kk, ru and en, punctuation, repeated whitespace, tokenizer identifiers and deterministic rows.
- Next: Run `py -3.11 -m pytest tests/test_tokens.py -q`; run `make tokens` when T-002 provides `data/prompts.csv`.
- Blockers: Python 3.11 is not installed (`py -3.11` reports no matching runtime); `data/prompts.csv` is pending T-002.
- Files touched: `src/tokens.py`, `tests/test_tokens.py`, `STATUS.md`, `HANDOFF.md`.
- Resume: `git switch codex/T-001-tokens`.
