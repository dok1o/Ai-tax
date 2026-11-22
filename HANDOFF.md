# T-001 handoff

- Done: Replaced ByT5 with revision-pinned Qwen2.5 subword tokenization alongside `cl100k_base`.
- Done: Added `--input` CSV support for `text` and `lang`; prompt and question IDs are generated when absent.
- Done: Added 12 trilingual example phrases with shared question IDs and `is_example=true`.
- Verification: `git diff --check` passed. The user reports 7 local tests passed before this update; this environment has no Python runtime to rerun them.
- Next: Run `python -m src.tokens --input data/demo/sample_phrases.csv` to generate `data/tokens.csv`; Astra runs `make check` during review.
- Files touched: `src/tokens.py`, `tests/test_tokens.py`, `data/demo/sample_phrases.csv`, `STATUS.md`, `HANDOFF.md`.
- Resume: `git switch codex/T-001-tokens`.
