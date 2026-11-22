# T-010 — Automate the project Definition of Done

- Agent: Codex (Luna). Branch: `codex/T-010-done-check` from the latest dev.
- Goal: implement the read-only PowerShell-compatible command `python run.py done`.
- Inputs: DONE.md criteria and confirmed thresholds, CSV contracts, saved real data, logs, folds, manifests, repo history and release sign-offs.
- Owned paths: Astra delegates the done subcommand in run.py for this card, plus src/done_check.py and tests/codex/test_done_check.py.
- Output: print one row per gate with PASS/FAIL, evidence path or Git ref, and the specific missing or failed check. Exit nonzero while any required evidence is missing or fails.
- Check every automatable criterion, including per-cell coverage and parsing, grouped splits, baselines, matched costs, offline app timing, rebuild comparisons, provenance, history scans and tag ancestry.
- For human criteria, check recorded evidence and threshold confirmations at the reviewed revision. Identify pending human decisions explicitly. Only the human can supply sign-off.
- Acceptance: targeted tests cover complete, missing, malformed, fixture-only and mixed-provenance inputs. A missing file or pending confirmation never yields PASS.
- Acceptance: test failures identify the gate and evidence path. Tests run offline. The checker performs no live collection, Git writes or credential disclosure.
- Integration: run `python run.py test tests/codex/test_done_check.py` and `python run.py check`; update STATUS.md and HANDOFF.md.
- Token budget: 6,000 agent tokens.
