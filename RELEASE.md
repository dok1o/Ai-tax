# Release status after T-013

The integration includes T-001 through T-012. T-009 was already on dev. The fetched ARENA tip `dfaf97d` contains its original T-005 through T-008 work, subsequent fixes and T-011. The runner, native-review export and human-only import now use the integrated contracts. The v2 migration changed the encoding of existing marks without verifying any answers.

## Remaining work

- A native speaker must review the 60 Kazakh questions and golds. Four former yes/no questions now have explicit A/B options. Reconcile any earlier worksheet against the current text before human import.
- T-001 still needs a cold-cache offline guard and a measured run over all production prompts. Its saved sample counts and ratios are under data/demo/. T-003 caches answers with missing usage, but overlapping live collectors still need duplicate-call protection. Keep live collection single-process until that review is complete.
- Supply credentials and model versions locally, approve a budget, collect real observations and record all query metadata. Rebuild labels, metrics, cost/risk models and router comparisons from real data. Fixture observations and trained fixture artifacts only support the demo.
- Reconcile the paired test with DONE.md before reporting live significance. The merged stats fix compares per-question mean correctness across repetitions. DONE.md specifies one prespecified matched repetition.
- Smoke-test and time the offline T-011 dashboard with the release inputs on the demo laptop. Confirm data provenance and all thresholds. Record the fresh-clone rebuild, history and citation checks, poster/pitch traceability, rehearsal, human diff review and release tag.
- Run `python run.py done` to inspect missing release evidence. Passing `python run.py check` covers software tests and the prompt validator.

## Historical review before consolidation

**Decision:** Keep the release on dev. The prompt template is integrated. The offline dashboard and live-data pipeline still have open review findings.

## Reviewed snapshots

T-002 was merged with `--no-ff` at `1bf4d35`. T-001 through T-004 were reviewed at the Codex branch tips. ARENA submitted T-005 through T-008 as successive commits on `origin/arena/ad3092b4-ai-tax`. The ARENA results below cover each cumulative snapshot. STATUS.md names the intended individual work branches for resubmission.

| Task | Reviewed commit | Ruff | pytest | Disposition and reason |
| --- | --- | --- | --- | --- |
| T-001 | `5b146f2` | Pass | 10 passed | Doing. Tokenizer initialization can fetch resources without `--live`. The production token CSV contains sample IDs outside the prompt dataset. |
| T-002 | `f66a762` | Pass | 16 passed | Merged. The validator confirms 60 questions, 180 localized prompts and ten questions in each of six categories. The builder refuses to overwrite existing prompts. |
| T-003 | `e2e5956` | Pass | 9 passed | Doing. A stub returning an answer without usage was called twice for one key across two runs. Live environment variables also differ from the template. |
| T-004 | `a4b6f16` | Pass | 16 passed | Doing. With gold 12, the grader rejects `7 + 5 = 12` and accepts `12 - 1 = 11`. Exact McNemar treats repetitions of the same question as independent pairs. |
| T-005 | `c121aa3` | 6 findings | 13 passed | Doing. The fixture generator fails lint at this tip. The cost submission also introduces T-007 risk training and SHAP before its review. |
| T-006 | `3b3ee47` | Pass | 22 passed | Doing. The page renders in isolation. Its fixtures and predictor imports depend on the unmerged stack. Its loader calls every canonical data CSV real, including fake-client observations. |
| T-007 | `b323912` | Pass | 31 passed, 1 failed | Doing. SHAP figure generation needs undeclared matplotlib. Training can mark fake-client labels from data/labels.csv as real. |
| T-008 | `95c3cfe` | Pass | 44 passed, 2 failed | Doing. Numeric correctness flags are inverted. Threshold selection reuses evaluation outcomes. Both risk and router figure tests require matplotlib. |

## Integration checks

- On Windows with Python 3.11, `python run.py check` passed Ruff, all 16 tests and the prompt validator. The verified share is zero.
- The offline smoke command `python run.py dashboard` exits with “The offline dashboard is pending T-006.” At the T-006 snapshot, the same page entry point rendered in the Streamlit test harness with external connections blocked, credentials cleared and the example-data banner visible.
- The diff review found no tracked .env files or common credential signatures in changed files. The T-003 CLI defaults to its fixture client and gates model calls on `--live`. No live collection ran.
- Cost and risk use grouped folds by q_id, and risk explanations use held-out rows. The router selects its final threshold on the same held-out outcomes used to report its savings. Task cards record corrections and ownership issues for resubmission.
- Treat the branch reports and saved ARENA artifacts as fixture evidence. This review establishes no measured model cost, accuracy or routing advantage on live data.

## Human work before release

1. Have a Kazakh native speaker verify the kk prompts and gold answers for meaning and correctness. Record their corrections and verification marks.
2. After T-003 passes review, supply credentials locally for the two intended model versions and confirm the collection budget.
3. Authorize the explicit live collection, retain the append-only cache and call logs, then regenerate token counts, grades and paired metrics from the observed responses.
4. Rerun the ARENA cost, risk and routing tasks on real data. Supply observed route legs and model prices for cost comparisons. Verify fixture provenance, rerun the full check and smoke-test the integrated dashboard offline before promoting dev.
