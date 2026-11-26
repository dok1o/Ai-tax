# Release status

## Round 8 integration, 2026-11-27

By human decision Claude Code acts as Astra. The human also waived the same-session review rule for Sol `7eaabe4`, which this session wrote.

| Merge order | Submission | Checks on the integrated tree |
| --- | --- | --- |
| 1 | ARENA `5cea469` (T-016 offline tokenizers and redesign; T-005/T-007/T-008 rerun on real v2.1 data) as `b9e4700` | `python run.py check`: Ruff clean, 225 passed, 2 failed (the known warm-cache tokenizer tests). STATUS.md conflict resolved with the dev side. |
| 2 | Sol `fix/T-016-hermetic` `7eaabe4` as `c6d89fb` | `python run.py check`: Ruff clean, 227 passed, validator 60 questions / 180 rows. Changes only `app/tokenizers.py` and `tests/arena/test_tokenizers.py`, as the human allowed. |

| Check | Result |
| --- | --- |
| Paths and secrets | ARENA changes stay in app/, models/, src/features.py, src/router.py, tests/arena/ and HANDOFF.md. No key signatures. The only URL added is tiktoken's cl100k_base blob path, used as the cache key; its download path is blocked. |
| Network | No network path without `--live`. Tokenizers load with `local_files_only=True`, forced hub offline mode and a tiktoken download tripwire; the tests trip on sockets, `urlopen` and `requests`. |
| Labels and native marks | Fixture labels are resolved only in example mode and tests. `verified_by_native` sums to 0 over 180 rows. |

The router null result against `always_via_english` and its three caveats from round 7 stand.

## Left for the human

1. Update the local dev checkout: `git switch dev` and `git pull`.
2. Start the offline demo with `python run.py dashboard` on the demo laptop (warm tokenizer caches, no network) and time it.
3. Native verification: a Kazakh native speaker reviews the 60 kk questions and gold answers; import their marks through the human-only import. All `verified_by_native` flags are 0 now, so every result is preliminary.
4. Poster numbers: take every figure from `models/REPORT.md`, `models/*_metrics.json` and `data/metrics.csv` only, with n and CI. Write TODO where no file backs a number.
5. When satisfied, merge dev into main yourself.

## Round 7 review, 2026-11-27

ARENA `5cea469` (T-016 and T-005/T-007/T-008 rerun on real v2.1 data) was not merged. Reason: `python run.py check` on the owner's machine with warm tokenizer caches failed 2 of 227 tests. By human decision ARENA Sol fixes both tests on the owner's machine before resubmission.

| Check | Result |
| --- | --- |
| Paths and secrets | Changes stay in app/, models/, src/features.py, src/router.py, tests/arena/, HANDOFF.md and STATUS.md. No credential files or key signatures; tests use a dummy key. |
| Network | Loaders run with `local_files_only=True`, forced hub offline mode and a tiktoken download tripwire. No network path without `--live`. |
| Labels and native marks | Fixture labels are resolved only in example mode and tests. `verified_by_native` stays 0. |
| `python run.py check` on the trial merge | Failed: `test_the_tiktoken_loader_refuses_to_download` and `test_the_subword_loader_never_connects_on_a_cold_cache` (both DID NOT RAISE). Ruff clean; 225 passed. Trial merge aborted, dev unchanged. |

Router decision: keep the null result. The router ties `always_via_english` (0 tokens, 0 accuracy difference, CI95 [0, 0], n = 180 keys, preliminary). No code change. Report it with three caveats:

1. The `via_english` cost is a lower bound: `data/routes.csv` does not exist, so legs are derived and translation tokens are 0.
2. No model prices: all costs are in tokens.
3. 106 of 360 Kazakh answers (29%) did not parse, which lowers kk accuracy.

## Round 6 review, 2026-11-27

T-016 `df87b23` was returned to ARENA and not merged. Reason: two T-016 tests assume a cold tokenizer cache; ARENA (Sol) must make them hermetic. The network guard itself works.

| Check | Result |
| --- | --- |
| Paths and secrets | Changes stay in app/, tests/arena/, HANDOFF.md and STATUS.md. No credential files or key signatures. Tests use a dummy key. Live translation still needs `LT_LIVE` and a key. |
| Corrupt-cache regression test | `test_corrupt_caches_never_open_a_socket` exists and passes. |
| Offline happy path | With sockets, `urlopen` and `requests` blocked, `load_backends()` loaded `cl100k_base` and the pinned Qwen snapshot from the warm local cache with zero network attempts. |
| `python run.py check` on the trial merge | Failed, 2 of 215 tests. `test_the_tiktoken_loader_refuses_to_download` passes alone but fails in the full run because tiktoken keeps loaded encodings in process. `test_the_subword_loader_never_connects_on_a_cold_cache` loads the real Qwen snapshot because setting `HF_HOME` after import does not move the hub cache. |

Fixture labels and `verified_by_native` are untouched; the branch changes no data/ or models/ files.

## Round 5 integration, 2026-11-27

By human decision Claude Code acts as Astra for this sprint.

| Merge order | Submission | Checks on the integrated tree |
| --- | --- | --- |
| 1 | Schema v2.1 `9e000c9` | `python run.py check` passed, 179 tests. The metrics enum adds `accuracy_strict` and `accuracy_parsed`. `src/common/schema.py` and `tests/test_schema.py` validate metric names in data/metrics.csv and its fixture. |
| 2 | T-004 `55fc27a` | `python run.py check` passed, 192 tests. 1,080 responses from three Ollama models, 360 each, with `data_source=external`. The nine `accuracy_strict` and nine `accuracy_parsed` rows now fit the contract. |
| 3 | T-017 `662b18c` | `python run.py check` passed, 192 tests. q_038-ru now uses the masculine participle. The report notes that six responses used the earlier wording. |

All diffs stayed within Codex paths plus HANDOFF.md and STATUS.md. The scan found no credential files or common key signatures. The changed source adds no network calls. All 180 prompt rows keep `verified_by_native=0`.

T-016 `1a01017` was not merged. Its STATUS row is still blocked and the branch has no regression test for zero network requests on a corrupt tokenizer cache.

The release checker in `src/done_check.py` still reads only `accuracy`, so `python run.py done` will reject the v2.1 metrics until Codex updates it.

## Round 4 review, 2026-11-27

The three submissions are pending in the approved order. The root Streamlit configuration matches the theme in T-016. On the dev baseline with that configuration, `python run.py check` passed Ruff, 174 tests and validation of 60 questions across 180 prompt rows. The review results below refer to the submitted snapshots.

| Submission | Decision and evidence |
| --- | --- |
| T-016 `1a01017` | Returned to ARENA. In a fresh Python process with a temporary cache, the empty cache triggered zero HTTP requests. Placing truncated bytes at the expected cl100k_base cache key triggered one HTTP request in `load_backends()`. The probe intercepted the request before transmission. Require a regression test that exercises the damaged cache and disables the backend with zero requests. |
| T-017 `9aadb18` | Held behind T-016. The diff changes three localized prompt texts and regenerates their token data. All 180 native verification flags remain zero. The revised q_038-ru also needs masculine agreement between the noun for “rim” and its participle. Preserve the exact wording used for previously collected observations. |
| T-004 `55fc27a` | Returned to Codex. The 1,080 distinct responses cover three Ollama model IDs, 360 each. All 1,080 grades match the v2 labels header, keys, prompt metadata and parse-status rules. The v2 metrics enum admits `accuracy` and `mcnemar_exact`. Nine `accuracy_strict` and nine `accuracy_parsed` rows fall outside it. Restore `accuracy` for strict grading in the canonical output and place parsed-only estimates in a supplemental output. The release checker reads `accuracy`. |

The reviewed changes stay within the respective app, data, analysis, test and handoff paths. The credential scan found no common key signatures or credential filenames in the changes. The cost, risk and router fixture manifests retain their provenance flags. The model IDs in the submitted responses are `gemma2:2b`, `llama3.2:3b` and `qwen2.5:3b`.

## Round 3 integration, 2026-11-26

| Merge order | Reviewed submission | Checks on the integrated tree |
| --- | --- | --- |
| 1 | ARENA Sol `8548911` | `python run.py check` passed, 158 tests. The human approved review without a separate STATUS row. |
| 2 | T-014 `55d343c` | `python run.py check` passed, 163 tests. The 360 measured token rows cover all 180 prompt IDs across two tokenizers. |
| 3 | T-015 `c5a013e` | `python run.py check` passed, 174 tests. The provider tests use mocks. |

Each check passed Ruff and validated 60 questions in three languages. All 180 imported rows have `verified_by_native=0`. The ARENA merge ends at the approved commit. T-016 remains pending.

Astra reviewed the changed paths and approved the T-015 addition of `google-genai` to the shared requirements. The changed paths and diffs contained no tracked credential files or common key signatures.

With external connections and DNS blocked, the collector ran each of its three provider selections without `--live` on all 180 prompts. Each run recorded fixture provenance and made zero connection attempts. The dashboard displayed its example-data banner with the integrated inputs. The router manifest carries `fixture: true`. Live model collection still requires human approval.

## T-013 integration

The integration includes T-001 through T-012. T-009 was already on dev. The fetched ARENA tip `dfaf97d` contains its original T-005 through T-008 work, subsequent fixes and T-011. The runner, native-review export and human-only import now use the integrated contracts. The v2 migration changed the encoding of existing marks without verifying any answers.

## Remaining work

- The saved native-review worksheet shares zero question IDs with the T-014 dataset. Preserve the old worksheet and export a fresh one for a native speaker to review all 60 Kazakh questions and gold answers before human import.
- T-001 still needs a cold-cache offline guard. A round 3 probe blocked a DNS attempt during tokenizer initialization with an empty tiktoken cache. T-014 supplies the measured production counts in data/tokens.csv and data/tokens_ratio.csv. T-003 caches answers with missing usage, but overlapping live collectors still need duplicate-call protection. Keep live collection single-process until that review is complete.
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
