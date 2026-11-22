# Definition of Done for the 27 November 2026 demo and pitch

## 1. Scope and cut order

**In scope.** Measure token counts, tokens per word and answer accuracy for matched Kazakh, Russian and English questions. Compare token-cost and error-risk predictors with simple baselines. Evaluate routing on observed calls and show saved results in an offline dashboard. Deliver an A0 poster, a timed pitch and written Q&A.
**Out of scope.** Production serving and claims about providers, languages or populations outside the measured sample.
**Cut in order.** First drop training our own tokenizer (keep tiktoken and a pinned open tokenizer). Next drop code-switching experiments. Then drop the third answering model (the stronger route can use the stronger of the two measured models). Always retain measurement, the token-cost predictor and the dashboard.

## 2. Release gates

Evaluate the candidate commit on `dev`. Section 5 records its current state. A pass requires evidence from that commit and its real observations. Review-branch fixtures can demonstrate a test or interface. Record human confirmations, evidence paths and dates in `RELEASE.md`. Paths identified as pending below are release deliverables.

| Gate | Owner | Pass condition | Command or file evidence |
| --- | --- | --- | --- |
| DATA | Codex / human | The validator exits 0. Meet D1 for distinct questions, translations and categories, and D2 for native review. Count a kk row as verified only when a native speaker has reviewed both the prompt and its gold answer. | `python -m data.validate_prompts`, `data/prompts.csv` grouped by q_id, lang and category, and the human confirmation in `RELEASE.md`. |
| MEASUREMENT | Codex | For every included q_id, obtain ≥2 distinct real models × all 3 languages × ≥2 distinct repeats per cell. Join each response to one successful raw call by (model, prompt_id, lang, rep). Log every attempt and outcome, UTC date/time, model/version, temperature, input/output tokens and latency. Mark unavailable failed-attempt metadata explicitly. Grade every response, keep unparsed answers in the denominator and meet M1. Save measured token counts and fertility for the same prompts. | `data/prompts.csv`, `data/tokens.csv`, `data/responses.csv`, local `data/raw/attempts.jsonl` and `data/raw/responses.jsonl`, `data/labels.csv`, `data/label_status.csv`. |
| STATISTICS | Codex | Report accuracy by model and language with question-cluster bootstrap CIs. Pair languages within model and q_id using one prespecified matched repetition per question for exact McNemar. Apply Holm across the declared comparison family. Show distinct questions and observations for each estimate, contrast, plot and table. State exclusions and label small-sample estimates preliminary. | `src/stats.py`, `data/metrics.csv`, `data/labels.csv` and the comparison-family definition in the report. Use S1 for CIs. |
| MODELS | ARENA | A leakage test finds zero shared q_ids between training and evaluation in every fold, including translations and repeats. Use only pre-call features. On identical held-out rows report token-count MAE versus the training per-language/tokenizer mean and error-risk Brier score versus the training error rate. Show n, baseline, model and signed difference for each. A tie or loss passes when reported explicitly. | Saved folds, held-out predictions and metric tables under `models/token_cost/` and `models/error_risk/` (pending). Run the integrated leakage tests with `python run.py test`. |
| ROUTER | ARENA | Compare the frozen routing policy with **always_direct** and **always_via_english** on identical eligible held-out questions. Select thresholds on training questions. Plot observed accuracy against cost, counting translation and all model legs. State cost unit, n and excluded routes. At each baseline's matched accuracy report paired mean cost saved (baseline minus router) with a q_id-bootstrap CI (S1). If held-out accuracy cannot be matched or the CI includes zero, report that outcome explicitly. An honest null passes. | Saved per-route observations, decisions, cost-quality curve and savings table under `models/router/` (pending). Record model prices and query dates when reporting money. |
| APP | ARENA | On the demo laptop with external networking disabled and credentials unset, launch from saved results within A1. Render each of kk, ru and en. A real-only run has no “EXAMPLE DATA” banner. Loading any fixture, including a fixture model in a production-named CSV, visibly labels the displayed examples. | `python run.py dashboard`, `app/dashboard.py`, saved inputs under `data/demo/` and a timed smoke record for both provenance cases in `RELEASE.md`. |
| REPRODUCIBILITY | Codex | In a fresh clone of the candidate, create a Python 3.11 venv, install requirements and pass the full check. With the same saved inputs and SEED = 42, rebuild all measurement, statistical, predictor and routing tables offline. Compare keys, counts and values with the release tables and record the exact commands and input versions. | `py -3.11 -m venv .venv`, then in the venv `python -m pip install -r requirements.txt` and `python run.py check`. Seed in `src/common/config.py`; rebuild commands and comparison results in `RELEASE.md` (pending). |
| INTEGRITY | Codex / human | Trace every reported result to real observations and a generating script. Keep all fixture-derived numbers inside explicitly labelled fixture material. Scan all reachable Git history for secrets and tracked .env files, validate neutral topics, and record an opened, verified source for every citation. Record actual model names, versions and query dates for every reported model. | `git log --all --full-history -- .env ':(glob)**/.env'`, redacted history-wide secret scan, `data/banned_topics.txt`, prompt validator and raw call log. Human records citation URL, claim checked and date opened in `RELEASE.md`. |
| PRESENTATION | human | Map every quantitative claim on the poster and in the pitch to its repo file, row/filter and generating command. Reconcile units, rounding and n. Rehearse a 2–3 minute pitch and record its duration. Write answers on originality, language coverage, uncertainty, leakage, baselines and null results. | `Language_Tax_Poster.pdf`; pitch text, claim-to-file mapping, rehearsal duration and judge Q&A in `RELEASE.md` (pending). |
| MERGE | human | After the preceding nine gates pass on the same dev commit, review the diff and run the offline demo by hand. The human then merges dev into main and tags the release. Pass when the tag identifies the released main commit and that commit contains the reviewed dev tip. | `git diff main...dev --stat` before merge, human smoke sign-off in `RELEASE.md`, then `git merge-base --is-ancestor <reviewed-dev-sha> <release-tag>` and the tagged main commit. |

### Proposed thresholds for sign-off

| ID | Proposed value and reasoning | Decision |
| --- | --- | --- |
| D1 | Minimum n = 60 distinct q_ids. The current validator expects 10 questions in each of six neutral categories and kk/ru/en for every question (180 rows). This preserves balanced coverage. | proposed, human confirms |
| D2 | Minimum native-verified share = 100% of included kk rows (60/60 at D1). Each question affects all paired accuracy comparisons, and the review set is small enough to enumerate. Publish the numerator and denominator. | proposed, human confirms |
| M1 | Unparsed answers <5% within each model × language cell, divided by all its collected responses. A cell-level limit exposes language-specific grading failures that a pooled rate can hide. | proposed, human confirms |
| S1 | Use 95% CIs for accuracy and cost savings, resampling whole q_ids to keep translations and repeats together. A common level makes uncertainty comparable across results. | proposed, human confirms |
| A1 | Cold start <15 seconds from launching the process to a populated, interactive results view on the demo laptop with dependencies installed. This leaves most of the pitch for the comparison. | proposed, human confirms |

The two-model, three-language, two-repeat floor comes from the demo brief. At D1 it requires at least 720 distinct successful observations. The fixed seed, pitch duration and freeze time are project requirements.

## 3. Contest map

| Criterion | Exact evidence and what it supports |
| --- | --- |
| Originality | `data/prompts.csv` and `data/SCHEMA.md` show the aligned trilingual questions and shared IDs. `95c3cfe:models/REPORT.md` describes the team's features and routing formulation and identifies the standard algorithms. |
| Technical complexity | `a4b6f16:src/stats.py`, `95c3cfe:tests/arena/test_cost_split.py`, `95c3cfe:tests/arena/test_risk.py` and `95c3cfe:tests/arena/test_router.py` show paired inference, grouped evaluation, SHAP and routing tests. `RELEASE.md` records known defects. Real-data evidence is pending the STATISTICS, MODELS and ROUTER gates. |
| Demo quality | `3b3ee47:app/main.py` and `3b3ee47:tests/arena/test_app.py` show the offline fixture interface and empty state. `RELEASE.md` records its branch smoke test. A timed real-data run on dev is pending. |
| Q&A | `RELEASE.md` records actual sample coverage, fixture limitations and review findings. `a4b6f16:notebooks/REPORT.md` documents data provenance. The rehearsed answers and their evidence links belong in the pending judge Q&A section of `RELEASE.md`. |

## 4. Timeline

- **Thu 2026-11-26 at 12:00, Asia/Qyzylorda** — code freeze. After the freeze, only Sol may make bug fixes. Astra reviews the fixes on dev and reruns affected gates.
- **Fri 2026-11-27** — human runs the offline demo by hand, gives the rehearsed pitch and completes the release procedure once the gates pass.

## 5. Current state

Snapshot of dev at `383e4e1` from `STATUS.md`, `RELEASE.md` and the two branch reports cited above. **fail** records a known unmet condition, **doing** records incomplete verification, **todo** awaits work, and **pass** requires complete evidence.

| Gate | State | Blocker |
| --- | --- | --- |
| DATA | fail | Validator passed for 60 questions and six balanced categories. Native verification is 0/60 kk rows. |
| MEASUREMENT | fail | The reported 360 responses came from the fake client. Real call coverage, complete logs and the real unparsed rate are missing. |
| STATISTICS | fail | Branch metrics are fixture-only. Numeric grading and the treatment of repeated pairs in McNemar are open review findings. |
| MODELS | fail | Grouped folds exist on the ARENA branch. Real held-out comparisons are missing, and cost lint and risk figure generation failed review. |
| ROUTER | fail | The fixture run selected its threshold on evaluation outcomes. Correctness decoding is also under review. Real paired costs and savings CIs are missing. |
| APP | fail | The dev launcher stops pending T-006. The fixture branch rendered offline, but provenance handling and timed real-data startup remain open. |
| REPRODUCIBILITY | doing | The full dev check passed with 16 tests. A fresh-clone rebuild of all real tables has no recorded result. |
| INTEGRITY | doing | Review scanned changed files. History-wide scanning, citation verification and live model provenance still need evidence. |
| PRESENTATION | todo | The poster exists. Claim reconciliation, timed rehearsal and written judge answers have no recorded sign-off. |
| MERGE | todo | Upstream gates remain open. Human diff review, hand-run demo, merge and release tag are pending. |
