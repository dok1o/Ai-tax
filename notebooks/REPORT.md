# Autonomous Batch Report

## Task status

| Task | Branch / commit | Verification | Pushed | State |
| --- | --- | --- | --- | --- |
| T-001 | `codex/T-001-tokens` / `5b146f2` | 10 tests passed; Ruff passed. `run.py` is absent on this branch. | No; automatic approval review rejected push. | review |
| T-002 | `codex/T-002-prompts` / `f66a762` (includes `7caa2a8`) | `run.py check`: 16 passed; validator: 60 questions, 180 rows, 6 categories. | No; automatic approval review rejected push. | review |
| T-003 | `codex/T-003-collect` / `e2e5956` | `run.py check`: 9 passed; fake collector wrote 360 fixture responses. | No; automatic approval review rejected push. | review |
| T-004 | `codex/T-004-real-run` | Targeted tests: 29 passed; Ruff passed; full `run.py check` stalled after seven test dots and was interrupted at ~10 minutes. | No; auto-review rejected unverified origin | review |

## Key outputs

- `data/tokens.csv` retains the SCHEMA v1 columns. `data/tokens_ratio.csv` stores `tokens_vs_en` separately; its 24 ratios were derived from the saved sample token counts. A fresh tokenization attempt could not fetch the missing `cl100k_base` encoding file because network name resolution was unavailable.
- `data/prompts_draft.csv` contains 60 example questions with `verified_by_native=0`; `data/prompts.csv` contains the 180 v1 rows with `verified_by_native=false`. All 60 questions have kk, ru and en rows across six categories. No native verification is claimed.
- `data/responses.csv` contains 360 fake-client rows. Model IDs begin with `fixture/`; append-only cache and attempt logs are in ignored local `data/raw/` files and are not committed.
- `data/labels.csv`, `data/label_status.csv` and `data/metrics.csv` are fixture outputs. The v1 labels header stays unchanged; unparsed status is explicit in the extra status file. The 12 metrics rows include seeded question-cluster bootstrap intervals, exact paired McNemar p-values and Holm adjustment.

## Real, fixture and missing

- Real: `data/responses.csv` now contains 1,080 human-collected Ollama responses from three models, 60 questions, three languages, two repeats, and temperature 0.5. The real-only labels and metrics were regenerated from this file.
- Real tokenizer rows for the sample phrases remain in `data/tokens.csv`; the ratio file is calculated from those saved counts. They were not freshly recomputed in this run.
- Fixture: offline demo fixtures remain under `data/demo/`. No `fixture/*` model rows were included in the real labels or metrics.
- Missing: native-speaker review and a fresh offline tokenizer recomputation. No API was called and `.env` was not read.

## PowerShell verification commands

Run after assembling the four branches on `dev`:

```powershell
& '.\.venv\Scripts\python.exe' run.py check
& '.\.venv\Scripts\python.exe' -m data.validate_prompts
& '.\.venv\Scripts\python.exe' -m src.tokens --input data/demo/sample_phrases.csv
& '.\.venv\Scripts\python.exe' -m src.collect --input data/prompts.csv
& '.\.venv\Scripts\python.exe' -m src.judge --prompts data/prompts.csv --responses data/responses.csv
& '.\.venv\Scripts\python.exe' -m src.stats --labels data/labels.csv
```

## Push commands blocked by automatic approval review

The review rejected pushes because it could not verify the external GitHub destination `origin`. No retry or alternate push method was used.

```powershell
git push origin codex/T-001-tokens
git push origin codex/T-002-prompts
git push origin codex/T-003-collect
git push origin codex/T-004-judge-stats
```

## Astra review order

Review and merge T-001, T-002, T-003, then T-004. Each branch was cut independently from `dev`; Astra may need to reconcile the task status rows and repeated Requests entries while merging. Review the Requests list for the proposed v2 contract changes before any schema revision.


## T-010 — Definition of Done checker

- Targeted command: `\.venv\Scripts\python.exe run.py test tests/codex/test_done_check.py` — 7 passed; Ruff passed for task files.
- Full command: `\.venv\Scripts\python.exe run.py check` — Ruff passed, 23 tests passed, prompt validator passed (60 questions / 180 rows; native verification 0.000).
- Gate command: `\.venv\Scripts\python.exe run.py done` — expected exit 1; all ten release gates currently fail on missing evidence or human decisions. DATA reports 0/60 verified kk rows and pending D1/D2; MEASUREMENT lacks `data/tokens.csv` and has pending M1; STATISTICS lacks `data/metrics.csv`; MODELS and ROUTER lack saved artifacts; APP lacks offline timing/sign-off; REPRODUCIBILITY lacks `notebooks/rebuild_comparison.csv`; INTEGRITY and PRESENTATION lack human evidence; MERGE has no release tag.
- The checker is read-only and emits one PASS/FAIL row per gate. No live calls or data edits were made.
- Next: Astra reviews this branch; then resolve release evidence and human sign-offs before expecting `run.py done` to pass.
- Push: not retried because automatic approval review previously rejected this external origin as unverified. After approval, run `git push -u origin codex/T-010-done-check`.


## T-012 — Native review worksheet

- Targeted command: `\.venv\Scripts\python.exe run.py test tests/codex/test_native_review.py` — 6 passed; Ruff passed for the task files.
- Full command: `\.venv\Scripts\python.exe run.py check` — Ruff passed, 22 tests passed, prompt validator passed (60 questions / 180 rows; native verification 0.000).
- Export command: `\.venv\Scripts\python.exe -m src.native_review export` wrote 60 rows to the extra `data/native_review.csv` worksheet.
- The human-only import command was not run. `data/prompts.csv` and its existing `verified_by_native` values were not changed. Import logic keeps SCHEMA v1 and serializes new marks as `true`; the numeric 1/0 mismatch and proposed v2 update are recorded under Requests in STATUS.md.
- Next: a Kazakh native speaker completes the worksheet; Astra reviews this branch. Push was not retried after the prior auto-review rejection; after approval, run `git push -u origin codex/T-012-native-review`.

## T-014 — Imported multilingual dataset

- Source `data/incoming/prompts_v2_proposed.csv` is retained for reproducibility. Conversion added `prompt_id` and preserved source `answer_type`, `gold_answer`, and `verified_by_native=0`; output is UTF-8 without BOM in the v2 column order.
- Category mapping: Математика и логика → `math_logic`; Повседневные ситуации → `everyday_situations`; Здоровье и образование (базовое) → `health_education_basics`; Казахстан: история и культура → `kazakhstan_history_culture`; Наука → `science`; География и история мира → `world_geography_history`.
- Validator adaptation: `DRAFT_PATH` now points to archived `data/prompts_draft_old.csv`; no other validator checks were changed.
- Validation command `PYTHONUTF8=1 .\.venv\Scripts\python.exe -m data.validate_prompts --input data/prompts.csv` passed: 60 questions / 180 rows, verified share 0.000. Token command `PYTHONUTF8=1 .\.venv\Scripts\python.exe -m src.tokens --input data/prompts.csv --output data/tokens.csv` used cl100k_base and pinned Qwen revision `5796c1d2033cb6257f44cdffa28957680359236d`.
- Means of `tokens_vs_en`, real, translations not yet native-verified, n=60: cl100k_base kk=2.380282 and ru=1.614634; Qwen/Qwen2.5-0.5B@5796c1d2033cb6257f44cdffa28957680359236d kk=1.995566 and ru=1.318569.
- Targeted test/lint commands: `run.py test tests/codex/test_dataset_import.py tests/codex/test_prompts.py tests/test_schema.py` — 22 passed; Ruff passed.

## T-004 — Real Ollama run (2026-11-27)

The source is the human-collected `data/responses.csv`: qwen2.5:3b, gemma2:2b, and llama3.2:3b; temperature 0.5; two repeats; 60 shared questions in kk, ru, and en. It contains 1,080 rows and no fixture models. All nine model/language cells have n=120 responses from 60 q_id values. `data/labels.csv` contains the frozen labels columns; `data/metrics.csv` contains the metrics. The frozen schema's `data_source` value is `external` for these real observations.

The judge sanity check exposed two parsing gaps: numeric explanations with the result after a phrase such as “составляет” were left unparsed, and Cyrillic renderings of Latin multiple-choice labels were not recognized. The judge now uses the final equation value, extracts a final value after supported result markers, and accepts unambiguous leading Cyrillic choice labels. Ambiguous numeric alternatives remain unparsed. Regression coverage includes `17 + 25 = 42.` and these language cases.

| Model | kk unparsed | ru unparsed | en unparsed |
| --- | ---: | ---: | ---: |
| gemma2:2b | 29/120 (24.2%) | 1/120 (0.8%) | 3/120 (2.5%) |
| llama3.2:3b | 65/120 (54.2%) | 4/120 (3.3%) | 2/120 (1.7%) |
| qwen2.5:3b | 13/120 (10.8%) | 1/120 (0.8%) | 5/120 (4.2%) |

Unparsed rows are counted incorrect in accuracy. The under-5% target is missed for all three Kazakh cells. Most remaining unparsed rows are Kazakh numeric explanations or invalid/unrelated answers; the llama3.2:3b Kazakh cell also has 52 unparsed MCQ rows. No rows were dropped to improve accuracy.

| Model | Language | Accuracy (95% q_id bootstrap CI) | n responses / q_id |
| --- | --- | ---: | ---: |
| gemma2:2b | kk | 0.325 (0.233–0.417) | 120 / 60 |
| gemma2:2b | ru | 0.775 (0.667–0.867) | 120 / 60 |
| gemma2:2b | en | 0.783 (0.683–0.875) | 120 / 60 |
| llama3.2:3b | kk | 0.183 (0.108–0.267) | 120 / 60 |
| llama3.2:3b | ru | 0.592 (0.483–0.692) | 120 / 60 |
| llama3.2:3b | en | 0.808 (0.708–0.892) | 120 / 60 |
| qwen2.5:3b | kk | 0.450 (0.342–0.558) | 120 / 60 |
| qwen2.5:3b | ru | 0.817 (0.717–0.908) | 120 / 60 |
| qwen2.5:3b | en | 0.833 (0.733–0.917) | 120 / 60 |

Accuracy intervals use 1,000 bootstrap draws of q_id clusters (seed 42). Repetitions stay together within each sampled question. No model/language cell reached ceiling accuracy; the highest was 0.833.

| Model | Pair (A–B) | Mean Δ accuracy | Exact McNemar p | Holm p (within model) | n paired q_id |
| --- | --- | ---: | ---: | ---: | ---: |
| gemma2:2b | kk–ru | -0.450 | 5.72e-6 | 1.72e-5 | 60 |
| gemma2:2b | kk–en | -0.458 | 3.59e-5 | 7.18e-5 | 60 |
| gemma2:2b | ru–en | -0.008 | 1.000 | 1.000 | 60 |
| llama3.2:3b | kk–ru | -0.408 | 6.10e-5 | 1.22e-4 | 60 |
| llama3.2:3b | kk–en | -0.625 | 9.31e-10 | 2.79e-9 | 60 |
| llama3.2:3b | ru–en | -0.217 | 0.0313 | 0.0313 | 60 |
| qwen2.5:3b | kk–ru | -0.367 | 7.63e-5 | 1.53e-4 | 60 |
| qwen2.5:3b | kk–en | -0.383 | 1.10e-5 | 3.29e-5 | 60 |
| qwen2.5:3b | ru–en | -0.017 | 0.453 | 0.453 | 60 |

For each model, the exact McNemar test uses majority-aggregated repetition outcomes per q_id; tied repetitions do not contribute discordant pairs. Holm correction covers that model's three language pairs. Δ is the mean question-level accuracy difference, A minus B. These results show lower Kazakh accuracy for every model; they remain specific to this 60-question set.

| Model | Language | Answer | Extracted | Gold | Label |
| --- | --- | --- | --- | --- | --- |
| qwen2.5:3b | ru | `20` | `20` | `20` | true |
| gemma2:2b | ru | `D` | `D` | `D` | true |
| gemma2:2b | kk | `Тіктөртбұрыштың периметрі: 7 см + 5 см = 12 см; 12 см × 2 = 24 см` | `24` | `24` | true |
| qwen2.5:3b | en | `B` | `B` | `B` | true |
| gemma2:2b | ru | `C` | `C` | `C` | true |
| gemma2:2b | kk | `B` | `B` | `B` | true |
| gemma2:2b | en | `C` | `C` | `C` | true |
| gemma2:2b | kk | `B` | `B` | `B` | true |
| qwen2.5:3b | kk | `A` | `A` | `A` | true |
| gemma2:2b | en | `D` | `D` | `D` | true |

The sample contains 10 deterministic random parse-success rows (seed 42), selected before comparing label values. Single-letter answers occurred only for MCQ rows (0 non-MCQ single-letter rows). One Gemma Kazakh response ends with a Russian “не знаю, как перевести” after giving a conversion; this is an isolated language-mismatch/refusal-like response, not a cell-wide refusal. `llama3.2:3b` rows have `model_version=unknown`; the other two models have recorded versions.

Verification: targeted judge/statistics tests — 29 passed; Ruff passed. Full `run.py check` passed Ruff, then pytest made no further progress after seven test dots for ~10 minutes; it was interrupted with no assertion failure reported. Auto-review rejected push because it could not verify `origin` as the authorized destination. After the human confirms that destination, the command is `git push -u origin codex/T-004-real-run`.
