# ARENA batch report: T-005 → T-008

## Preflight

Python 3.11.2 in `.venv` (created with `python3 -m venv .venv`), packages from
`requirements.txt` plus `streamlit 1.67.0`, `shap 0.51.0`, `scikit-learn 1.9.1`,
`lightgbm 4.7.0`, `pandas 3.1.1`, `ruff 0.17.6`. `pytest` and `ruff` run; every
number below was produced by an executed command. Push to `origin` works.

## Task table

| Task | Branch | Tests passed/failed | State | Data used |
| --- | --- | --- | --- | --- |
| T-005 cost predictor | `arena/ad3092b4-ai-tax` (see branch note) | 9 / 0 (`tests/arena/test_cost_split.py`) | review | fixture |
| T-006 app skeleton | `arena/ad3092b4-ai-tax` | 14 / 0 (`tests/arena/test_app.py`, also covers T-007/T-008 wiring) | review | fixture |
| T-007 error risk + SHAP | `arena/ad3092b4-ai-tax` | 9 / 0 (`tests/arena/test_risk.py`) | review | fixture |
| T-008 router | `arena/ad3092b4-ai-tax` | 14 / 0 (`tests/arena/test_router.py`) + 1 lint guard (`tests/arena/test_lint.py`) | review | fixture |
| Whole repo | — | 51 / 0 (`make check`, includes codex `tests/test_schema.py`) | review | fixture |

Branch note: the platform pinned this session to `arena/ad3092b4-ai-tax`, cut from
`e47a220` of `dev`. It refused to create `arena/T-005-cost-predictor`,
`arena/T-006-app-skeleton`, `arena/T-007-error-predictor` or `arena/T-008-router`,
so the four tasks are four commits on one pushed branch instead of four stacked
branches. Commit order is T-005 → T-006 (plus one lint fix) → T-007 → T-008, so
Astra can review or cherry-pick them in that order.

## Review fixes (Sol run, same branch)

| Finding | Root cause | Fix | Regression test |
| --- | --- | --- | --- |
| Router picked its threshold on the evaluation data | `pick_threshold` ran on the full held-out sweep | `select_threshold` + `nested_decisions`: every outer fold is routed with the threshold chosen on the other folds only; the full-data value is stored as `deployment_threshold`, for the app alone | `test_threshold_selection_ignores_the_held_out_fold` (corrupting fold 0 outcomes and costs leaves its threshold untouched), `test_nested_decisions_cover_every_key_once` |
| Correctness decoding inverted numeric flags | `is_incorrect = values.isin({"false", "1"})` read `1` as incorrect | `decode_correctness` maps `true`/`1` to correct and `false`/`0` to incorrect and rejects anything else, naming the offending values | `test_decode_correctness_reads_booleans_and_flags_alike` |
| Cost-predictor lint errors | Long lines, an unused method argument, a shadowed loop variable; `make check` uses ruff defaults that skip E501 | ARENA files are clean under `E,F,W,I,UP,B,SIM,PLW,ARG,RET,ISC` (E402 and C420 ignored for documented reasons); `decide` was vectorized on the way, 86 s to 5 s for the router tests | `tests/arena/test_lint.py` runs that ruff selection as a test |
| Risk figures failed to generate | `matplotlib` is only a shap extra (`shap[plots]`) and is absent from `requirements.txt`, so the import died after training | `save_shap_summary` and `save_curve_figure` return `None` without matplotlib; the run still writes CSV, JSON, SHAP tables and joblib models and records a note in the manifest | `test_risk_survives_a_missing_matplotlib` |

Nested selection did not move the fixture numbers: all five folds independently
select threshold 0.95, so cost and accuracy are unchanged. The fix removes the
selection leak; it does not rescue this fixture's outcome.

## Key metrics (all FIXTURE data, n small → preliminary, not results)

Fixtures: `models/fixtures/make_fixtures.py`, SEED 42, 60 questions × 3 languages,
2 tokenizers, 2 answering models, 3 strategies. Every artifact name ends in
`_fixture` and every manifest carries `"fixture": true`.

T-005 token cost, out-of-fold over 5 GroupKFold folds by `q_id`, n = 180 prompts
per tokenizer (60 questions × 3 languages):

| Model | Tokenizer | MAE mean ± std | R² mean ± std | n |
| --- | --- | --- | --- | --- |
| lang_mean (baseline) | hf bge-m3 | 10.5787 ± 0.0946 | 0.0612 ± 0.0028 | 180 |
| linear | hf bge-m3 | 0.5840 ± 0.0782 | 0.9966 ± 0.0009 | 180 |
| lightgbm | hf bge-m3 | 0.5379 ± 0.0888 | 0.9971 ± 0.0009 | 180 |
| lang_mean (baseline) | tiktoken cl100k | 11.9194 ± 0.1647 | 0.2571 ± 0.0058 | 180 |
| linear | tiktoken cl100k | 0.7299 ± 0.0504 | 0.9968 ± 0.0005 | 180 |
| lightgbm | tiktoken cl100k | 0.5657 ± 0.0647 | 0.9980 ± 0.0003 | 180 |

Honest answer to "does the model beat the per-language mean baseline?": yes on this
fixture, by roughly 20× in MAE, but that is a property of the generator, where token
counts are an almost deterministic function of character and word counts. It says
nothing about real prompts. Measured fertility (diagnostic only, never a feature):
cl100k kk 3.44, ru 3.00, en 1.86; bge-m3 kk 2.83, ru 2.57, en 1.97 (n = 60 prompts
each) — the Kazakh language tax shows up here even in synthetic data.

T-007 error risk, target `is_incorrect = 1 - is_correct`, out-of-fold over 5 folds
by `q_id`, n = 360 observations (60 questions, 105 incorrect):

| Model | ROC-AUC pooled | ROC-AUC fold mean ± std | Brier pooled | Accuracy | n |
| --- | --- | --- | --- | --- | --- |
| train_rate (baseline) | 0.4435 | 0.5000 ± 0.0000 | 0.2076 | 0.7083 | 360 |
| logistic | 0.6398 | 0.6475 ± 0.0429 | 0.1938 | 0.7306 | 360 |
| lightgbm | 0.6204 | 0.6285 ± 0.0546 | 0.2177 | 0.6694 | 360 |

Per language (logistic, pooled, n = 120 each): kk AUC 0.6544 (52 incorrect),
ru 0.5057 (32), en 0.4687 (21). Within-language discrimination is weak and below
chance for English: most of the pooled signal is the between-language gap. SHAP mean
|value| (logistic): `n_words` 0.523, `n_chars` 0.476, `mean_word_chars` 0.391,
`lang=kk` 0.218. Figures: `models/figs/shap_risk_logistic_fixture.png`,
`models/figs/shap_risk_lightgbm_fixture.png`. WAITING FOR `data/labels.csv`.

T-008 router, evaluated only on held-out questions (5 GroupKFold folds by `q_id`),
n = 180 question-language keys (60 questions), realized tokens and realized
correctness of the chosen route, cost estimate from T-005 (lightgbm, bge-m3) and
risk estimate from T-007 (logistic, best pooled AUC), both out-of-fold, with
per-strategy multipliers and offsets fitted on the other folds only:

| Policy | Mean tokens | Accuracy | n |
| --- | --- | --- | --- |
| always_direct | 95.828 | 0.628 | 180 |
| always_via_english | 170.650 | 0.750 | 180 |
| always_stronger_model | 97.000 | 0.811 | 180 |
| router @ threshold 0.95 | 96.05 | 0.733 | 180 |

Cost saved at the baseline accuracy, bootstrap 95% percentile CI over 1000
resamples of question ids (SEED 42):

| Baseline | Tokens saved | % saved (CI95) | Router accuracy | Baseline accuracy |
| --- | --- | --- | --- | --- |
| always_direct | −0.2 | −0.2% (−2.2 … 1.8) | 0.733 | 0.628 |
| always_via_english | 74.6 | 43.7% (42.2 … 45.1) | 0.733 | 0.750 |
| always_stronger_model | 0.9 | 1.0% (−1.1 … 2.8) | 0.733 | 0.811 |

Reading: on this fixture the router buys +0.105 accuracy over always_direct at the
same cost, and is 43.7% cheaper than always_via_english at −0.017 accuracy. The
threshold is selected by a nested split: every held-out fold is routed with the
largest swept threshold whose accuracy on the other folds is not below
always_direct, and all five folds chose 0.95 (`thresholds_per_fold` in
`models/router_metrics_fixture.json`). `deployment_threshold` repeats the rule on
all rows and is only what the app uses. The interval covers resampling noise only. Sweep range:
cost 96.0–106.9 tokens, accuracy 0.733–0.789 over 19 thresholds
(`models/router_curve_fixture.csv`, `models/figs/router_curve_fixture.png`).

Standard vs ours. Standard: GroupKFold, linear and logistic regression, LightGBM,
MAE, R², ROC-AUC, Brier score, calibration curves, SHAP, bootstrap percentile
intervals, Streamlit. Ours: the trilingual pre-tokenization feature set (including
the Kazakh-specific letter share), the per-language mean and training-rate
baselines, the route formulation (direct / via_english / stronger_model with the
English prompt scored for the translation route), the fold-wise per-strategy cost
and risk calibration, and the cost-quality evaluation at equal accuracy.

## Verify on Windows PowerShell

```powershell
cd <path-to>\Ai-tax
git fetch origin
git switch arena/ad3092b4-ai-tax
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python models\fixtures\make_fixtures.py          # rewrites identical fixture CSVs
python -m src.train cost --seed 42 --n-splits 5  # T-005
python -m src.train risk --seed 42 --n-splits 5  # T-007 + SHAP figures
python -m src.router --seed 42 --n-splits 5      # T-008 + curve figure
python -m ruff format --check .
python -m ruff check .
python -m pytest tests\arena -q                  # 42 tests
python -m pytest -q                              # 46 tests, same as make check
python -m streamlit run app\main.py              # offline demo, T-006
```

`make check` and `make dashboard` also work where GNU Make is installed; the
Makefile still points at `app/dashboard.py`, which is now a shim that runs
`app/main.py`. Regenerating the fixtures must leave the CSV files byte-identical
(`git status` stays clean) — that is the determinism check.

## What is real, what is fixture, what is missing

Real: the code (`src/features.py`, `src/train.py`, `src/router.py`, `app/`), the
46 passing tests, the grouped-split and leakage guarantees, the artifact and
manifest formats, the SHAP and curve figures as rendered objects, and the
reproducibility of every command above.

Fixture (labelled in file names, plots, manifests and here — never a result):
`models/fixtures/{prompts,tokens,labels,routes}_fixture.csv`,
`models/cost_*_fixture.{json,csv,joblib}`, `models/risk_*_fixture.{json,csv,joblib}`,
`models/shap_risk_*_fixture.{csv,png}`, `models/router_*_fixture.{json,csv,joblib}`,
`models/figs/*_fixture.png`, and every number in the tables above.

Missing (upstream, codex-owned): `data/prompts.csv` (T-002), `data/tokens.csv`
(T-001), `data/responses.csv` (T-003), `data/labels.csv` (T-004) and real router
observations plus model prices for T-008. `src/tokens.py`, `src/collect.py`,
`src/judge.py`, `src/stats.py` and `notebooks/` are untouched placeholders. No API
call was made and no money was spent.

## Handoff for Astra

1. Review order: `git diff e47a220...arena/ad3092b4-ai-tax --stat`, then
   `git log --oneline e47a220..HEAD`: first T-005 version, T-005 rework for this
   brief (`c121aa3`), T-006 plus its lint fix, T-007 (`b323912`), T-008.
   Run `make check PYTHON=.venv/bin/python` (or the PowerShell block above).
2. Decide the branch question: four stacked branches were requested, the platform
   allowed one. Rename or split on your side; content matches the four cards.
3. Resolve two card contradictions: `tasks/T-005.md` forbids `fertility` as a model
   input while the run brief lists it as a feature — I kept it out of the models and
   report it as a diagnostic, because `token_count = fertility × word_count` is an
   exact leak. `tasks/T-007.md` names `src/risk.py` and `tests/test_risk_split.py`,
   the run brief limits me to `src/train.py` and `tests/arena/`; risk training lives
   in `src/train.py` (`train_risk`) and tests in `tests/arena/test_risk.py`.
4. Makefile and requirements (Astra-only): add `cost`, `risk`, `router` targets,
   repoint `dashboard` from `app/dashboard.py` to `app/main.py`, and add
   `matplotlib` to `requirements.txt` — today it is only a shap extra, so figures
   are skipped without it. Optionally pin a ruff config (`line-length = 88`,
   `select = ["E", "F", "W", "I", "UP", "B", "SIM", "PLW", "ARG", "RET", "ISC"]`)
   so `make check` catches what `tests/arena/test_lint.py` guards.
5. `data/demo/` is delegated to ARENA by the T-006 card but excluded by the run
   permissions, so the fixtures live in `models/fixtures/`. The app already reads
   `data/*.csv` → `data/demo/*.csv` → `models/fixtures/*_fixture.csv` in that order
   and drops the banner only for real data; copy or move the fixtures when you want
   them under `data/demo/`.
6. When T-001…T-004 land: rerun the three commands, delete every `*_fixture*`
   artifact, and re-read the tables — the fixture results must not survive into the
   poster or the pitch. Real runs need the same seed and produce the same file names
   without the `_fixture` label (`src.features.artifact_name`).
7. Next ARENA work: real-data rerun, prices per 1k tokens for a money-denominated
   cost-quality curve, and native-speaker review of the Kazakh prompts before any
   number is published.
