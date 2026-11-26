# ARENA batch report: T-005 → T-008, rerun on the real data

Every number in the "Key metrics" section is now produced by `data/*.csv` (contract
v2.1), not by the committed fixtures. The fixture artifacts still exist, are still
labelled `_fixture`, and are used by the test suite only.

## Preflight

Python 3.11.2 in `.venv`, packages from `requirements.txt`: `pandas 3.1.1`,
`numpy 2.4.6`, `scikit-learn 1.9.1`, `lightgbm 4.7.0`, `shap 0.51.0`,
`streamlit 1.67.0`, `plotly 7.1.0`, `ruff 0.17.6`. Offline: no network, no
tokenizer cache, no API call, no money spent. `python run.py check` passes: ruff
clean on `.`, **227 passed**, prompts validator "60 questions / 180 rows; verified
share 0.000". `tests/arena` alone is **131 passed**. Every number below was produced
by an executed command and is reproducible with the same seed.

## Task table

| Task | Branch | Tests passed/failed | State | Data used |
| --- | --- | --- | --- | --- |
| T-005 cost predictor | `arena/ad3092b4-ai-tax` | 9 / 0 (`test_cost_split.py`) | review | **real** |
| T-006 app skeleton | `arena/ad3092b4-ai-tax` | 22 / 0 (`test_app.py`) | done (T-011 in review) | real + fixture (example mode) |
| T-007 error risk + SHAP | `arena/ad3092b4-ai-tax` | 9 / 0 (`test_risk.py`) | review | **real** |
| T-008 router | `arena/ad3092b4-ai-tax` | 14 / 0 (`test_router.py`) + 1 lint guard | review | **real**, derived route legs |
| Real-data rerun | `arena/ad3092b4-ai-tax` | 12 / 0 (`test_real_run.py`) | review | real |
| Whole repo | — | 227 / 0 (`python run.py check`) | review | mixed, labelled |

Branch note: the platform pins this session to `arena/ad3092b4-ai-tax`, cut from
`e47a220` of `dev` and merged with `origin/dev` (`f018e40`) as `0d4bfe9`. It refuses
to create the four task branches, so the tasks stay four commits on one pushed
branch. The rerun used exactly the same code path as the fixture runs — only the
resolved inputs changed.

## Inputs that were actually read

| File | Rows | What it holds |
| --- | --- | --- |
| `data/prompts.csv` | 180 | 60 questions × 3 languages (kk, ru, en), 6 categories × 10, `verified_by_native` = 0 everywhere |
| `data/tokens.csv` | 360 | the same 180 prompts × 2 tokenizers: `Qwen/Qwen2.5-0.5B@5796c1d2…` and `cl100k_base` |
| `data/responses.csv` | 1080 | 180 prompts × 3 local models (`gemma2:2b`, `llama3.2:3b`, `qwen2.5:3b`) × 2 reps, `usage_available` true on every row |
| `data/labels.csv` | 1080 | 958 graded + 122 unparsed; 668 correct, 412 incorrect |
| `data/metrics.csv` | 27 | external accuracy table (18 accuracy + 9 `mcnemar_exact`), used only as a cross-check |
| `data/routes.csv` | **absent** | no observed route legs are published, so the legs are derived (below) |

Native-verified share is **0.000**: no Kazakh reviewer has signed off on any prompt,
so every grade in this report is a programmatic one. That is stated on the page too.

## What the real data forced into the code

| Finding | Root cause | Fix | Regression test (`tests/arena/test_real_run.py`) |
| --- | --- | --- | --- |
| Risk training died with `LightGBMError: Do not support special JSON characters in feature name` | real model ids carry a colon, and one-hot names embed them (`model=gemma2:2b`) | `design_matrix` translates JSON-unsafe characters (`:` `[` `]` `{` `}` `"` `,` → `_`) and refuses two levels that collapse onto one name | `test_one_hot_levels_are_sanitized_for_lightgbm` (fits a real booster), `test_two_levels_that_collapse_onto_one_name_are_refused` |
| The router silently billed fixture legs | `data/routes.csv` does not exist and `prepare_routes` refuses to impute, so it fell back to `models/fixtures/routes_fixture.csv` | route-derivation layer in `src/features.py`: `load_observations`, `route_roles`, `build_route_rows`, `derive_routes`, `ensure_routes`; `src/router.py` records `inputs.routes_source` (`observed` / `derived` / `fixture`) and appends the derived-leg notes to the manifest | `test_observations_are_graded_strictly_and_never_imputed`, `test_roles_are_read_from_cost_and_accuracy`, `test_route_legs_use_the_observed_english_prompt_for_via_english`, `test_a_missing_observation_is_an_error_not_a_gap` |
| Router baselines printed n = 360 instead of 180 | real labels carry rep 0 and 1, so `risk_oof` has two identical rows per `(prompt_id, answer_model)` and the merge doubled every leg | `prepare_routes` drops repeated risk rows per pair and raises if two reps disagree | `test_repeated_observations_do_not_double_the_routes` |
| Fixture tests broke once real artifacts landed in `models/` | `resolve_artifact` and `app.logic.candidate` prefer real files, so fixture routes were joined to real `cost_oof` | `LT_FORCE_EXAMPLE=1` reverses that preference in both resolvers, `real_routes_available()` no longer counts a stale `routes_derived.csv` on its own, and `tests/arena/conftest.py` sets the flag at import time | `test_routes_stay_on_the_fixture_without_the_real_inputs`, `test_the_state_resolves_every_input_as_real`, `test_the_page_shows_the_real_badge_and_the_real_panels` |
| A test wrote 18 rows into the committed `models/routes_derived.csv` | `derive_routes(out_path: Path = DERIVED_ROUTES_PATH)` bound the destination at import time, so the file written and the path returned could disagree | the destination is resolved when the call happens | `test_routes_are_derived_when_the_real_observations_exist`, `test_the_committed_real_artifacts_are_labelled_real` |

## Key metrics (REAL data)

### T-005 token cost

Out-of-fold over 5 GroupKFold folds by `q_id`, seed 42, target `token_count`,
n = 180 prompts per tokenizer (`models/cost_metrics.json`, `cost_oof.csv` = 360 rows).
MAE and R² are fold means ± fold std.

| Model | Tokenizer | MAE mean ± std | R² mean ± std | n |
| --- | --- | --- | --- | --- |
| lang_mean (baseline) | Qwen2.5-0.5B | 19.3130 ± 3.2895 | 0.3257 ± 0.1069 | 180 |
| linear | Qwen2.5-0.5B | 6.6952 ± 1.0986 | 0.9139 ± 0.0339 | 180 |
| **lightgbm** | Qwen2.5-0.5B | **5.7563 ± 1.3304** | **0.9355 ± 0.0301** | 180 |
| lang_mean (baseline) | cl100k_base | 22.4894 ± 2.6722 | 0.3801 ± 0.0950 | 180 |
| linear | cl100k_base | 7.9933 ± 1.3588 | 0.9172 ± 0.0359 | 180 |
| **lightgbm** | cl100k_base | **6.6139 ± 1.2637** | **0.9381 ± 0.0319** | 180 |

Per language (n = 60 prompts each), lightgbm MAE ± std / R² ± std:

| Language | Qwen2.5-0.5B | cl100k_base |
| --- | --- | --- |
| en | 4.0447 ± 0.8009 / 0.8946 ± 0.0511 | 3.8392 ± 0.3597 / 0.9023 ± 0.0405 |
| kk | 8.1090 ± 2.7961 / 0.8918 ± 0.0667 | 9.5818 ± 2.8101 / 0.8894 ± 0.0738 |
| ru | 5.1151 ± 1.5638 / 0.9123 ± 0.0328 | 6.4206 ± 0.9709 / 0.8935 ± 0.0356 |

Honest answer to "does the model beat the per-language mean baseline?": **yes**, and
not by a generator artifact this time. LightGBM cuts pooled MAE 3.4× (19.31 → 5.76
tokens with the Qwen tokenizer) and wins in every language separately. The baseline
is weaker than it looks: its pooled R² of 0.3257 comes entirely from the
between-language gap, and within a language it is **negative** (Qwen: en −0.0561,
kk −0.0746, ru −0.1018; cl100k: en −0.0516, kk −0.0823, ru −0.0904), i.e. predicting
one language's mean is worse than predicting the global mean inside that language.
`linear` stays close to LightGBM (6.70 vs 5.76 pooled) and ties it on cl100k/ru
(6.4152 vs 6.4206), so the gradient boosting is worth ~15% MAE, not an order of
magnitude. Kazakh remains the hard case: ~8–10 tokens of absolute error on prompts
of ~93–109 tokens.

Language tax, diagnostic only — `fertility` and `token_count` are never features:

| Tokenizer | en | kk | ru |
| --- | --- | --- | --- |
| Qwen2.5-0.5B fertility (tokens/word) | 1.6140 | 3.8044 | 2.4521 |
| cl100k_base fertility | 1.5604 | 4.3985 | 2.9115 |
| Qwen2.5-0.5B mean `token_count` | 46.87 | 93.27 | 61.83 |
| cl100k_base mean `token_count` | 45.53 | 108.52 | 74.18 |
| Qwen2.5-0.5B mean `tokens_vs_en` | 1.0000 | 1.9920 | 1.3166 |
| cl100k_base mean `tokens_vs_en` | 1.0000 | 2.3760 | 1.6129 |

The same question costs ~2× the tokens in Kazakh as in English (2.38× under
cl100k_base), which is the input-side half of the tax.

### T-007 error risk

Target `is_incorrect = 1 − is_correct`, out-of-fold over 5 folds by `q_id`, seed 42,
n = 1080 observations (412 incorrect). Features are prompt text statistics, `lang`,
`category` and the answering model only — no response text, no grade, no post-call
usage (`models/risk_metrics.json`).

| Model | ROC-AUC pooled | ROC-AUC fold mean ± std | Brier pooled (± std) | Accuracy | n |
| --- | --- | --- | --- | --- | --- |
| train_rate (baseline) | 0.4513 | 0.5000 ± 0.0000 | 0.2369 (± 0.0115) | 0.6185 | 1080 |
| **logistic** | **0.7815** | **0.7907 ± 0.0326** | **0.1844 (± 0.0189)** | **0.7315** | 1080 |
| lightgbm | 0.7377 | 0.7458 ± 0.0490 | 0.2222 (± 0.0337) | 0.6833 | 1080 |

Per language (pooled AUC, n = 360 each): logistic en 0.6900 (69 incorrect),
kk 0.6052 (245), ru 0.6909 (98); lightgbm en 0.7098, kk 0.5748, ru 0.5714.

Does it beat its baseline? **Yes — logistic, on every metric** (AUC 0.7815 vs 0.4513,
Brier 0.1844 vs 0.2369, accuracy 0.7315 vs 0.6185). LightGBM also beats the baseline
but **loses to logistic** (0.7377 vs 0.7815 pooled AUC, 0.2222 vs 0.1844 Brier), so
the deployed risk model is the logistic one; the trees add nothing on 1080 rows with
23 features. The caveat: within-language discrimination is weak (kk 0.605), so most
of the pooled AUC is the between-language gap — the model largely learns "Kazakh is
riskier", which the SHAP values confirm.

SHAP mean |value| over the 1080 held-out rows, 23 features
(`models/shap_risk_logistic.csv`, `models/figs/shap_risk_logistic.png`):
`lang=kk` 0.4873, `lang=ru` 0.3684, `category=world_geography_history` 0.3007,
`model=llama3.2_3b` 0.2303, `category=kazakhstan_history_culture` 0.2265,
`cyrillic_share` 0.2238. LightGBM
(`models/shap_risk_lightgbm.csv`, `…/shap_risk_lightgbm.png`): `kk_share` 1.1799,
`unique_char_ratio` 0.4553, `model=llama3.2_3b` 0.3465, `punct_share` 0.3297,
`model=qwen2.5_3b` 0.3272, `category=kazakhstan_history_culture` 0.2786. The
Kazakh-letter share is the single strongest tree feature, and the language dummies
are the strongest linear ones — same story, two encodings.

Observed answering models behind those labels (both reps, n = 360 per model):

| Model | Mean observed tokens | Strict accuracy | en | kk | ru |
| --- | --- | --- | --- | --- | --- |
| gemma2:2b | 80.61 | 0.6278 | 0.7833 | 0.3250 | 0.7750 |
| llama3.2:3b | 103.20 | 0.5278 | 0.8083 | 0.1833 | 0.5917 |
| qwen2.5:3b | 102.64 | 0.7000 | 0.8333 | 0.4500 | 0.8167 |

Cross-check against the external table in `data/metrics.csv` (n = 120 per cell,
strict accuracy with its 95% CI, then `accuracy_parsed` with the graded n):
gemma2:2b kk 0.3250 [0.2331, 0.4169] / 0.4286 (91), ru 0.7750 [0.6831, 0.8669] /
0.7815 (119), en 0.7833 [0.6831, 0.8750] / 0.8034 (117); llama3.2:3b kk 0.1833
[0.1083, 0.2583] / 0.3929 (56), ru 0.5917 [0.4917, 0.6917] / 0.6121 (116), en 0.8083
[0.7083, 0.9000] / 0.8220 (118); qwen2.5:3b kk 0.4500 [0.3331, 0.5667] / 0.5047
(107), ru 0.8167 [0.7250, 0.9000] / 0.8235 (119), en 0.8333 [0.7333, 0.9085] /
0.8696 (115). These match the recomputation exactly, and the confidence intervals
overlap heavily between models within a language: with 60 questions, no model
ranking inside one language is significant. McNemar (exact, Holm-adjusted) says the
gap is between languages, not between models: kk vs ru and kk vs en are significant
for all three models (p from 9.3e-10 to 7.6e-5), while ru vs en is not (gemma
p = 1.000, qwen p = 0.453, llama p = 0.031).

Parsing, stated plainly: 106 of the 360 Kazakh answers never parsed (29%), against 10
in English and 6 in Russian. Unparsed answers count as **wrong** in `accuracy_strict`
and are excluded from `accuracy_parsed`, so part of the Kazakh deficit is an output
format failure, not a knowledge failure — gemma2:2b kk moves 0.3250 → 0.4286. Both
numbers are reported; neither is a clean capability estimate.

### T-008 router

No observed route legs were published, so they are derived from the real
observations: `models/routes_derived.csv`, 540 rows = 60 questions × 3 languages ×
3 strategies, `source=derived`, one observation per key (rep 0). Roles come from the
data: `direct` is the cheapest observed model (**gemma2:2b**, 80.61 tokens mean),
`stronger_model` is the most accurate one that is not `direct` (**qwen2.5:3b**,
0.7000), and `via_english` is the cheapest model answering the English prompt of the
same question. `llama3.2:3b` is never a leg — it costs more than gemma2:2b (103.20
tokens) and is less accurate than qwen2.5:3b (0.5278).

| Strategy | Language | Mean tokens | Strict accuracy | n |
| --- | --- | --- | --- | --- |
| direct | en | 65.07 | 0.7667 | 60 |
| direct | kk | 108.70 | 0.3500 | 60 |
| direct | ru | 69.97 | 0.8000 | 60 |
| via_english | en / kk / ru | 65.07 | 0.7667 | 180 |
| stronger_model | en | 80.83 | 0.8333 | 60 |
| stronger_model | kk | 132.15 | 0.4500 | 60 |
| stronger_model | ru | 93.88 | 0.8500 | 60 |

`translation_tokens` is **0** on every `via_english` leg because no translation call
was ever made: the cost is the provider's own `input_tokens + output_tokens` for the
English prompt, and the `via_english` cost is therefore a **lower bound**. If a real
translation call costs t tokens, every `via_english` leg grows by t and the savings
below shrink. Nothing is padded with an invented overhead.

Held-out evaluation, 5 GroupKFold folds by `q_id`, n = 180 question-language keys
(60 questions), realized tokens and realized correctness of the chosen route, cost
from T-005 (lightgbm, Qwen tokenizer) and risk from T-007 (logistic), both
out-of-fold, with per-strategy multipliers and offsets fitted on the other folds
only (`models/router_metrics.json`, `router_decisions.csv`):

| Policy | Mean tokens | Predicted cost | Accuracy | n |
| --- | --- | --- | --- | --- |
| always_direct | 81.2444 | 84.9597 | 0.6389 | 180 |
| always_via_english | 65.0667 | 67.5740 | 0.7667 | 180 |
| always_stronger_model | 102.2889 | 109.3876 | 0.7111 | 180 |
| **router @ threshold 0.95** | **65.0667** | 67.5740 | **0.7667** | 180 |

Cost saved at the baseline accuracy, bootstrap percentile CI over 1000 resamples of
`q_id`, seed 42:

| Baseline | Tokens saved | Tokens CI95 | % saved (CI95) | Router accuracy | Baseline accuracy |
| --- | --- | --- | --- | --- | --- |
| always_direct | +16.18 | 13.97 … 18.51 | +19.91% (17.48 … 22.45) | 0.7667 | 0.6389 |
| always_via_english | **0.00** | **0.00 … 0.00** | **0.00% (0.00 … 0.00)** | 0.7667 | 0.7667 |
| always_stronger_model | +37.22 | 34.65 … 39.77 | +36.39% (33.83 … 38.98) | 0.7667 | 0.7111 |

**Plain statement of the null result: the router does not beat `always_via_english`.**
It saves 0 tokens and gains 0 accuracy against it, and the interval is [0, 0] because
the two policies pick the same route for all 180 keys — `models/router_decisions.csv`
routes every English key to `direct` (60) and every Kazakh and Russian key to
`via_english` (60 + 60). The deployed policy *is* "answer in English unless the
question is already English". It does beat `always_direct` (+19.91% tokens at
+0.1278 accuracy) and `always_stronger_model` (+36.39% at +0.0556), but that gain is
the language tax being routed around, not a routing decision finer than a language
rule. Any poster claim has to be phrased that way.

Threshold selection stays nested: every held-out fold is routed with the largest
swept threshold whose accuracy on the other folds is not below `always_direct`, and
`deployment_threshold` is the mode of those per-fold choices, so it is never
re-selected on the rows it is judged on. All five folds chose **0.95**
(`thresholds_per_fold` = {0: 0.95, …, 4: 0.95}), the same as on the fixture data.
The sweep itself (19 thresholds, `models/router_curve.csv`,
`models/figs/router_curve.png`) spans 65.07 → 76.63 tokens and 0.7667 → 0.8000
accuracy: at threshold 0.05 the router sends 52% of keys to `stronger_model` and buys
+0.0333 accuracy for +11.57 tokens. The rule is "cheapest policy that does not lose
accuracy to `always_direct`", so it deliberately leaves that accuracy on the table.
Turning it into a cost-per-accuracy trade-off needs prices, which were not supplied —
all costs here are in tokens.

### Standard vs ours

Standard: GroupKFold, linear and logistic regression, LightGBM, MAE, R², ROC-AUC,
Brier score, calibration curves, SHAP, bootstrap percentile intervals, McNemar exact
(cross-check only), Streamlit. Ours: the trilingual pre-tokenization feature set
(including the Kazakh-letter share), the per-language mean and training-rate
baselines, the route formulation (`direct` / `via_english` / `stronger_model` with
the English prompt scored for the translation route and keyed to the native
question), the fold-wise per-strategy cost and risk calibration, the nested
threshold rule, and the cost-quality evaluation at equal accuracy.

## The app on the real data

With the real files present, `app/logic.load_state()` resolves `prompts`, `tokens`,
`labels`, `cost_metrics`, `risk_metrics` and `router_metrics` as **real** and
`routes` as **derived**, `AppState.is_example` is `False`, the sidebar badge reads
**"Real data"**, and no example banner is shown
(`test_the_page_shows_the_real_badge_and_the_real_panels`). The provenance table on
the page keeps naming `routes: derived`, so the derived legs stay visible rather than
passing as observed ones.

`LT_FORCE_EXAMPLE=1` reverses both resolvers: the same page then reads **"Example
data"** and serves `models/fixtures/*_fixture.csv` plus the `*_fixture` artifacts
even though the real ones sit next to them. That is the mode the whole test suite
runs in (`tests/arena/conftest.py` sets the flag at import time, before any fixture
is built), so a real rerun can never silently overturn a fixture assertion — and a
fixture assertion can never be mistaken for a result.

## Verify on Windows PowerShell

```powershell
cd <path-to>\Ai-tax
git fetch origin
git switch arena/ad3092b4-ai-tax
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.train all --out models --figs models\figs   # T-005 + T-007 + SHAP
python -m src.router                                      # T-008 + curve figure
python -m ruff format --check .
python -m ruff check .
python -m pytest tests\arena -q                           # 131 tests
python run.py check                                       # ruff + 227 tests + validator
python -m streamlit run app\main.py                       # real data, "Real data" badge
$env:LT_FORCE_EXAMPLE = "1"; python -m streamlit run app\main.py   # example mode
```

`python models\fixtures\make_fixtures.py` rewrites the fixture CSVs byte-identically
(`git status` stays clean) — that is the determinism check for the example mode.
Regenerating the real artifacts needs no network and no API key: everything comes
from `data/*.csv`. Re-running `src.router` reproduces the tables above exactly,
including the [0, 0] interval against `always_via_english`.

## What is real, what is fixture, what is missing

Real: `data/prompts.csv`, `data/tokens.csv`, `data/tokens_ratio.csv`,
`data/responses.csv`, `data/labels.csv`, `data/label_status.csv`, `data/metrics.csv`;
the unlabelled artifacts in `models/` (`cost_metrics.json`, `cost_oof.csv`,
`risk_metrics.json`, `risk_oof.csv`, `shap_risk_*.csv/png`, `router_metrics.json`,
`router_decisions.csv`, `router_curve.csv`, the `*.joblib` models,
`models/figs/*.png` without `_fixture`), the derived legs in
`models/routes_derived.csv`, and every number in this report.

Fixture (labelled in file names, plots, manifests and here — never a result, used by
the test suite only): `models/fixtures/{prompts,tokens,labels,routes}_fixture.csv`,
`models/*_fixture.{json,csv,joblib}`, `models/shap_risk_*_fixture.{csv,png}`,
`models/figs/*_fixture.png`.

Missing, and owed upstream: (1) **observed route legs** — `data/routes.csv` with the
translation call actually made, so `translation_tokens` stops being 0 and
`via_english` stops being a lower bound (RELEASE item 4); (2) **model prices per 1k
tokens**, so the cost-quality curve can be denominated in money instead of tokens and
the selection rule can trade cost against accuracy; (3) **native verification** of the
Kazakh prompts — the verified share is 0.000, so no Kazakh number here is
reviewer-signed; (4) a stronger model that is actually good at Kazakh, because
qwen2.5:3b at 0.4500 kk strict accuracy cannot rescue the direct route; (5) output
formatting for Kazakh answers, since 29% of them never parsed. `src/tokens.py`,
`src/collect.py`, `src/judge.py`, `src/stats.py` and `notebooks/` stay untouched
(codex-owned). No API call was made and no money was spent.

## Handoff for Astra

1. Review order: `git log --oneline e47a220..HEAD`, then the merge `0d4bfe9` and the
   rerun commit. `git diff 0d4bfe9...HEAD --stat` shows only ARENA paths:
   `src/features.py`, `src/router.py`, `app/logic.py`, `tests/arena/conftest.py`,
   `tests/arena/test_real_run.py`, `models/`. Run `python run.py check`.
2. Decide what to do with the **null result**. The router ties
   `always_via_english` exactly. Options: (a) publish it as the finding — with these
   three models, "route Kazakh and Russian through English" is the whole policy, and
   the learned risk threshold adds nothing; (b) change the selection rule to buy
   accuracy (threshold 0.05 gives 0.8000 accuracy for 76.63 tokens) and re-run once
   prices exist; (c) wait for a stronger Kazakh-capable model, because at 0.45 kk
   accuracy the `stronger_model` leg is dominated. My recommendation is (a) now and
   (b) when prices land.
3. Unblock the two upstream inputs: observed route legs including the translation
   call, and prices per 1k tokens. Both are named in `RELEASE.md` item 4 and both
   change the numbers here — the derived legs are a lower bound on `via_english`, so
   today's +19.91% saving against `always_direct` is an upper bound.
4. Keep the example mode as the test switch: `tests/arena/conftest.py` sets
   `LT_FORCE_EXAMPLE=1` at import, which is what keeps 131 ARENA tests on the
   committed fixtures while `models/` holds real artifacts. Do not "clean up" that
   fixture; without it the fixture suite silently evaluates real cost predictions.
5. Card contradictions already resolved and still standing: `fertility` stays a
   diagnostic, never a feature (`token_count = fertility × word_count` is an exact
   leak); risk training lives in `src/train.py::train_risk`, not `src/risk.py`;
   `data/demo/` is not written to, the fixtures live in `models/fixtures/`.
6. Makefile and requirements (Astra-only): add `cost`, `risk`, `router` targets that
   write into `models/` with `--figs models/figs`, and repoint `dashboard` at
   `app/main.py`.
7. Next ARENA work once the inputs arrive: rerun with observed legs and prices, add a
   money-denominated cost-quality curve, and re-check the Kazakh numbers after native
   review.
