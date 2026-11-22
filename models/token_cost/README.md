# Token-cost model outputs (T-005)

Regenerate everything in this directory with one command from the repository root:

    python -m src.train --seed 42 --n-splits 5

Inputs are `data/prompts.csv` (T-002) and `data/tokens.csv` (T-001). Both are still
pending, so no results are stored here yet and the command exits with status 2 until
the two CSV files exist. Results are generated data: they are reproducible from the
command above and are not hand-edited.

| Output | Content |
| --- | --- |
| `metrics.csv` | Out-of-fold MAE and n per model, tokenizer and language. `all` in a column aggregates over that axis. |
| `oof_predictions.csv` | One row per prompt, tokenizer and model with fold, target, prediction and absolute error. |
| `folds.csv` | GroupKFold assignment per `q_id`. A question never spans folds. |
| `run.json` | Command, seed, fold count, input sizes, tokenizer list and package versions. |
| `cost_<model>__<tokenizer>.joblib` | Model refitted on all rows of that tokenizer, with the exact design-matrix column order. |

Compared models: `linear` (scikit-learn `LinearRegression`) and `lightgbm`
(`LGBMRegressor`, fixed seed). Both see the identical one-hot design matrix that
`src/features.py` builds from prompt text and metadata available before tokenization.
Measured `token_count` and `fertility` are rejected as inputs by
`src.features.assert_no_leakage`.
