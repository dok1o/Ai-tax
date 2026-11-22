"""Token-cost predictor training and model comparison (T-005).

Reproducible command (seed 42, five folds grouped by ``q_id``)::

    python -m src.train --seed 42 --n-splits 5

Linear regression and LightGBM are fitted on the identical design matrix and
the identical GroupKFold splits, one pair per tokenizer, and every reported
number is out-of-fold.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from src.common.config import MODELS_DIR, PROMPTS_PATH, SEED, TOKENS_PATH
from src.features import (
    META_FEATURES,
    TEXT_FEATURES,
    design_matrix,
    fold_frame,
    load_dataset,
)

COST_DIR = MODELS_DIR / "token_cost"
MODEL_NAMES: tuple[str, ...] = ("linear", "lightgbm")
DEFAULT_SPLITS = 5
RUN_COMMAND = "python -m src.train --seed 42 --n-splits 5"


def make_models(seed: int = SEED) -> dict[str, Any]:
    """Return the two compared regressors with fixed seeds."""
    return {
        "linear": LinearRegression(),
        "lightgbm": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=2,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbose=-1,
        ),
    }


def assign_folds(groups: Sequence[str], n_splits: int, seed: int = SEED) -> np.ndarray:
    """Assign one fold per row so every ``q_id`` stays on a single side."""
    unique_groups = pd.unique(pd.Series(list(groups), dtype="string"))
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"need at least {n_splits} distinct q_id groups, found {len(unique_groups)}"
        )
    order = np.random.default_rng(seed).permutation(len(groups))
    grouped = np.asarray(groups, dtype=object)[order]
    folds = np.zeros(len(groups), dtype=int)
    for fold, (_, test) in enumerate(
        GroupKFold(n_splits=n_splits).split(np.zeros((len(order), 1)), groups=grouped)
    ):
        folds[order[test]] = fold
    return folds


def cross_validate_cost(
    dataset: pd.DataFrame, n_splits: int = DEFAULT_SPLITS, seed: int = SEED
) -> dict[str, pd.DataFrame]:
    """Return out-of-fold predictions, metrics and fold assignments."""
    data = dataset.reset_index(drop=True)
    folds = assign_folds(list(data["q_id"]), n_splits=n_splits, seed=seed)
    blocks: list[pd.DataFrame] = []
    for tokenizer, block in data.groupby("tokenizer", sort=True):
        positions = block.index.to_numpy()
        block = block.reset_index(drop=True)
        block_folds = folds[positions]
        matrix = design_matrix(block)
        target = block["token_count"].to_numpy(dtype=float)
        for name in MODEL_NAMES:
            prediction = np.empty(len(block))
            for fold in range(n_splits):
                train = block_folds != fold
                test = block_folds == fold
                if not train.any() or not test.any():
                    raise ValueError(f"fold {fold} is degenerate for {tokenizer}")
                model = make_models(seed)[name]
                model.fit(matrix[train], target[train])
                prediction[test] = model.predict(matrix[test])
            blocks.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "tokenizer": tokenizer,
                        "prompt_id": block["prompt_id"],
                        "q_id": block["q_id"],
                        "lang": block["lang"],
                        "fold": block_folds,
                        "token_count": block["token_count"].astype(int),
                        "prediction": prediction,
                    }
                )
            )
    oof = pd.concat(blocks, ignore_index=True)
    oof["absolute_error"] = (oof["prediction"] - oof["token_count"]).abs()
    return {
        "oof": oof,
        "metrics": summarise_oof(oof),
        "folds": fold_frame(data, folds),
    }


def summarise_oof(oof: pd.DataFrame) -> pd.DataFrame:
    """Aggregate out-of-fold MAE and n by model, language and tokenizer."""

    def aggregate(keys: Sequence[str], **scope: str) -> pd.DataFrame:
        grouped = (
            oof.groupby(["model", *keys], sort=True)["absolute_error"]
            .agg(n="size", mae="mean")
            .reset_index()
        )
        for column, value in scope.items():
            grouped[column] = value
        return grouped

    metrics = pd.concat(
        [
            aggregate(["tokenizer", "lang"]),
            aggregate(["tokenizer"], lang="all"),
            aggregate(["lang"], tokenizer="all"),
            aggregate([], tokenizer="all", lang="all"),
        ],
        ignore_index=True,
    )
    metrics["mae"] = metrics["mae"].round(4)
    return (
        metrics[["model", "tokenizer", "lang", "n", "mae"]]
        .sort_values(["model", "tokenizer", "lang"])
        .reset_index(drop=True)
    )


def slug(value: str) -> str:
    """Make a tokenizer identifier safe for a filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))[:60]


def refit_artifacts(
    dataset: pd.DataFrame, seed: int = SEED
) -> dict[str, dict[str, Any]]:
    """Refit both models on every row of a tokenizer for offline reuse."""
    artifacts: dict[str, dict[str, Any]] = {}
    for tokenizer, block in dataset.groupby("tokenizer", sort=True):
        matrix = design_matrix(block.reset_index(drop=True))
        target = block["token_count"].to_numpy(dtype=float)
        for name in MODEL_NAMES:
            model = make_models(seed)[name]
            model.fit(matrix, target)
            artifacts[f"{name}__{slug(tokenizer)}"] = {
                "task": "token_cost",
                "model_name": name,
                "tokenizer": tokenizer,
                "seed": seed,
                "columns": list(matrix.columns),
                "text_features": list(TEXT_FEATURES),
                "meta_features": list(META_FEATURES),
                "n_rows": len(block),
                "estimator": model,
            }
    return artifacts


def save_outputs(
    results: dict[str, pd.DataFrame],
    dataset: pd.DataFrame,
    out_dir: Path = COST_DIR,
    command: str = RUN_COMMAND,
    n_splits: int = DEFAULT_SPLITS,
    seed: int = SEED,
    prompts_path: Path = PROMPTS_PATH,
    tokens_path: Path = TOKENS_PATH,
) -> dict[str, Path]:
    """Write metrics, predictions, folds, refit models and the run manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "metrics": out_dir / "metrics.csv",
        "oof_predictions": out_dir / "oof_predictions.csv",
        "folds": out_dir / "folds.csv",
        "run": out_dir / "run.json",
    }
    results["metrics"].to_csv(written["metrics"], index=False)
    results["oof"].to_csv(written["oof_predictions"], index=False)
    results["folds"].to_csv(written["folds"], index=False)
    artifacts = refit_artifacts(dataset, seed=seed)
    for key, payload in artifacts.items():
        joblib.dump(payload, out_dir / f"cost_{key}.joblib")
    manifest = {
        "task": "token_cost",
        "command": command,
        "seed": seed,
        "n_splits": n_splits,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "group_by": "q_id",
        "inputs": {
            "prompts": str(prompts_path),
            "tokens": str(tokens_path),
            "n_prompts": int(dataset["prompt_id"].nunique()),
            "n_questions": int(dataset["q_id"].nunique()),
            "tokenizers": sorted(set(dataset["tokenizer"])),
        },
        "models": list(MODEL_NAMES),
        "artifacts": sorted(f"cost_{key}.joblib" for key in artifacts),
        "versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }
    written["run"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return written


def train_token_cost(
    prompts_path: Path = PROMPTS_PATH,
    tokens_path: Path = TOKENS_PATH,
    out_dir: Path = COST_DIR,
    n_splits: int = DEFAULT_SPLITS,
    seed: int = SEED,
    command: str = RUN_COMMAND,
) -> pd.DataFrame:
    """Run the full T-005 pipeline and return the metrics table."""
    dataset = load_dataset(prompts_path, tokens_path)
    results = cross_validate_cost(dataset, n_splits=n_splits, seed=seed)
    save_outputs(
        results,
        dataset,
        out_dir=out_dir,
        command=command,
        n_splits=n_splits,
        seed=seed,
        prompts_path=prompts_path,
        tokens_path=tokens_path,
    )
    return results["metrics"]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the cost-training command line."""
    parser = argparse.ArgumentParser(
        description="Train the T-005 token-cost predictor."
    )
    parser.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    parser.add_argument("--tokens", type=Path, default=TOKENS_PATH)
    parser.add_argument("--out", type=Path, default=COST_DIR)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns 2 while the upstream CSV inputs are still missing."""
    args = parse_args(argv)
    command = " ".join([sys.argv[0], *(sys.argv[1:] if argv is None else argv)])
    for path in (args.prompts, args.tokens):
        if not path.exists():
            print(f"missing input {path}; waiting on T-001 and T-002", file=sys.stderr)
            return 2
    metrics = train_token_cost(
        prompts_path=args.prompts,
        tokens_path=args.tokens,
        out_dir=args.out,
        n_splits=args.n_splits,
        seed=args.seed,
        command=command,
    )
    overall = metrics[(metrics["lang"] == "all") & (metrics["tokenizer"] == "all")]
    print(overall.to_string(index=False))
    print(f"wrote {len(metrics)} metric rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
