"""Cost (T-005) and error-risk (T-007) model training with grouped splits.

Reproducible commands, run from the repository root::

    python -m src.train cost --seed 42 --n-splits 5
    python -m src.train risk --seed 42 --n-splits 5
    python -m src.train all  --seed 42 --n-splits 5

Standard parts: GroupKFold, scikit-learn linear and logistic regression,
LightGBM, ROC-AUC, Brier score, SHAP. Ours: the trilingual pre-tokenization
features, the per-language mean baseline and the fixture labelling rule.
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
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, r2_score, roc_auc_score

from src.common.config import MODELS_DIR, SEED
from src.features import (
    DEFAULT_SPLITS,
    TARGET,
    artifact_name,
    assign_folds,
    assign_folds_frame,
    design_matrix,
    fertility_table,
    load_cost_dataset,
    load_labels,
    resolve_input,
)

FIGS_DIR = MODELS_DIR / "figs"
COST_MODELS: tuple[str, ...] = ("lang_mean", "linear", "lightgbm")
RISK_MODELS: tuple[str, ...] = ("train_rate", "logistic", "lightgbm")
RISK_META: tuple[str, ...] = ("lang", "category", "model")
TARGET_RISK = "is_incorrect"
COST_COMMAND = "python -m src.train cost --seed 42 --n-splits 5"
RISK_COMMAND = "python -m src.train risk --seed 42 --n-splits 5"


class LanguageMeanBaseline:
    """Predict the training mean token count of each language (T-005 baseline)."""

    def __init__(self) -> None:
        self.lang_columns_: list[str] = []
        self.means_: dict[int, float] = {}
        self.fallback_: float = 0.0

    def fit(self, matrix: pd.DataFrame, target: np.ndarray) -> LanguageMeanBaseline:
        self.lang_columns_ = [
            column for column in matrix.columns if column.startswith("lang=")
        ]
        slots = matrix[self.lang_columns_].to_numpy().argmax(axis=1)
        self.fallback_ = float(np.mean(target))
        self.means_ = {
            slot: float(np.mean(target[slots == slot]))
            for slot in range(len(self.lang_columns_))
            if (slots == slot).any()
        }
        return self

    def predict(self, matrix: pd.DataFrame) -> np.ndarray:
        slots = matrix[self.lang_columns_].to_numpy().argmax(axis=1)
        return np.array([self.means_.get(slot, self.fallback_) for slot in slots])


class TrainRateBaseline:
    """Predict the training incorrect rate for every row (T-007 baseline)."""

    def __init__(self) -> None:
        self.rate_: float = 0.5

    def fit(self, _matrix: pd.DataFrame, target: np.ndarray) -> TrainRateBaseline:
        self.rate_ = float(np.mean(target)) if len(target) else 0.5
        return self

    def predict_proba(self, matrix: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(matrix), self.rate_)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, matrix: pd.DataFrame) -> np.ndarray:
        return np.full(len(matrix), int(self.rate_ >= 0.5), dtype=int)


def cost_estimator(name: str, seed: int = SEED) -> Any:
    """Return one unfitted token-cost model."""
    if name == "lang_mean":
        return LanguageMeanBaseline()
    if name == "linear":
        return LinearRegression()
    if name == "lightgbm":
        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=3,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbose=-1,
        )
    raise ValueError(f"unknown cost model: {name}")


def risk_estimator(name: str, seed: int = SEED) -> Any:
    """Return one unfitted error-risk model."""
    if name == "train_rate":
        return TrainRateBaseline()
    if name == "logistic":
        return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=3,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbose=-1,
        )
    raise ValueError(f"unknown risk model: {name}")


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation over the finite fold scores."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std(ddof=1)) if finite.size > 1 else 0.0


def regression_fold_scores(oof: pd.DataFrame) -> pd.DataFrame:
    """MAE and R2 per fold, pooled over languages and per language."""
    rows: list[dict[str, Any]] = []
    for keys in (
        ("model", "tokenizer", "fold"),
        ("model", "tokenizer", "lang", "fold"),
    ):
        for key, block in oof.groupby(list(keys), sort=True):
            model, tokenizer, fold = key[0], key[1], key[-1]
            lang = "all" if len(keys) == 3 else key[2]
            truth = block[TARGET].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model,
                    "tokenizer": tokenizer,
                    "lang": lang,
                    "fold": int(fold),
                    "n": len(block),
                    "mae": float(block["absolute_error"].mean()),
                    "r2": r2_score(truth, block["prediction"].to_numpy())
                    if np.ptp(truth) > 0
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def summarise_regression(scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold scores into MAE and R2 mean and standard deviation."""
    summary = pd.DataFrame(
        [
            {
                "model": model,
                "tokenizer": tokenizer,
                "lang": lang,
                "n": int(block["n"].sum()),
                "n_folds": int(block["fold"].nunique()),
                "mae_mean": mean_std(block["mae"])[0],
                "mae_std": mean_std(block["mae"])[1],
                "r2_mean": mean_std(block["r2"])[0],
                "r2_std": mean_std(block["r2"])[1],
            }
            for (model, tokenizer, lang), block in scores.groupby(
                ["model", "tokenizer", "lang"], sort=True
            )
        ]
    )
    numeric = ["mae_mean", "mae_std", "r2_mean", "r2_std"]
    return (
        summary.sort_values(["tokenizer", "lang", "model"])
        .reset_index(drop=True)
        .round({column: 4 for column in numeric})
    )


def classification_fold_scores(oof: pd.DataFrame) -> pd.DataFrame:
    """ROC-AUC, Brier score and accuracy per fold, pooled and per language."""
    rows: list[dict[str, Any]] = []
    for keys in (("model", "fold"), ("model", "lang", "fold")):
        for key, block in oof.groupby(list(keys), sort=True):
            model, fold = key[0], key[-1]
            lang = "all" if len(keys) == 2 else key[1]
            truth = block[TARGET_RISK].to_numpy(dtype=int)
            risk = block["risk"].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model,
                    "lang": lang,
                    "fold": int(fold),
                    "n": len(block),
                    "n_incorrect": int(truth.sum()),
                    "roc_auc": roc_auc_score(truth, risk)
                    if 0 < truth.sum() < len(truth)
                    else float("nan"),
                    "brier": brier_score_loss(truth, risk),
                    "accuracy": accuracy_score(truth, (risk >= 0.5).astype(int)),
                }
            )
    return pd.DataFrame(rows)


def pooled_risk_metrics(block: pd.DataFrame) -> dict[str, Any]:
    """Held-out metrics over all rows of one scope, with n."""
    truth = block[TARGET_RISK].to_numpy(dtype=int)
    risk = block["risk"].to_numpy(dtype=float)
    return {
        "n": len(truth),
        "n_incorrect": int(truth.sum()),
        "roc_auc_pooled": roc_auc_score(truth, risk)
        if 0 < truth.sum() < len(truth)
        else float("nan"),
        "brier_pooled": brier_score_loss(truth, risk),
        "accuracy_pooled": accuracy_score(truth, (risk >= 0.5).astype(int)),
    }


def summarise_classification(scores: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Fold mean and standard deviation plus pooled held-out metrics per model."""
    pooled = {
        (model, "all"): pooled_risk_metrics(block)
        for model, block in oof.groupby("model", sort=True)
    }
    pooled.update(
        {
            (model, lang): pooled_risk_metrics(block)
            for (model, lang), block in oof.groupby(["model", "lang"], sort=True)
        }
    )
    rows = []
    for (model, lang), block in scores.groupby(["model", "lang"], sort=True):
        row = {
            "model": model,
            "lang": lang,
            "n": int(block["n"].sum()),
            "n_folds": int(block["fold"].nunique()),
            "roc_auc_mean": mean_std(block["roc_auc"])[0],
            "roc_auc_std": mean_std(block["roc_auc"])[1],
            "brier_mean": mean_std(block["brier"])[0],
            "brier_std": mean_std(block["brier"])[1],
        }
        row.update(pooled[(model, lang)])
        rows.append(row)
    summary = pd.DataFrame(rows)
    return (
        summary.sort_values(["lang", "model"])
        .reset_index(drop=True)
        .round({column: 4 for column in summary.select_dtypes("float").columns})
    )


def cross_validate_cost(
    dataset: pd.DataFrame, n_splits: int = DEFAULT_SPLITS, seed: int = SEED
) -> dict[str, pd.DataFrame]:
    """Out-of-fold cost predictions per tokenizer for every compared model."""
    data = dataset.reset_index(drop=True)
    folds = assign_folds(list(data["q_id"]), n_splits, seed)
    blocks: list[pd.DataFrame] = []
    for tokenizer, group in data.groupby("tokenizer", sort=True):
        positions = group.index.to_numpy()
        block = group.reset_index(drop=True)
        block_folds = folds[positions]
        matrix = design_matrix(block)
        target = block[TARGET].to_numpy(dtype=float)
        for name in COST_MODELS:
            prediction = np.empty(len(block))
            for fold in range(n_splits):
                train, test = block_folds != fold, block_folds == fold
                if not train.any() or not test.any():
                    raise ValueError(f"fold {fold} is degenerate for {tokenizer}")
                estimator = cost_estimator(name, seed)
                estimator.fit(matrix[train], target[train])
                prediction[test] = estimator.predict(matrix[test])
            blocks.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "tokenizer": tokenizer,
                        "prompt_id": block["prompt_id"],
                        "q_id": block["q_id"],
                        "lang": block["lang"],
                        "fold": block_folds,
                        TARGET: block[TARGET].astype(int),
                        "prediction": prediction,
                    }
                )
            )
    oof = pd.concat(blocks, ignore_index=True)
    oof["absolute_error"] = (oof["prediction"] - oof[TARGET]).abs()
    scores = regression_fold_scores(oof)
    return {
        "oof": oof,
        "scores": scores,
        "summary": summarise_regression(scores),
        "folds": assign_folds_frame(data, folds),
        "fertility": fertility_table(data),
    }


def cross_validate_risk(
    dataset: pd.DataFrame, n_splits: int = DEFAULT_SPLITS, seed: int = SEED
) -> dict[str, pd.DataFrame]:
    """Out-of-fold P(incorrect) predictions plus pooled held-out SHAP values."""
    data = dataset.reset_index(drop=True)
    if data[TARGET_RISK].nunique() < 2:
        raise ValueError("labels have a single class; risk cannot be learned")
    folds = assign_folds(list(data["q_id"]), n_splits, seed)
    matrix = design_matrix(data, meta_columns=RISK_META)
    target = data[TARGET_RISK].to_numpy(dtype=int)
    oof_blocks: list[pd.DataFrame] = []
    shap_blocks: dict[str, np.ndarray] = {
        name: np.full(matrix.shape, np.nan) for name in RISK_MODELS
    }
    for name in RISK_MODELS:
        risk = np.empty(len(data))
        for fold in range(n_splits):
            train, test = folds != fold, folds == fold
            if not train.any() or not test.any():
                raise ValueError(f"fold {fold} is degenerate for risk model {name}")
            if len(np.unique(target[train])) < 2:
                raise ValueError(f"fold {fold} training split has a single class")
            estimator = risk_estimator(name, seed)
            estimator.fit(matrix[train], target[train])
            risk[test] = estimator.predict_proba(matrix[test])[:, 1]
            if name != "train_rate":
                shap_blocks[name][
                    np.ix_(np.flatnonzero(test), np.arange(matrix.shape[1]))
                ] = positive_class_values(
                    shap_values_for(name, estimator, matrix[train], matrix[test])
                )
        oof_blocks.append(
            pd.DataFrame(
                {
                    "model": name,
                    "prompt_id": data["prompt_id"],
                    "q_id": data["q_id"],
                    "lang": data["lang"],
                    "category": data["category"],
                    "answer_model": data["model"],
                    "fold": folds,
                    TARGET_RISK: target,
                    "risk": risk,
                }
            )
        )
    oof = pd.concat(oof_blocks, ignore_index=True)
    oof["correct"] = (oof["risk"] < 0.5).astype(int)
    scores = classification_fold_scores(oof)
    explanations = {
        name: pd.DataFrame(values, columns=list(matrix.columns))
        for name, values in shap_blocks.items()
        if not np.isnan(values).all()
    }
    return {
        "oof": oof,
        "scores": scores,
        "summary": summarise_classification(scores, oof),
        "folds": assign_folds_frame(data, folds),
        "matrix": matrix,
        "explanations": explanations,
    }


def shap_values_for(
    name: str, estimator: Any, train: pd.DataFrame, test: pd.DataFrame
) -> Any:
    """SHAP values of one fitted fold model on its held-out rows."""
    import shap

    if name == "lightgbm":
        return shap.TreeExplainer(estimator).shap_values(test)
    return shap.LinearExplainer(estimator, train).shap_values(test)


def positive_class_values(values: Any) -> np.ndarray:
    """Reduce multi-class SHAP output to the positive class."""
    if isinstance(values, (list, tuple)):
        values = values[-1]
    array = np.asarray(values, dtype=float)
    return array[..., -1] if array.ndim == 3 else array


def calibration_frame(oof: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Quantile calibration curve per model from held-out predictions."""
    from sklearn.calibration import calibration_curve

    rows = []
    for name, block in oof.groupby("model", sort=True):
        truth = block[TARGET_RISK].to_numpy(dtype=int)
        if truth.sum() in (0, len(truth)):
            continue
        fraction, mean_value = calibration_curve(
            truth, block["risk"].to_numpy(), n_bins=n_bins, strategy="quantile"
        )
        rows.append(
            pd.DataFrame(
                {
                    "model": name,
                    "predicted_risk": mean_value,
                    "observed_incorrect": fraction,
                }
            ).assign(n=len(truth))
        )
    return pd.concat(rows, ignore_index=True).round(4)


def slug(value: str) -> str:
    """Make an identifier safe for a filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))[:48]


def save_shap_summary(
    values: np.ndarray, matrix: pd.DataFrame, path: Path, title: str
) -> Path | None:
    """Write a SHAP bar summary plot as PNG.

    Returns None when matplotlib is not installed: it is an optional extra of
    shap and is missing from a plain ``pip install -r requirements.txt``.
    """
    try:
        import matplotlib
    except ImportError:
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    explanation = shap.Explanation(
        values=values,
        data=matrix.to_numpy(dtype=float),
        feature_names=list(matrix.columns),
    )
    plt.figure()
    shap.plots.bar(explanation.abs.mean(0), max_display=15, show=False)
    plt.title(title, fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def versions() -> dict[str, str]:
    """Record package versions so a run can be reproduced."""
    import shap

    return {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "shap": shap.__version__,
    }


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize a table with NaN mapped to JSON null."""
    return json.loads(frame.to_json(orient="records"))


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a manifest with stable ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def train_cost(
    out_dir: Path = MODELS_DIR,
    n_splits: int = DEFAULT_SPLITS,
    seed: int = SEED,
    command: str = COST_COMMAND,
) -> dict[str, pd.DataFrame]:
    """Train, compare and persist the token-cost models (T-005)."""
    prompts_path, prompts_fixture = resolve_input("prompts")
    tokens_path, tokens_fixture = resolve_input("tokens")
    is_fixture = prompts_fixture or tokens_fixture
    dataset = load_cost_dataset(prompts_path, tokens_path)
    results = cross_validate_cost(dataset, n_splits=n_splits, seed=seed)
    written: list[str] = []
    for tokenizer, block in dataset.groupby("tokenizer", sort=True):
        matrix = design_matrix(block.reset_index(drop=True))
        for name in COST_MODELS:
            estimator = cost_estimator(name, seed)
            estimator.fit(matrix, block[TARGET].to_numpy(dtype=float))
            artifact = out_dir / artifact_name(
                f"cost_{name}__{slug(tokenizer)}", "joblib", is_fixture
            )
            joblib.dump(
                {
                    "task": "token_cost",
                    "model_name": name,
                    "tokenizer": tokenizer,
                    "seed": seed,
                    "fixture": is_fixture,
                    "columns": list(matrix.columns),
                    "n_rows": len(block),
                    "estimator": estimator,
                },
                artifact,
            )
            written.append(artifact.name)
    results["oof"].to_csv(
        out_dir / artifact_name("cost_oof", "csv", is_fixture), index=False
    )
    results["folds"].to_csv(
        out_dir / artifact_name("cost_folds", "csv", is_fixture), index=False
    )
    write_json(
        out_dir / artifact_name("cost_metrics", "json", is_fixture),
        {
            "task": "token_cost",
            "command": command,
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seed": seed,
            "n_splits": n_splits,
            "group_by": "q_id",
            "fixture": is_fixture,
            "inputs": {
                "prompts": str(prompts_path),
                "tokens": str(tokens_path),
                "n_questions": int(dataset["q_id"].nunique()),
                "n_prompts": int(dataset["prompt_id"].nunique()),
                "n_rows": len(dataset),
                "tokenizers": sorted(set(dataset["tokenizer"])),
            },
            "target": TARGET,
            "models": list(COST_MODELS),
            "results": json_records(results["summary"]),
            "fertility_diagnostic": json_records(results["fertility"]),
            "notes": [
                "fertility and token_count are targets or diagnostics, never features",
                "lang_mean is the per-language training mean baseline",
                "r2 is null for folds whose held-out target is constant",
            ],
            "artifacts": sorted(written),
            "versions": versions(),
        },
    )
    return results


def train_risk(
    out_dir: Path = MODELS_DIR,
    figs_dir: Path = FIGS_DIR,
    n_splits: int = DEFAULT_SPLITS,
    seed: int = SEED,
    command: str = RISK_COMMAND,
) -> dict[str, Any]:
    """Train, compare, explain and persist the error-risk models (T-007)."""
    prompts_path, prompts_fixture = resolve_input("prompts")
    labels_path, labels_fixture = resolve_input("labels")
    is_fixture = prompts_fixture or labels_fixture
    labels = load_labels(prompts_path, labels_path)
    labels[TARGET_RISK] = 1 - labels["is_correct"]
    results = cross_validate_risk(labels, n_splits=n_splits, seed=seed)
    results["calibration"] = calibration_frame(results["oof"])
    matrix = results["matrix"]
    written: list[str] = []
    missing_plots = False
    for name in RISK_MODELS:
        estimator = risk_estimator(name, seed)
        estimator.fit(matrix, labels[TARGET_RISK].to_numpy(dtype=int))
        artifact = out_dir / artifact_name(f"risk_{name}", "joblib", is_fixture)
        joblib.dump(
            {
                "task": "error_risk",
                "model_name": name,
                "seed": seed,
                "fixture": is_fixture,
                "columns": list(matrix.columns),
                "meta_columns": list(RISK_META),
                "n_rows": len(labels),
                "estimator": estimator,
            },
            artifact,
        )
        written.append(artifact.name)
    for name, explanation in results["explanations"].items():
        table = out_dir / artifact_name(f"shap_risk_{name}", "csv", is_fixture)
        explanation.assign(
            prompt_id=labels["prompt_id"].to_numpy(), q_id=labels["q_id"].to_numpy()
        ).to_csv(table, index=False)
        figure = save_shap_summary(
            explanation.to_numpy(),
            matrix,
            figs_dir / artifact_name(f"shap_risk_{name}", "png", is_fixture),
            f"SHAP mean |value|, error-risk {name}"
            + (" (FIXTURE data, not a result)" if is_fixture else ""),
        )
        written.append(table.name)
        if figure is None:
            missing_plots = True
        else:
            written.append(figure.name)
    results["oof"].to_csv(
        out_dir / artifact_name("risk_oof", "csv", is_fixture), index=False
    )
    results["calibration"].to_csv(
        out_dir / artifact_name("risk_calibration", "csv", is_fixture), index=False
    )
    notes = [
        "features are prompt text statistics, lang, category and answering model only",
        "no response text, grade or post-call usage is used as a feature",
        "roc_auc is null for folds whose held-out labels are single-class",
        "train_rate is the training incorrect-rate baseline",
        "SHAP values are held-out per fold; the train_rate baseline has none",
    ]
    if missing_plots:
        notes.append("figures skipped: matplotlib is not installed (shap plots extra)")
    write_json(
        out_dir / artifact_name("risk_metrics", "json", is_fixture),
        {
            "task": "error_risk",
            "command": command,
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seed": seed,
            "n_splits": n_splits,
            "group_by": "q_id",
            "fixture": is_fixture,
            "inputs": {
                "prompts": str(prompts_path),
                "labels": str(labels_path),
                "n_questions": int(labels["q_id"].nunique()),
                "n_rows": len(labels),
                "answer_models": sorted(set(labels["model"])),
            },
            "target": TARGET_RISK,
            "models": list(RISK_MODELS),
            "results": json_records(results["summary"]),
            "notes": notes,
            "artifacts": sorted(written),
            "versions": versions(),
        },
    )
    return results


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the training command line."""
    parser = argparse.ArgumentParser(
        description="Train the T-005 and T-007 predictors."
    )
    parser.add_argument(
        "task", choices=("cost", "risk", "all"), default="all", nargs="?"
    )
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    parser.add_argument("--figs", type=Path, default=FIGS_DIR)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested training task and print a short summary."""
    args = parse_args(argv)
    command = " ".join([sys.argv[0], *(sys.argv[1:] if argv is None else argv)])
    try:
        if args.task in ("cost", "all"):
            cost = train_cost(args.out, args.n_splits, args.seed, command)
            overall = cost["summary"][(cost["summary"]["lang"] == "all")]
            print(overall.to_string(index=False))
        if args.task in ("risk", "all"):
            risk = train_risk(args.out, args.figs, args.n_splits, args.seed, command)
            pooled = risk["summary"][risk["summary"]["lang"] == "all"]
            print(
                pooled.drop(columns=["roc_auc_mean", "roc_auc_std"]).to_string(
                    index=False
                )
            )
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"outputs in {args.out} and figures in {args.figs}")
    return 0


if __name__ == "__main__":
    # Delegate to the importable module: pickle records the defining module, so
    # baselines saved from a `python -m src.train` run must come from src.train.
    import src.train

    raise SystemExit(src.train.main())
