"""Language-aware router and cost-quality evaluation (T-008).

Strategies: ``direct``, ``via_english`` and ``stronger_model``. The router takes
the cheapest strategy whose predicted risk of a wrong answer stays under a
threshold. Standard parts: GroupKFold, logistic regression, LightGBM, bootstrap
percentile intervals. Ours: the route formulation, the per-strategy calibration
of T-005/T-007 predictions and the cost-quality evaluation.

Reproducible command::

    python -m src.router --seed 42 --n-splits 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.common.config import MODELS_DIR, SEED
from src.features import (
    DEFAULT_SPLITS,
    artifact_name,
    assign_folds,
    read_contract_csv,
    require_columns,
    resolve_artifact,
    resolve_input,
)

FIGS_DIR = MODELS_DIR / "figs"
BOOTSTRAP_RESAMPLES = 1000
DIRECT = "direct"
VIA_ENGLISH = "via_english"
STRONGER = "stronger_model"
STRATEGIES: tuple[str, ...] = (DIRECT, VIA_ENGLISH, STRONGER)
BASELINES: tuple[str, ...] = STRATEGIES
ROUTE_COLUMNS: tuple[str, ...] = (
    "q_id",
    "prompt_id",
    "lang",
    "strategy",
    "model",
    "total_tokens",
    "is_correct",
)
ROUTER_COMMAND = "python -m src.router --seed 42 --n-splits 5"
KEY = ["q_id", "lang"]


def decode_correctness(values: pd.Series) -> pd.DataFrame:
    """Decode contract booleans or 0/1 flags into graded 0/1 outcome columns.

    ``true`` and ``1`` mean the answer was correct; ``false`` and ``0`` mean it
    was not. Any other spelling is rejected instead of being read as false.
    """
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "0", "1"}).all():
        bad = sorted(set(normalized) - {"true", "false", "0", "1"})
        raise ValueError(f"routes has non-boolean is_correct values: {bad}")
    incorrect = normalized.isin({"false", "0"})
    return pd.DataFrame(
        {"is_correct": (~incorrect).astype(int), "is_incorrect": incorrect.astype(int)},
        index=values.index,
    )


def prepare_routes(
    routes: pd.DataFrame,
    prompts: pd.DataFrame,
    cost_oof: pd.DataFrame,
    risk_oof: pd.DataFrame,
    cost_model: str,
    tokenizer: str,
    risk_model: str,
) -> pd.DataFrame:
    """Attach out-of-fold cost and risk predictions to every route observation.

    ``via_english`` is scored with the English prompt of the same question, the
    only prompt the router would actually send. Nothing is imputed: a route
    without both predictions is rejected.
    """
    require_columns(routes, ROUTE_COLUMNS, "routes")
    data = routes[list(ROUTE_COLUMNS)].copy()
    data["total_tokens"] = pd.to_numeric(data["total_tokens"])
    data[["is_correct", "is_incorrect"]] = decode_correctness(data["is_correct"])
    unknown = sorted(set(data["strategy"]) - set(STRATEGIES))
    if unknown:
        raise ValueError(f"routes has unknown strategies: {unknown}")
    english = prompts.loc[prompts["lang"] == "en", ["q_id", "prompt_id"]].rename(
        columns={"prompt_id": "en_prompt_id"}
    )
    data = data.merge(english, on="q_id", how="left")
    if data["en_prompt_id"].isna().any():
        missing = sorted(set(data.loc[data["en_prompt_id"].isna(), "q_id"]))
        raise ValueError(f"no English prompt for q_id: {missing}")
    data["risk_prompt_id"] = np.where(
        data["strategy"] == VIA_ENGLISH, data["en_prompt_id"], data["prompt_id"]
    )
    costs = cost_oof.loc[
        (cost_oof["model"] == cost_model) & (cost_oof["tokenizer"] == tokenizer),
        ["prompt_id", "prediction"],
    ].rename(columns={"prediction": "prompt_tokens_hat"})
    if costs.empty:
        raise ValueError(
            f"no out-of-fold cost predictions for {cost_model}/{tokenizer}"
        )
    data = data.merge(costs, on="prompt_id", how="left")
    risks = risk_oof.loc[
        risk_oof["model"] == risk_model, ["prompt_id", "answer_model", "risk"]
    ]
    if risks.empty:
        raise ValueError(f"no out-of-fold risk predictions for model {risk_model}")
    data = data.merge(
        risks,
        left_on=["risk_prompt_id", "model"],
        right_on=["prompt_id", "answer_model"],
        how="left",
    ).drop(columns=["prompt_id_y", "answer_model"], errors="ignore")
    data = (
        data.rename(columns={"prompt_id_x": "prompt_id"})
        if "prompt_id_x" in data
        else data
    )
    for column in ("prompt_tokens_hat", "risk"):
        if data[column].isna().any():
            raise ValueError(
                f"{int(data[column].isna().sum())} route rows have no {column}; "
                "run T-005 and T-007 first, never impute measured outcomes"
            )
    data = data.rename(columns={"risk": "risk_raw"})
    return data.drop(
        columns=["en_prompt_id", "risk_prompt_id", "is_incorrect"]
    ).reset_index(drop=True)


def calibrate(routes: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    """Estimate per-strategy cost and risk corrections on the other folds only.

    ``cost_hat`` scales the predicted prompt tokens of a route to the tokens the
    route actually bills; ``risk_hat`` shifts the raw risk to the realized
    incorrect rate of that strategy and language. Both use training folds only.
    """
    data = routes.copy()
    data["risk_hat"] = np.nan
    data["cost_hat"] = np.nan
    for fold in sorted(set(folds)):
        train = data[folds != fold]
        test = folds == fold
        stats = route_statistics(train)
        block = data.loc[
            test, ["strategy", "lang", "prompt_tokens_hat", "risk_raw"]
        ].merge(stats, on=["strategy", "lang"], how="left")
        block["cost_multiplier"] = block["cost_multiplier"].fillna(
            block["strategy"].map(fallback_multiplier(train)).astype(float)
        )
        block["risk_offset"] = block["risk_offset"].fillna(
            block["strategy"].map(fallback_offset(train)).astype(float)
        )
        data.loc[test, "cost_hat"] = (
            block["prompt_tokens_hat"].to_numpy() * block["cost_multiplier"].to_numpy()
        )
        data.loc[test, "risk_hat"] = np.clip(
            block["risk_raw"].to_numpy() + block["risk_offset"].to_numpy(), 0.0, 1.0
        )
    if data[["cost_hat", "risk_hat"]].isna().any().any():
        raise ValueError("calibration left unrouted rows")
    return data


def route_statistics(routes: pd.DataFrame) -> pd.DataFrame:
    """Mean cost multiplier and risk offset per strategy and language."""
    grouped = routes.assign(
        ratio=routes["total_tokens"] / routes["prompt_tokens_hat"].clip(lower=1e-9),
        gap=(1 - routes["is_correct"]) - routes["risk_raw"],
    ).groupby(["strategy", "lang"], sort=True)
    return grouped.agg(
        cost_multiplier=("ratio", "mean"), risk_offset=("gap", "mean")
    ).reset_index()


def fallback_multiplier(routes: pd.DataFrame) -> dict[str, float]:
    """Strategy-level multiplier used when a language has no training rows."""
    ratio = routes["total_tokens"] / routes["prompt_tokens_hat"].clip(lower=1e-9)
    return routes.assign(ratio=ratio).groupby("strategy")["ratio"].mean().to_dict()


def fallback_offset(routes: pd.DataFrame) -> dict[str, float]:
    """Strategy-level risk offset used when a language has no training rows."""
    gap = (1 - routes["is_correct"]) - routes["risk_raw"]
    return routes.assign(gap=gap).groupby("strategy")["gap"].mean().to_dict()


def eligible_keys(routes: pd.DataFrame) -> pd.DataFrame:
    """Question-language keys where every strategy was observed."""
    counts = routes.groupby(KEY)["strategy"].nunique()
    keep = counts[counts == len(STRATEGIES)].reset_index()[KEY]
    return routes.merge(keep, on=KEY)[["q_id", "lang"]].drop_duplicates()


def choose_strategy(options: pd.DataFrame, threshold: float) -> str:
    """Cheapest strategy under the risk threshold, else the safest one."""
    ordered = options.sort_values("strategy")
    feasible = ordered[ordered["risk_hat"] <= threshold]
    if feasible.empty:
        return str(ordered.loc[ordered["risk_hat"].idxmin(), "strategy"])
    return str(feasible.loc[feasible["cost_hat"].idxmin(), "strategy"])


def decide(routes: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One routing decision per question and language at a fixed threshold.

    The cheapest feasible strategy wins; when nothing is feasible the safest one
    does. Ties break on the strategy name, so a decision is deterministic.
    """
    feasible = routes["risk_hat"] <= threshold
    ordered = routes.assign(
        infeasible=(~feasible).astype(int),
        rank=np.where(feasible, routes["cost_hat"], routes["risk_hat"]),
    ).sort_values([*KEY, "infeasible", "rank", "strategy"], kind="mergesort")
    chosen = ordered.groupby(KEY, sort=True).head(1).reset_index(drop=True)
    return pd.DataFrame(
        {
            "q_id": chosen["q_id"],
            "lang": chosen["lang"],
            "threshold": threshold,
            "strategy": chosen["strategy"],
            "cost_hat": chosen["cost_hat"].astype(float),
            "risk_hat": chosen["risk_hat"].astype(float),
            "total_tokens": chosen["total_tokens"].astype(float),
            "is_correct": chosen["is_correct"].astype(int),
        }
    )


def baseline_decisions(routes: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Always use one strategy; only keys where it was observed."""
    block = routes[routes["strategy"] == strategy].copy()
    if block.empty:
        raise ValueError(f"strategy {strategy} is missing from the data")
    return (
        pd.DataFrame(
            {
                "q_id": block["q_id"],
                "lang": block["lang"],
                "threshold": float("nan"),
                "strategy": strategy,
                "cost_hat": block["cost_hat"].astype(float),
                "risk_hat": block["risk_hat"].astype(float),
                "total_tokens": block["total_tokens"].astype(float),
                "is_correct": block["is_correct"].astype(int),
            }
        )
        .sort_values(KEY)
        .reset_index(drop=True)
    )


def outcome_summary(decisions: pd.DataFrame, label: str) -> dict[str, Any]:
    """Cost, accuracy and n of one routing policy on held-out questions."""
    return {
        "policy": label,
        "n": len(decisions),
        "cost_tokens_mean": float(decisions["total_tokens"].mean())
        if len(decisions)
        else float("nan"),
        "cost_hat_mean": float(decisions["cost_hat"].mean())
        if len(decisions)
        else float("nan"),
        "accuracy": float(decisions["is_correct"].mean())
        if len(decisions)
        else float("nan"),
        "expected_accuracy": float(1.0 - decisions["risk_hat"].mean())
        if len(decisions)
        else float("nan"),
    }


def sweep_thresholds(
    routes: pd.DataFrame, thresholds: Sequence[float]
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    """Cost-quality curve over thresholds plus the decisions behind each point."""
    rows = []
    decisions: dict[float, pd.DataFrame] = {}
    for threshold in thresholds:
        table = decide(routes, float(threshold))
        decisions[float(threshold)] = table
        summary = outcome_summary(table, f"router@{threshold:.2f}")
        shares = table["strategy"].value_counts(normalize=True)
        summary.update(
            {
                "threshold": float(threshold),
                "share_direct": float(shares.get(DIRECT, 0.0)),
                "share_via_english": float(shares.get(VIA_ENGLISH, 0.0)),
                "share_stronger_model": float(shares.get(STRONGER, 0.0)),
            }
        )
        rows.append(summary)
    return pd.DataFrame(rows), decisions


def align_outcomes(router: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Pair router and baseline outcomes on identical question-language keys."""
    merged = router.merge(
        baseline, on=KEY, suffixes=("_router", "_baseline"), how="inner"
    )
    if merged.empty:
        raise ValueError("router and baseline share no question-language keys")
    return merged


def bootstrap_saving(
    router: pd.DataFrame,
    baseline: pd.DataFrame,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
) -> dict[str, Any]:
    """Tokens saved at the baseline's accuracy, with a bootstrap 95% interval.

    Resampling unit is the question id, the same unit the grouped splits use, so
    the three translations of a question are always resampled together.
    """
    merged = align_outcomes(router, baseline)
    saving = merged["total_tokens_baseline"] - merged["total_tokens_router"]
    base_cost = merged["total_tokens_baseline"]
    questions = merged["q_id"].to_numpy()
    rng = np.random.default_rng(seed)
    unique = pd.unique(pd.Series(questions))
    index_by_question = {
        question: np.flatnonzero(questions == question) for question in unique
    }
    draws = np.empty(n_resamples)
    draw_percents = np.empty(n_resamples)
    for sample in range(n_resamples):
        picked = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([index_by_question[question] for question in picked])
        draws[sample] = saving.to_numpy()[positions].mean()
        draw_percents[sample] = (
            100.0 * draws[sample] / base_cost.to_numpy()[positions].mean()
        )
    low, high = np.percentile(draws, [2.5, 97.5])
    percent_low, percent_high = np.percentile(draw_percents, [2.5, 97.5])
    return {
        "baseline": str(baseline["strategy"].iloc[0]),
        "n_keys": len(merged),
        "n_questions": len(unique),
        "resamples": n_resamples,
        "seed": seed,
        "accuracy_router": float(merged["is_correct_router"].mean()),
        "accuracy_baseline": float(merged["is_correct_baseline"].mean()),
        "cost_router": float(merged["total_tokens_router"].mean()),
        "cost_baseline": float(base_cost.mean()),
        "saving_tokens": float(saving.mean()),
        "saving_tokens_ci95": [float(low), float(high)],
        "saving_percent": float(100.0 * saving.mean() / base_cost.mean()),
        "saving_percent_ci95": [float(percent_low), float(percent_high)],
    }


def pick_threshold(curve: pd.DataFrame, baseline_accuracy: float) -> float:
    """Largest threshold whose accuracy on the given rows is not below always-direct."""
    usable = curve[curve["accuracy"] >= baseline_accuracy]
    if usable.empty:
        return float(curve.loc[curve["accuracy"].idxmax(), "threshold"])
    return float(usable["threshold"].max())


def select_threshold(routes: pd.DataFrame, thresholds: Sequence[float]) -> float:
    """Choose the operating threshold from the given rows only.

    Called with the training folds of one outer split, never with the rows the
    chosen threshold is then evaluated on.
    """
    curve, _ = sweep_thresholds(routes, thresholds)
    direct = outcome_summary(baseline_decisions(routes, DIRECT), f"always_{DIRECT}")
    return pick_threshold(curve, float(direct["accuracy"]))


def nested_decisions(
    routes: pd.DataFrame, folds: np.ndarray, thresholds: Sequence[float]
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Decide every held-out fold with the threshold picked on the other folds."""
    blocks: list[pd.DataFrame] = []
    per_fold: dict[int, float] = {}
    for fold in sorted({int(value) for value in folds}):
        train, test = folds != fold, folds == fold
        if not train.any() or not test.any():
            raise ValueError(f"fold {fold} is degenerate for threshold selection")
        threshold = select_threshold(routes[train], thresholds)
        per_fold[fold] = threshold
        block = decide(routes[test], threshold)
        block["fold"] = fold
        blocks.append(block)
    decisions = pd.concat(blocks, ignore_index=True)
    return decisions.sort_values(KEY).reset_index(drop=True), per_fold


def threshold_grid(step: float = 0.05) -> list[float]:
    """Thresholds swept for the cost-quality curve."""
    return [round(value, 2) for value in np.arange(step, 1.0 - step / 2, step)]


def save_curve_figure(
    curve: pd.DataFrame, baselines: pd.DataFrame, path: Path, is_fixture: bool
) -> Path | None:
    """Plot realized cost against realized accuracy for the sweep and baselines.

    Returns None when matplotlib is not installed: it is an optional extra of
    shap and is missing from a plain ``pip install -r requirements.txt``.
    """
    try:
        import matplotlib
    except ImportError:
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.plot(
        curve["accuracy"], curve["cost_tokens_mean"], marker="o", label="router sweep"
    )
    for _, row in baselines.iterrows():
        axis.scatter(
            row["accuracy"], row["cost_tokens_mean"], label=row["policy"], s=45
        )
    axis.set_xlabel("held-out accuracy")
    axis.set_ylabel("mean tokens per question")
    title = "Router cost-quality curve" + (
        " (FIXTURE data, not a result)" if is_fixture else ""
    )
    axis.set_title(title, fontsize=10)
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def best_risk_model(manifest: dict[str, Any] | None) -> str:
    """Pick the risk model with the best pooled held-out ROC-AUC."""
    if not manifest or not manifest.get("results"):
        raise ValueError("run T-007 first: models/risk_metrics*.json is missing")
    results = pd.DataFrame(manifest["results"])
    overall = results[results["lang"] == "all"].dropna(subset=["roc_auc_pooled"])
    if overall.empty:
        raise ValueError("no risk model has a usable pooled ROC-AUC")
    return str(overall.loc[overall["roc_auc_pooled"].idxmax(), "model"])


def best_cost_model(manifest: dict[str, Any] | None) -> tuple[str, str]:
    """Pick the cost model and tokenizer with the lowest held-out MAE."""
    if not manifest or not manifest.get("results"):
        raise ValueError("run T-005 first: models/cost_metrics*.json is missing")
    results = pd.DataFrame(manifest["results"])
    overall = results[results["lang"] == "all"].dropna(subset=["mae_mean"])
    learned = overall[overall["model"] != "lang_mean"]
    if learned.empty:
        raise ValueError("no learned cost model in the manifest")
    best = learned.loc[learned["mae_mean"].idxmin()]
    return str(best["model"]), str(best["tokenizer"])


def train_router(
    source_dir: Path = MODELS_DIR,
    out_dir: Path = MODELS_DIR,
    figs_dir: Path = FIGS_DIR,
    n_splits: int = DEFAULT_SPLITS,
    seed: int = SEED,
    command: str = ROUTER_COMMAND,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Evaluate the router on held-out questions and persist every artifact."""
    routes_path, routes_fixture = resolve_input("routes")
    prompts_path, prompts_fixture = resolve_input("prompts")
    is_fixture = routes_fixture or prompts_fixture
    routes = pd.read_csv(routes_path, dtype=str, keep_default_na=False)
    prompts = read_contract_csv(prompts_path)
    cost_manifest = json.loads(
        resolve_artifact(source_dir, "cost_metrics", "json")[0].read_text("utf-8")
    )
    risk_manifest = json.loads(
        resolve_artifact(source_dir, "risk_metrics", "json")[0].read_text("utf-8")
    )
    cost_oof = pd.read_csv(resolve_artifact(source_dir, "cost_oof", "csv")[0])
    risk_oof = pd.read_csv(resolve_artifact(source_dir, "risk_oof", "csv")[0])
    cost_model, tokenizer = best_cost_model(cost_manifest)
    risk_model = best_risk_model(risk_manifest)
    data = prepare_routes(
        routes, prompts, cost_oof, risk_oof, cost_model, tokenizer, risk_model
    )
    data = data[
        data.set_index(KEY).index.isin(eligible_keys(data).set_index(KEY).index)
    ].reset_index(drop=True)
    folds = assign_folds(list(data["q_id"]), n_splits, seed)
    data = calibrate(data, folds)
    data["fold"] = folds
    thresholds = threshold_grid()
    curve, _ = sweep_thresholds(data, thresholds)
    chosen, thresholds_per_fold = nested_decisions(data, folds, thresholds)
    threshold = select_threshold(data, thresholds)
    baselines = pd.DataFrame(
        [
            outcome_summary(baseline_decisions(data, strategy), f"always_{strategy}")
            for strategy in BASELINES
        ]
    )
    savings = {
        f"always_{strategy}": bootstrap_saving(
            chosen,
            baseline_decisions(data, strategy),
            n_resamples=n_resamples,
            seed=seed,
        )
        for strategy in BASELINES
    }
    figure = save_curve_figure(
        curve,
        baselines,
        figs_dir / artifact_name("router_curve", "png", is_fixture),
        is_fixture,
    )
    curve.round(4).to_csv(
        out_dir / artifact_name("router_curve", "csv", is_fixture), index=False
    )
    chosen.round(4).to_csv(
        out_dir / artifact_name("router_decisions", "csv", is_fixture), index=False
    )
    statistics = route_statistics(data)
    joblib.dump(
        {
            "task": "router",
            "fixture": is_fixture,
            "seed": seed,
            "threshold": threshold,
            "thresholds": thresholds,
            "strategies": list(STRATEGIES),
            "strategy_models": data.groupby("strategy")["model"]
            .agg(lambda values: values.mode().iloc[0])
            .to_dict(),
            "cost_model": cost_model,
            "tokenizer": tokenizer,
            "risk_model": risk_model,
            "statistics": statistics.to_dict("records"),
            "fallback_multiplier": fallback_multiplier(data),
            "fallback_offset": fallback_offset(data),
        },
        out_dir / artifact_name("router_policy", "joblib", is_fixture),
    )
    manifest = {
        "task": "router",
        "command": command,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "n_splits": n_splits,
        "group_by": "q_id",
        "fixture": is_fixture,
        "inputs": {
            "routes": str(routes_path),
            "prompts": str(prompts_path),
            "n_questions": int(data["q_id"].nunique()),
            "n_keys": int(len(data) // len(STRATEGIES)),
            "n_route_rows": len(data),
            "cost_model": cost_model,
            "tokenizer": tokenizer,
            "risk_model": risk_model,
        },
        "deployment_threshold": threshold,
        "thresholds_per_fold": {
            str(fold): value for fold, value in thresholds_per_fold.items()
        },
        "selection_rule": (
            "nested by q_id: every held-out fold is routed with the largest swept "
            "threshold whose accuracy on the other folds is not below always_direct; "
            "deployment_threshold repeats that rule on all rows and is only "
            "used by the app"
        ),
        "curve": json.loads(curve.round(4).to_json(orient="records")),
        "baselines": json.loads(baselines.round(4).to_json(orient="records")),
        "savings": savings,
        "notes": [
            "the sweep table is descriptive; the policy uses per-fold thresholds",
            "figures are skipped when matplotlib is absent; numbers stay in CSV/JSON",
            "risk from T-007 out-of-fold predictions, cost from T-005 out-of-fold",
            "via_english is scored with the English prompt of the same question",
            "per-strategy multipliers and offsets are fitted on the other folds only",
            "evaluation uses realized tokens and correctness of the chosen route",
            "bootstrap resamples question ids, 1000 draws, percentile interval",
        ],
        "figure": None if figure is None else figure.name,
    }
    path = out_dir / artifact_name("router_metrics", "json", is_fixture)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "curve": curve,
        "baselines": baselines,
        "decisions": chosen,
        "manifest": manifest,
    }


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the router command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate the T-008 language-aware router."
    )
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    parser.add_argument("--figs", type=Path, default=FIGS_DIR)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the router evaluation and print the headline numbers."""
    args = parse_args(argv)
    command = " ".join([sys.argv[0], *(sys.argv[1:] if argv is None else argv)])
    try:
        results = train_router(
            MODELS_DIR,
            args.out,
            args.figs,
            args.n_splits,
            args.seed,
            command,
            args.resamples,
        )
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(results["baselines"].to_string(index=False))
    for policy, saving in results["manifest"]["savings"].items():
        low, high = saving["saving_percent_ci95"]
        print(
            f"router vs {policy}: saving {saving['saving_tokens']:.1f} tokens "
            f"({saving['saving_percent']:.1f}%, CI95 {low:.1f}..{high:.1f}), "
            f"accuracy {saving['accuracy_router']:.3f} vs "
            f"{saving['accuracy_baseline']:.3f}, n={saving['n_keys']}"
        )
    manifest = results["manifest"]
    per_fold = sorted(set(manifest["thresholds_per_fold"].values()))
    print(f"deployment threshold {manifest['deployment_threshold']}, folds {per_fold}")
    print(f"outputs in {args.out}")
    return 0


if __name__ == "__main__":
    # Delegate to the importable module so pickled helpers reference src.router.
    import src.router

    raise SystemExit(src.router.main())
