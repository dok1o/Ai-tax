"""Router guarantees for T-008: hand-checkable cases, sweep monotonicity, outputs."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.common.config import MODELS_DIR, SEED
from src.features import (
    assign_folds,
    read_contract_csv,
    resolve_artifact,
    resolve_input,
)
from src.router import (
    BASELINES,
    DEFAULT_SPLITS,
    KEY,
    STRATEGIES,
    baseline_decisions,
    best_cost_model,
    best_risk_model,
    bootstrap_saving,
    calibrate,
    choose_strategy,
    decide,
    eligible_keys,
    outcome_summary,
    prepare_routes,
    sweep_thresholds,
    threshold_grid,
    train_router,
)

HAND_COSTS = {
    "q1": {"direct": 100.0, "via_english": 180.0, "stronger_model": 110.0},
    "q2": {"direct": 90.0, "via_english": 170.0, "stronger_model": 95.0},
}
HAND_CORRECT = {
    "q1": {"direct": 1, "via_english": 0, "stronger_model": 1},
    "q2": {"direct": 0, "via_english": 1, "stronger_model": 1},
}
HAND_RISK = {
    "q1": {"direct": 0.60, "via_english": 0.30, "stronger_model": 0.10},
    "q2": {"direct": 0.80, "via_english": 0.40, "stronger_model": 0.20},
}


def hand_routes() -> pd.DataFrame:
    """Six-row case whose baseline numbers are computable by hand."""
    rows = [
        {
            "q_id": q_id,
            "lang": "kk",
            "strategy": strategy,
            "model": "fixture-small-model",
            "total_tokens": HAND_COSTS[q_id][strategy],
            "is_correct": HAND_CORRECT[q_id][strategy],
            "cost_hat": HAND_COSTS[q_id][strategy],
            "risk_hat": HAND_RISK[q_id][strategy],
        }
        for q_id in ("q1", "q2")
        for strategy in STRATEGIES
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def prepared() -> pd.DataFrame:
    """Fixture routes with out-of-fold cost and risk predictions attached."""
    routes = pd.read_csv(resolve_input("routes")[0], dtype=str, keep_default_na=False)
    prompts = read_contract_csv(resolve_input("prompts")[0])
    cost_manifest = json.loads(
        resolve_artifact(MODELS_DIR, "cost_metrics", "json")[0].read_text(
            encoding="utf-8"
        )
    )
    risk_manifest = json.loads(
        resolve_artifact(MODELS_DIR, "risk_metrics", "json")[0].read_text(
            encoding="utf-8"
        )
    )
    cost_model, tokenizer = best_cost_model(cost_manifest)
    data = prepare_routes(
        routes,
        prompts,
        pd.read_csv(resolve_artifact(MODELS_DIR, "cost_oof", "csv")[0]),
        pd.read_csv(resolve_artifact(MODELS_DIR, "risk_oof", "csv")[0]),
        cost_model,
        tokenizer,
        best_risk_model(risk_manifest),
    )
    keys = eligible_keys(data)
    return data[data.set_index(KEY).index.isin(keys.set_index(KEY).index)].reset_index(
        drop=True
    )


@pytest.fixture(scope="session")
def calibrated(prepared: pd.DataFrame) -> pd.DataFrame:
    """Prepared routes after the fold-wise cost and risk calibration."""
    return calibrate(
        prepared, assign_folds(list(prepared["q_id"]), DEFAULT_SPLITS, SEED)
    )


def test_baselines_reproduce_hand_computed_numbers() -> None:
    routes = hand_routes()
    expected = {
        "direct": (95.0, 0.5),
        "via_english": (175.0, 0.5),
        "stronger_model": (102.5, 1.0),
    }
    for strategy in BASELINES:
        summary = outcome_summary(
            baseline_decisions(routes, strategy), f"always_{strategy}"
        )
        cost, accuracy = expected[strategy]
        assert summary["cost_tokens_mean"] == cost
        assert summary["accuracy"] == accuracy
        assert summary["n"] == 2


def test_choose_strategy_prefers_the_cheapest_feasible_route() -> None:
    routes = hand_routes()
    assert choose_strategy(routes[routes["q_id"] == "q1"], 0.65) == "direct"
    assert choose_strategy(routes[routes["q_id"] == "q2"], 0.65) == "stronger_model"
    assert choose_strategy(routes[routes["q_id"] == "q1"], 0.35) == "stronger_model"


def test_choose_strategy_falls_back_to_the_safest_route() -> None:
    routes = hand_routes()
    assert choose_strategy(routes[routes["q_id"] == "q1"], 0.05) == "stronger_model"
    assert choose_strategy(routes[routes["q_id"] == "q2"], 0.05) == "stronger_model"


def test_decisions_reproduce_hand_computed_threshold() -> None:
    decisions = decide(hand_routes(), 0.65)
    assert decisions["strategy"].tolist() == ["direct", "stronger_model"]
    summary = outcome_summary(decisions, "router@0.65")
    assert summary["cost_tokens_mean"] == 97.5
    assert summary["accuracy"] == 1.0


def test_router_never_picks_a_strategy_missing_from_the_data() -> None:
    routes = hand_routes()
    routes = routes[
        ~((routes["q_id"] == "q2") & (routes["strategy"] == "stronger_model"))
    ]
    for threshold in threshold_grid():
        decisions = decide(routes, threshold)
        available = routes.groupby(KEY)["strategy"].apply(set)
        for _, row in decisions.iterrows():
            assert row["strategy"] in available.loc[(row["q_id"], row["lang"])]


def test_bootstrap_saving_on_the_hand_case() -> None:
    routes = hand_routes()
    router = decide(routes, 0.65)
    saving = bootstrap_saving(
        router, baseline_decisions(routes, "direct"), n_resamples=200
    )
    assert saving["n_questions"] == 2
    assert saving["saving_tokens"] == -2.5
    assert saving["accuracy_router"] == 1.0
    assert saving["accuracy_baseline"] == 0.5
    low, high = saving["saving_tokens_ci95"]
    assert low <= -2.5 <= high
    assert low >= -5.0 and high <= 0.0


def test_sweep_is_monotone_on_the_fixture(calibrated: pd.DataFrame) -> None:
    curve, decisions = sweep_thresholds(calibrated, threshold_grid())
    assert list(curve["threshold"]) == threshold_grid()
    assert (curve["expected_accuracy"].diff().dropna() <= 1e-9).all()
    assert (curve["cost_hat_mean"].diff().dropna() <= 1e-9).all()
    assert (curve["n"] == len(calibrated) // len(STRATEGIES)).all()
    assert set(decisions) == set(threshold_grid())
    shares = curve[["share_direct", "share_via_english", "share_stronger_model"]].sum(
        axis=1
    )
    assert (shares.round(6) == 1.0).all()


def test_calibration_is_bounded(calibrated: pd.DataFrame) -> None:
    assert calibrated[["cost_hat", "risk_hat"]].notna().all().all()
    assert calibrated["risk_hat"].between(0.0, 1.0).all()
    assert (calibrated["cost_hat"] > 0).all()
    assert calibrated.groupby("q_id")["strategy"].nunique().max() == len(STRATEGIES)


def test_calibration_ignores_the_held_out_fold(prepared: pd.DataFrame) -> None:
    folds = assign_folds(list(prepared["q_id"]), DEFAULT_SPLITS, SEED)
    baseline = calibrate(prepared, folds)
    perturbed = prepared.copy()
    held_out = folds == 0
    perturbed.loc[held_out, "total_tokens"] = (
        perturbed.loc[held_out, "total_tokens"] * 5
    )
    perturbed.loc[held_out, "is_correct"] = 1 - perturbed.loc[held_out, "is_correct"]
    after = calibrate(perturbed, folds)
    # Estimates of fold 0 are fitted on the other folds, so corrupting fold 0
    # outcomes must not change what the router would predict for fold 0.
    for column in ("cost_hat", "risk_hat"):
        pd.testing.assert_series_equal(
            baseline.loc[held_out, column].reset_index(drop=True),
            after.loc[held_out, column].reset_index(drop=True),
        )


def test_prepare_routes_rejects_missing_predictions() -> None:
    routes = pd.read_csv(resolve_input("routes")[0], dtype=str, keep_default_na=False)
    prompts = read_contract_csv(resolve_input("prompts")[0])
    cost_oof = pd.read_csv(resolve_artifact(MODELS_DIR, "cost_oof", "csv")[0])
    risk_oof = pd.read_csv(resolve_artifact(MODELS_DIR, "risk_oof", "csv")[0])
    manifest = json.loads(
        resolve_artifact(MODELS_DIR, "cost_metrics", "json")[0].read_text(
            encoding="utf-8"
        )
    )
    cost_model, tokenizer = best_cost_model(manifest)
    for broken in (
        cost_oof[cost_oof["model"] != cost_model],
        cost_oof.assign(model="unknown"),
    ):
        with pytest.raises(ValueError, match="no out-of-fold cost predictions"):
            prepare_routes(
                routes, prompts, broken, risk_oof, cost_model, tokenizer, "logistic"
            )
    with pytest.raises(ValueError, match="no out-of-fold risk predictions"):
        prepare_routes(
            routes,
            prompts,
            cost_oof,
            risk_oof.assign(model="unknown"),
            cost_model,
            tokenizer,
            "logistic",
        )


def test_train_router_writes_labelled_outputs(tmp_path: Path) -> None:
    figs = tmp_path / "figs"
    results = train_router(
        source_dir=MODELS_DIR,
        out_dir=tmp_path,
        figs_dir=figs,
        n_splits=DEFAULT_SPLITS,
        seed=SEED,
        n_resamples=200,
    )
    manifest = json.loads(
        (tmp_path / "router_metrics_fixture.json").read_text(encoding="utf-8")
    )
    assert manifest["fixture"] is True
    assert manifest["seed"] == SEED
    assert manifest["group_by"] == "q_id"
    assert manifest["operating_threshold"] in threshold_grid()
    assert set(manifest["savings"]) == {f"always_{strategy}" for strategy in BASELINES}
    assert manifest["inputs"]["n_questions"] == 60
    assert manifest["command"].startswith("python -m src.router")
    for saving in manifest["savings"].values():
        assert saving["resamples"] == 200
        assert len(saving["saving_tokens_ci95"]) == 2
        assert saving["n_keys"] > 0
    assert (tmp_path / "router_curve_fixture.csv").exists()
    assert (tmp_path / "router_decisions_fixture.csv").exists()
    figure = figs / "router_curve_fixture.png"
    assert figure.exists() and figure.stat().st_size > 5000
    policy = joblib.load(tmp_path / "router_policy_fixture.joblib")
    assert policy["strategies"] == list(STRATEGIES)
    assert set(policy["strategy_models"]) == set(STRATEGIES)
    assert policy["threshold"] == manifest["operating_threshold"]
    assert len(results["curve"]) == len(threshold_grid())
