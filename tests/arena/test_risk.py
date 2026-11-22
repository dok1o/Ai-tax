"""Split, metric, calibration and SHAP guarantees for the T-007 risk predictor."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.common.config import SEED
from src.features import FIXTURE_DIR, load_labels
from src.train import (
    DEFAULT_SPLITS,
    RISK_MODELS,
    TARGET_RISK,
    assign_folds,
    calibration_frame,
    cross_validate_risk,
    train_risk,
)

PROMPTS_FIXTURE = FIXTURE_DIR / "prompts_fixture.csv"
LABELS_FIXTURE = FIXTURE_DIR / "labels_fixture.csv"
LEARNED = ("logistic", "lightgbm")


@pytest.fixture(scope="session")
def labels() -> pd.DataFrame:
    frame = load_labels(PROMPTS_FIXTURE, LABELS_FIXTURE)
    frame[TARGET_RISK] = 1 - frame["is_correct"]
    return frame


@pytest.fixture(scope="session")
def results(labels: pd.DataFrame) -> dict:
    return cross_validate_risk(labels, n_splits=DEFAULT_SPLITS, seed=SEED)


def test_no_q_id_in_both_train_and_test(labels: pd.DataFrame) -> None:
    folds = assign_folds(list(labels["q_id"]), DEFAULT_SPLITS, SEED)
    questions = labels["q_id"].to_numpy()
    for fold in range(DEFAULT_SPLITS):
        train = set(questions[folds != fold])
        test = set(questions[folds == fold])
        assert train and test and not train.intersection(test)
        assert train.union(test) == set(questions)


def test_risk_oof_keeps_questions_whole(results: dict) -> None:
    oof = results["oof"]
    assert set(oof["model"]) == set(RISK_MODELS)
    assert oof.groupby("q_id")["fold"].nunique().max() == 1
    risk = oof["risk"].to_numpy()
    assert np.isfinite(risk).all()
    assert risk.min() >= 0.0 and risk.max() <= 1.0
    per_model = oof.groupby("model").size()
    assert (per_model == len(oof) // len(RISK_MODELS)).all()


def test_metrics_report_auc_brier_and_n(labels: pd.DataFrame, results: dict) -> None:
    summary = results["summary"]
    overall = summary[summary["lang"] == "all"].set_index("model")
    assert set(overall.index) == set(RISK_MODELS)
    for _, row in overall.iterrows():
        assert row["n"] == len(labels)
        assert row["n_folds"] == DEFAULT_SPLITS
        assert 0.0 <= row["brier_pooled"] <= 1.0
        assert 0.0 <= row["accuracy_pooled"] <= 1.0
    # The fixture has a deterministic language and model signal, so both learned
    # models must separate incorrect answers better than chance.
    assert overall.loc["logistic", "roc_auc_pooled"] > 0.5
    assert overall.loc["lightgbm", "roc_auc_pooled"] > 0.5
    per_language = summary[summary["lang"] != "all"]
    assert set(per_language["lang"]) == {"kk", "ru", "en"}
    assert per_language.groupby("lang")["n"].sum().min() > 0


def test_single_class_labels_are_rejected(labels: pd.DataFrame) -> None:
    single = labels.copy()
    single[TARGET_RISK] = 0
    with pytest.raises(ValueError, match="single class"):
        cross_validate_risk(single, n_splits=DEFAULT_SPLITS, seed=SEED)


def test_calibration_curve_is_bounded(results: dict) -> None:
    calibration = calibration_frame(results["oof"])
    assert set(calibration["model"]) == set(RISK_MODELS)
    assert calibration["predicted_risk"].between(0.0, 1.0).all()
    assert calibration["observed_incorrect"].between(0.0, 1.0).all()
    assert (calibration["n"] > 0).all()


def test_shap_explanations_cover_every_held_out_row(results: dict) -> None:
    explanations = results["explanations"]
    matrix = results["matrix"]
    assert set(explanations) == set(LEARNED)
    for name, table in explanations.items():
        assert table.shape == matrix.shape, name
        values = table.to_numpy()
        assert np.isfinite(values).all()
        assert np.abs(values).mean() > 0.0
    kazakh = explanations["lightgbm"]["kk_share"].abs().mean()
    assert kazakh > 0.0


def test_same_seed_gives_same_risk(labels: pd.DataFrame) -> None:
    subset = labels[labels["model"] == min(set(labels["model"]))].reset_index(drop=True)
    first = cross_validate_risk(subset, n_splits=DEFAULT_SPLITS, seed=SEED)["oof"]
    second = cross_validate_risk(subset, n_splits=DEFAULT_SPLITS, seed=SEED)["oof"]
    pd.testing.assert_frame_equal(first, second)


def test_train_risk_writes_labelled_outputs(
    labels: pd.DataFrame, tmp_path: Path
) -> None:
    figs = tmp_path / "figs"
    results = train_risk(
        out_dir=tmp_path, figs_dir=figs, n_splits=DEFAULT_SPLITS, seed=SEED
    )
    manifest = json.loads(
        (tmp_path / "risk_metrics_fixture.json").read_text(encoding="utf-8")
    )
    assert manifest["fixture"] is True
    assert manifest["seed"] == SEED
    assert manifest["group_by"] == "q_id"
    assert manifest["target"] == TARGET_RISK
    assert manifest["inputs"]["n_questions"] == 60
    assert manifest["command"].startswith("python -m src.train risk")
    for name in RISK_MODELS:
        assert (tmp_path / f"risk_{name}_fixture.joblib").exists()
    for name in LEARNED:
        figure = figs / f"shap_risk_{name}_fixture.png"
        assert figure.exists() and figure.stat().st_size > 5000
        table = pd.read_csv(tmp_path / f"shap_risk_{name}_fixture.csv")
        assert len(table) == len(labels)
    assert (tmp_path / "risk_oof_fixture.csv").exists()
    assert (tmp_path / "risk_calibration_fixture.csv").exists()
    payload = joblib.load(tmp_path / "risk_lightgbm_fixture.joblib")
    assert payload["meta_columns"] == ["lang", "category", "model"]
    assert TARGET_RISK not in payload["columns"]
    assert results["summary"]["n"].min() > 0


def test_risk_survives_a_missing_matplotlib(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Figures are optional: matplotlib ships only in the shap `plots` extra."""
    import sys as sys_module

    monkeypatch.setitem(sys_module.modules, "matplotlib", None)
    monkeypatch.setitem(sys_module.modules, "matplotlib.pyplot", None)
    train_risk(out_dir=tmp_path, figs_dir=tmp_path / "figs", n_splits=3, seed=SEED)
    assert list((tmp_path / "figs").glob("*.png")) == []
    manifest = json.loads(
        (tmp_path / "risk_metrics_fixture.json").read_text(encoding="utf-8")
    )
    assert any("matplotlib" in note for note in manifest["notes"])
    assert not [name for name in manifest["artifacts"] if name.endswith(".png")]
    for name in LEARNED:
        table = pd.read_csv(tmp_path / f"shap_risk_{name}_fixture.csv")
        assert len(table) == manifest["inputs"]["n_rows"]
