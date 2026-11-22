"""Split, leakage and reproducibility guarantees for the T-005 cost predictor."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.common.config import SEED
from src.features import (
    FIXTURE_DIR,
    assert_no_leakage,
    design_matrix,
    load_cost_dataset,
    text_features,
)
from src.train import (
    COST_MODELS,
    DEFAULT_SPLITS,
    LanguageMeanBaseline,
    assign_folds,
    cross_validate_cost,
    train_cost,
)

PROMPTS_FIXTURE = FIXTURE_DIR / "prompts_fixture.csv"
TOKENS_FIXTURE = FIXTURE_DIR / "tokens_fixture.csv"
PHRASES = {
    "en": "How many pens are in seven boxes?",
    "ru": "Сколько ручек в семи коробках?",
    "kk": "Қазақстанның ең биік шыңы қалай аталады?",
}


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    return load_cost_dataset(PROMPTS_FIXTURE, TOKENS_FIXTURE)


@pytest.fixture(scope="session")
def results(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return cross_validate_cost(dataset, n_splits=DEFAULT_SPLITS, seed=SEED)


def test_features_on_three_hand_written_phrases() -> None:
    english = text_features(PHRASES["en"])
    assert english["n_chars"] == 33
    assert english["n_words"] == 7
    assert english["max_word_chars"] == 6
    assert english["mean_word_chars"] == pytest.approx(27 / 7)
    assert english["latin_share"] == 1.0
    assert english["kk_share"] == 0.0

    russian = text_features(PHRASES["ru"])
    assert russian["n_chars"] == 30
    assert russian["n_words"] == 5
    assert russian["cyrillic_share"] == 1.0
    assert russian["kk_share"] == 0.0

    kazakh = text_features(PHRASES["kk"])
    assert kazakh["n_words"] == 6
    assert kazakh["cyrillic_share"] == 1.0
    assert kazakh["latin_share"] == 0.0
    # hand count: 34 letters, 7 Kazakh-specific (Қ қ ң ең ң і шыңы ң қалай қ)
    assert kazakh["kk_share"] == pytest.approx(7 / 34)
    assert russian["kk_share"] == english["kk_share"] == 0.0
    assert kazakh["kk_share"] > 0.0


def test_no_q_id_in_both_train_and_test(dataset: pd.DataFrame) -> None:
    folds = assign_folds(list(dataset["q_id"]), DEFAULT_SPLITS, SEED)
    questions = dataset["q_id"].to_numpy()
    assert len(folds) == len(dataset)
    for fold in range(DEFAULT_SPLITS):
        train = set(questions[folds != fold])
        test = set(questions[folds == fold])
        assert train and test
        assert not train.intersection(test)
        assert train.union(test) == set(questions)


def test_questions_keep_all_translations_and_tokenizers(
    dataset: pd.DataFrame, results: dict[str, pd.DataFrame]
) -> None:
    assignment = results["oof"][
        ["prompt_id", "tokenizer", "q_id", "fold"]
    ].drop_duplicates()
    per_question = assignment.groupby("q_id")["fold"].nunique()
    assert per_question.max() == 1
    joined = dataset.merge(assignment, on=["prompt_id", "tokenizer", "q_id"])
    assert len(joined) == len(dataset)
    assert joined.groupby("q_id")["lang"].nunique().max() == 3
    assert (
        joined.groupby("q_id")["tokenizer"].nunique().max()
        == dataset["tokenizer"].nunique()
    )


def test_predictions_are_finite(
    dataset: pd.DataFrame, results: dict[str, pd.DataFrame]
) -> None:
    oof = results["oof"]
    assert set(oof["model"]) == set(COST_MODELS)
    assert np.isfinite(oof["prediction"].to_numpy()).all()
    assert (oof["absolute_error"] >= 0).all()
    per_model = oof.groupby(["model", "tokenizer"]).size()
    assert (per_model == len(dataset) // dataset["tokenizer"].nunique()).all()


def test_same_seed_gives_same_result(dataset: pd.DataFrame) -> None:
    single = dataset[dataset["tokenizer"] == min(set(dataset["tokenizer"]))]
    first = cross_validate_cost(single, n_splits=DEFAULT_SPLITS, seed=SEED)
    second = cross_validate_cost(single, n_splits=DEFAULT_SPLITS, seed=SEED)
    pd.testing.assert_frame_equal(first["oof"], second["oof"])
    pd.testing.assert_frame_equal(first["summary"], second["summary"])


def test_measured_columns_never_reach_the_models(dataset: pd.DataFrame) -> None:
    matrix = design_matrix(dataset)
    assert "token_count" not in matrix.columns
    assert not [column for column in matrix.columns if column.startswith("fertility")]
    with pytest.raises(ValueError, match="must not be model inputs"):
        assert_no_leakage(["n_chars", "fertility", "token_count_bge"])


def test_language_mean_baseline_uses_training_means(dataset: pd.DataFrame) -> None:
    block = dataset.reset_index(drop=True)
    matrix = design_matrix(block)
    baseline = LanguageMeanBaseline().fit(matrix, block["token_count"].to_numpy(float))
    predicted = pd.Series(baseline.predict(matrix), index=block.index)
    expected = block.groupby("lang")["token_count"].transform("mean")
    assert predicted.round(6).equals(expected.astype(float).round(6))


def test_summary_reports_mae_and_r2_with_n(
    dataset: pd.DataFrame, results: dict[str, pd.DataFrame]
) -> None:
    summary = results["summary"]
    assert list(summary.columns) == [
        "model",
        "tokenizer",
        "lang",
        "n",
        "n_folds",
        "mae_mean",
        "mae_std",
        "r2_mean",
        "r2_std",
    ]
    overall = summary[summary["lang"] == "all"]
    per_tokenizer = dataset.groupby("tokenizer").size()
    for _, row in overall.iterrows():
        assert row["n"] == per_tokenizer.loc[row["tokenizer"]]
        assert row["n_folds"] == DEFAULT_SPLITS
        assert row["mae_mean"] >= 0
    assert set(overall["model"]) == set(COST_MODELS)


def test_train_cost_writes_labelled_artifacts(tmp_path: Path) -> None:
    results = train_cost(out_dir=tmp_path, n_splits=DEFAULT_SPLITS, seed=SEED)
    manifest_path = tmp_path / "cost_metrics_fixture.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["fixture"] is True
    assert manifest["seed"] == SEED
    assert manifest["group_by"] == "q_id"
    assert manifest["models"] == list(COST_MODELS)
    assert manifest["inputs"]["n_questions"] == 60
    assert manifest["command"].startswith("python -m src.train cost")
    artifacts = sorted(tmp_path.glob("cost_*.joblib"))
    assert len(artifacts) == len(COST_MODELS) * len(manifest["inputs"]["tokenizers"])
    payload = joblib.load(artifacts[0])
    assert payload["fixture"] is True
    assert "n_chars" in payload["columns"]
    assert "token_count" not in payload["columns"]
    assert (tmp_path / "cost_oof_fixture.csv").exists()
    assert (tmp_path / "cost_folds_fixture.csv").exists()
    assert results["summary"]["n"].min() > 0
