"""Split and leakage guarantees for the T-005 token-cost predictor."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.features import (
    LEAKED_COLUMNS,
    assert_no_leakage,
    design_matrix,
    load_dataset,
)
from src.train import assign_folds, cross_validate_cost, save_outputs

N_QUESTIONS = 12
N_SPLITS = 5
SEED = 42
CATEGORIES = ("arithmetic", "geography")
ANSWER_TYPES = ("integer", "short_text")
BASE_TEXTS = {
    "kk": "Қазақстанның ең биік шыңы қалай аталады, оның биіктігі неше метр?",
    "ru": "Как называется самая высокая вершина Казахстана и какова её высота?",
    "en": "What is the name of the highest peak in Kazakhstan and how tall is it?",
}
TOKENIZER_RATES = {
    "tiktoken/cl100k_base": {"kk": 0.72, "ru": 0.61, "en": 0.26},
    "hf-tokenizers/BAAI/bge-m3@a1b2c3d": {"kk": 0.58, "ru": 0.49, "en": 0.31},
}


def write_inputs(root: Path) -> tuple[Path, Path]:
    """Write a tiny contract v1 prompts.csv and tokens.csv pair."""
    prompts_path = root / "prompts.csv"
    tokens_path = root / "tokens.csv"
    prompt_rows: list[list[str]] = []
    token_rows: list[list[str]] = []
    for index in range(N_QUESTIONS):
        q_id = f"q{index:03d}"
        for lang in ("kk", "ru", "en"):
            prompt_id = f"{q_id}-{lang}"
            text = f"{BASE_TEXTS[lang]} [{index}]"
            prompt_rows.append(
                [
                    prompt_id,
                    q_id,
                    lang,
                    CATEGORIES[index % len(CATEGORIES)],
                    text,
                    ANSWER_TYPES[index % len(ANSWER_TYPES)],
                    str(index),
                    "",
                ]
            )
            words = len(text.split())
            for tokenizer, rates in TOKENIZER_RATES.items():
                tokens = max(1, math.ceil(len(text) * rates[lang]))
                token_rows.append(
                    [
                        prompt_id,
                        q_id,
                        lang,
                        tokenizer,
                        str(words),
                        str(tokens),
                        f"{tokens / words:.6f}",
                    ]
                )
    with prompts_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "prompt_id",
                "q_id",
                "lang",
                "category",
                "prompt",
                "answer_type",
                "gold_answer",
                "verified_by_native",
            ]
        )
        writer.writerows(prompt_rows)
    with tokens_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "prompt_id",
                "q_id",
                "lang",
                "tokenizer",
                "word_count",
                "token_count",
                "fertility",
            ]
        )
        writer.writerows(token_rows)
    return prompts_path, tokens_path


@pytest.fixture(scope="session")
def paths(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("token-cost")
    prompts_path, tokens_path = write_inputs(root)
    return root, prompts_path, tokens_path


@pytest.fixture(scope="session")
def dataset(paths: tuple[Path, Path, Path]) -> pd.DataFrame:
    return load_dataset(paths[1], paths[2])


@pytest.fixture(scope="session")
def results(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return cross_validate_cost(dataset, n_splits=N_SPLITS, seed=SEED)


def test_folds_have_no_qid_overlap(results: dict[str, pd.DataFrame]) -> None:
    folds = results["folds"].set_index("q_id")["fold"]
    assert folds.index.is_unique
    assert sorted(folds.unique()) == list(range(N_SPLITS))


def test_question_keeps_all_langs_and_tokenizers(
    results: dict[str, pd.DataFrame], dataset: pd.DataFrame
) -> None:
    oof = results["oof"]
    per_question = oof.groupby("q_id")["fold"].nunique()
    assert per_question.max() == 1
    assignment = oof[["prompt_id", "tokenizer", "fold"]].drop_duplicates()
    assert assignment.duplicated(subset=["prompt_id", "tokenizer"]).sum() == 0
    joined = dataset.merge(assignment, on=["prompt_id", "tokenizer"])
    assert len(joined) == len(dataset)
    assert joined.groupby("q_id")["fold"].nunique().max() == 1
    expected = dataset.groupby("q_id").agg(
        langs=("lang", "nunique"), tokenizers=("tokenizer", "nunique")
    )
    assert (expected["langs"] == 3).all()
    assert (expected["tokenizers"] == len(TOKENIZER_RATES)).all()


def test_fold_assignment_is_reproducible(dataset: pd.DataFrame) -> None:
    groups = list(dataset["q_id"])
    folds = assign_folds(groups, n_splits=N_SPLITS, seed=SEED)
    assert (folds == assign_folds(groups, n_splits=N_SPLITS, seed=SEED)).all()
    assert (
        pd.DataFrame({"q_id": groups, "fold": folds})
        .groupby("q_id")["fold"]
        .nunique()
        .max()
        == 1
    )


def test_assign_folds_rejects_too_few_groups() -> None:
    with pytest.raises(ValueError, match="distinct q_id groups"):
        assign_folds(["q1", "q1", "q2"], n_splits=5, seed=SEED)


def test_features_exclude_measured_counts(dataset: pd.DataFrame) -> None:
    matrix = design_matrix(dataset)
    assert LEAKED_COLUMNS.isdisjoint(matrix.columns)
    assert matrix.notna().all().all()
    with pytest.raises(ValueError, match="must not be model inputs"):
        assert_no_leakage(["char_count", "token_count"])


def test_metrics_cover_model_language_tokenizer(
    results: dict[str, pd.DataFrame], dataset: pd.DataFrame
) -> None:
    metrics = results["metrics"]
    assert list(metrics.columns) == ["model", "tokenizer", "lang", "n", "mae"]
    assert set(metrics["model"]) == {"linear", "lightgbm"}
    detailed = metrics[(metrics["lang"] != "all") & (metrics["tokenizer"] != "all")]
    expected = dataset.groupby(["tokenizer", "lang"]).size()
    for _, row in detailed.iterrows():
        assert row["n"] == expected.loc[(row["tokenizer"], row["lang"])]
        assert row["mae"] >= 0
    overall = metrics[(metrics["lang"] == "all") & (metrics["tokenizer"] == "all")]
    assert len(overall) == 2
    assert list(overall["n"]) == [len(dataset)] * 2
    oof = results["oof"]
    assert oof["absolute_error"].min() >= 0
    per_pair = oof.groupby(["model", "tokenizer"]).size()
    assert (per_pair == len(dataset) // len(TOKENIZER_RATES)).all()


def test_save_outputs_writes_manifest_and_artifacts(
    results: dict[str, pd.DataFrame],
    dataset: pd.DataFrame,
    paths: tuple[Path, Path, Path],
) -> None:
    root, prompts_path, tokens_path = paths
    out_dir = root / "models" / "token_cost"
    written = save_outputs(
        results,
        dataset,
        out_dir=out_dir,
        command="python -m src.train --seed 42 --n-splits 5",
        n_splits=N_SPLITS,
        seed=SEED,
        prompts_path=prompts_path,
        tokens_path=tokens_path,
    )
    for path in written.values():
        assert path.exists()
    manifest = json.loads(written["run"].read_text(encoding="utf-8"))
    assert manifest["seed"] == SEED
    assert manifest["n_splits"] == N_SPLITS
    assert manifest["command"] == "python -m src.train --seed 42 --n-splits 5"
    assert manifest["group_by"] == "q_id"
    assert manifest["inputs"]["n_questions"] == N_QUESTIONS
    assert manifest["inputs"]["tokenizers"] == sorted(TOKENIZER_RATES)
    artifacts = sorted(out_dir.glob("cost_*.joblib"))
    assert len(artifacts) == 2 * len(TOKENIZER_RATES)
    folds = pd.read_csv(written["folds"])
    assert sorted(folds["q_id"]) == sorted(dataset["q_id"].unique())
