"""Regression tests for the ARENA review fixes.

Three findings are pinned here:

1. the router picks its threshold on training folds only, nested by ``q_id``, and
   deploys the mode of those per-fold choices instead of re-selecting on the
   outcomes it is judged on;
2. correctness flags are decoded by one function with one polarity, so a numeric
   ``1`` is never read as incorrect and an unparsed answer is scored false;
3. the cost and risk pipelines resolve a prompts file that matches the partner
   table, which is what made risk training - and therefore its figures - fail as
   soon as a real ``data/prompts.csv`` sat next to fixture labels.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from src import features, router
from src.common.config import SEED
from src.features import FIXTURE_DIR, decode_correctness, load_labels, resolve_input
from src.router import deployment_threshold, train_router
from src.train import RISK_MODELS, train_cost, train_risk

PROMPTS_FIXTURE = FIXTURE_DIR / "prompts_fixture.csv"
LABELS_FIXTURE = FIXTURE_DIR / "labels_fixture.csv"
LEARNED = ("logistic", "lightgbm")


def write_labels(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a contract-shaped labels.csv with the v2 parse_status column."""
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def label_rows(
    flags: list[str], statuses: list[str] | None = None
) -> list[dict[str, str]]:
    """One graded row per flag, taken from the first questions of the fixture."""
    prompts = pd.read_csv(PROMPTS_FIXTURE, dtype=str, keep_default_na=False)
    statuses = statuses or ["graded"] * len(flags)
    rows = []
    for index, (flag, status) in enumerate(zip(flags, statuses, strict=True)):
        prompt = prompts.iloc[index]
        rows.append(
            {
                "model": "fixture-small-model",
                "prompt_id": prompt["prompt_id"],
                "lang": prompt["lang"],
                "rep": "0",
                "q_id": prompt["q_id"],
                "parsed_answer": prompt["gold_answer"],
                "is_correct": flag,
                "parse_status": status,
            }
        )
    return rows


def test_numeric_flags_keep_the_correct_polarity(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.csv"
    write_labels(labels_path, label_rows(["1", "0", "true", "false"]))
    frame = load_labels(PROMPTS_FIXTURE, labels_path)
    assert frame["is_correct"].tolist() == [1, 0, 1, 0]


def test_unparsed_answers_are_scored_false(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.csv"
    write_labels(
        labels_path,
        label_rows(["true", "true", "false"], ["graded", "unparsed", "unparsed"]),
    )
    frame = load_labels(PROMPTS_FIXTURE, labels_path)
    assert frame["is_correct"].tolist() == [1, 0, 0]


def test_unknown_correctness_and_status_values_are_rejected(tmp_path: Path) -> None:
    bad_flag = tmp_path / "bad_flag.csv"
    write_labels(bad_flag, label_rows(["yes"]))
    with pytest.raises(ValueError, match="non-boolean is_correct"):
        load_labels(PROMPTS_FIXTURE, bad_flag)
    bad_status = tmp_path / "bad_status.csv"
    write_labels(bad_status, label_rows(["true"], ["maybe"]))
    with pytest.raises(ValueError, match="unknown parse_status"):
        load_labels(PROMPTS_FIXTURE, bad_status)


def test_router_and_training_share_one_decoder() -> None:
    assert router.decode_correctness is decode_correctness
    decoded = decode_correctness(
        pd.Series(["1", "0", "true", "false"]), source="routes"
    )
    assert decoded["is_correct"].tolist() == [1, 0, 1, 0]
    assert decoded["is_incorrect"].tolist() == [0, 1, 0, 1]


def test_deployment_threshold_is_the_nested_mode() -> None:
    assert deployment_threshold({0: 0.5, 1: 0.95, 2: 0.95, 3: 0.95, 4: 0.95}) == 0.95
    # A tie breaks towards the smaller threshold: cheaper routes win the tie.
    assert deployment_threshold({0: 0.5, 1: 0.5, 2: 0.95, 3: 0.95, 4: 0.2}) == 0.5
    with pytest.raises(ValueError, match="no per-fold thresholds"):
        deployment_threshold({})


def test_prompts_follow_the_provenance_of_the_table_they_join(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_prompts = tmp_path / "prompts.csv"
    shutil.copy(PROMPTS_FIXTURE, real_prompts)
    monkeypatch.setattr(features, "PROMPTS_PATH", real_prompts)
    monkeypatch.setattr(features, "LABELS_PATH", tmp_path / "absent_labels.csv")
    monkeypatch.setattr(features, "TOKENS_PATH", tmp_path / "absent_tokens.csv")
    monkeypatch.setattr(features, "ROUTES_PATH", tmp_path / "absent_routes.csv")

    path, is_fixture = resolve_input("prompts")
    assert path == real_prompts and is_fixture is False
    for partner in ("labels", "tokens", "routes"):
        joined, joined_fixture = resolve_input("prompts", partner=partner)
        assert joined == FIXTURE_DIR / "prompts_fixture.csv"
        assert joined_fixture is True


def test_risk_training_writes_its_figures_next_to_a_real_prompts_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The reviewed failure: risk training died before any figure was written."""
    real_prompts = tmp_path / "prompts.csv"
    real_prompts.write_text(
        PROMPTS_FIXTURE.read_text(encoding="utf-8").replace("q000", "m000"),
        encoding="utf-8",
    )
    monkeypatch.setattr(features, "PROMPTS_PATH", real_prompts)
    monkeypatch.setattr(features, "LABELS_PATH", tmp_path / "absent_labels.csv")
    figs = tmp_path / "figs"
    train_risk(out_dir=tmp_path, figs_dir=figs, n_splits=3, seed=SEED)
    manifest = json.loads(
        (tmp_path / "risk_metrics_fixture.json").read_text(encoding="utf-8")
    )
    assert manifest["fixture"] is True
    assert manifest["inputs"]["prompts"].endswith("prompts_fixture.csv")
    for name in RISK_MODELS:
        assert (tmp_path / f"risk_{name}_fixture.joblib").exists()
    for name in LEARNED:
        figure = figs / f"shap_risk_{name}_fixture.png"
        assert figure.exists() and figure.stat().st_size > 5000


def test_cost_training_resolves_a_matching_prompts_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_prompts = tmp_path / "prompts.csv"
    shutil.copy(PROMPTS_FIXTURE, real_prompts)
    monkeypatch.setattr(features, "PROMPTS_PATH", real_prompts)
    monkeypatch.setattr(features, "TOKENS_PATH", tmp_path / "absent_tokens.csv")
    results = train_cost(out_dir=tmp_path, n_splits=3, seed=SEED)
    manifest = json.loads(
        (tmp_path / "cost_metrics_fixture.json").read_text(encoding="utf-8")
    )
    assert manifest["fixture"] is True
    assert manifest["inputs"]["prompts"].endswith("prompts_fixture.csv")
    assert not results["summary"].empty


def test_router_never_selects_a_threshold_on_the_rows_it_evaluates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Selection sees training folds only; deployment is the mode of those picks."""
    total_rows = len(pd.read_csv(resolve_input("routes")[0], dtype=str))
    seen: list[int] = []
    original = router.select_threshold

    def spy(frame: pd.DataFrame, thresholds: object) -> float:
        seen.append(len(frame))
        return original(frame, thresholds)

    monkeypatch.setattr(router, "select_threshold", spy)
    train_router(
        out_dir=tmp_path,
        figs_dir=tmp_path / "figs",
        n_splits=3,
        seed=SEED,
        n_resamples=50,
    )
    manifest = json.loads(
        (tmp_path / "router_metrics_fixture.json").read_text(encoding="utf-8")
    )
    per_fold = {
        int(fold): value for fold, value in manifest["thresholds_per_fold"].items()
    }
    assert seen and max(seen) < total_rows, "a selection saw every held-out row"
    assert len(seen) == len(per_fold)
    assert manifest["deployment_threshold"] == deployment_threshold(per_fold)
    assert manifest["deployment_threshold"] in set(per_fold.values())
    counts = Counter(per_fold.values())
    assert counts[manifest["deployment_threshold"]] == max(counts.values())
    decisions = pd.read_csv(tmp_path / "router_decisions_fixture.csv")
    # One decision per held-out question and language, exactly one fold each.
    assert decisions["fold"].nunique() == 3
    assert decisions.groupby(["q_id", "lang"]).size().max() == 1
    saved = {
        int(fold): float(value)
        for fold, value in decisions.groupby("fold")["threshold"].max().items()
    }
    assert saved == per_fold
