"""Offline tests for release-evidence validation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.done_check import (
    _csv,
    _fold_errors,
    _jsonl,
    _measurement_errors,
    _model_comparison_errors,
    _provenance,
    _router_observation_errors,
    _router_savings_errors,
)


def _complete_fixture() -> tuple[list[dict[str, str]], ...]:
    prompts: list[dict[str, str]] = []
    responses: list[dict[str, str]] = []
    labels: list[dict[str, str]] = []
    tokens: list[dict[str, str]] = []
    attempts: list[dict[str, str]] = []
    raw: list[dict[str, str]] = []
    for lang in ("kk", "ru", "en"):
        prompt_id = f"q1-{lang}"
        prompts.append({"prompt_id": prompt_id, "q_id": "q1", "lang": lang})
        tokens.append({"prompt_id": prompt_id, "word_count": "2", "token_count": "3", "fertility": "1.5"})
        for model in ("model-a", "model-b"):
            for rep in ("0", "1"):
                key = (model, prompt_id, lang, rep)
                responses.append(dict(zip(("model", "prompt_id", "lang", "rep", "q_id"), (*key, "q1"))))
                responses[-1].update(input_tokens="3", output_tokens="1", usage_available="true")
                labels.append({
                    "model": model,
                    "prompt_id": prompt_id,
                    "lang": lang,
                    "rep": rep,
                    "parsed_answer": "4",
                    "parse_status": "graded",
                    "is_correct": "true",
                })
                log = {
                    "model": model,
                    "model_version": "v1",
                    "called_at": "2026-11-22T00:00:00Z",
                    "temperature": "0",
                    "input_tokens": 3,
                    "output_tokens": 1,
                    "latency_ms": 10,
                    "source": "provider",
                    "outcome": "success",
                    "prompt_id": prompt_id,
                    "lang": lang,
                    "rep": rep,
                }
                attempts.append(log)
                raw.append({**log, "response": "4"})
    return prompts, responses, labels, tokens, attempts, raw


def test_complete_measurement_evidence_passes() -> None:
    assert _measurement_errors(*_complete_fixture()) == []


def test_missing_successful_raw_call_fails() -> None:
    fixture = list(_complete_fixture())
    fixture[-1] = fixture[-1][:-1]
    assert "matching successful raw call" in "; ".join(_measurement_errors(*fixture))


def test_fixture_only_and_mixed_provenance_fail() -> None:
    fixture = _complete_fixture()
    assert "fixture-only" in _provenance([{"source": "fixture"}])
    assert "mixed" in _provenance([{"source": "provider"}, {"source": "example"}])
    assert _provenance(fixture[1]) == "provenance is not recorded in observation metadata"


def test_grouped_fold_rejects_qid_leakage() -> None:
    valid = [
        {"fold": "0", "split": "train", "q_id": "q1"},
        {"fold": "0", "split": "test", "q_id": "q2"},
    ]
    assert _fold_errors(valid) == []
    leaked = valid + [{"fold": "0", "split": "eval", "q_id": "q1"}]
    assert "leaks q_ids" in _fold_errors(leaked)[0]


def test_model_comparison_requires_training_baseline_and_reports_nulls() -> None:
    row = {
        "metric": "mae",
        "n": "10",
        "baseline": "training per-language/tokenizer mean",
        "model": "LightGBM",
        "baseline_value": "1.2",
        "model_value": "1.3",
        "signed_difference": "0.1",
    }
    assert _model_comparison_errors([row], "mae") == []
    row["baseline"] = "global mean"
    assert "training-derived" in _model_comparison_errors([row], "mae")[0]


def test_router_requires_identical_eligible_questions_and_paired_ci() -> None:
    observations = [
        {
            "policy": policy,
            "q_id": q_id,
            "eligible": "true",
            "accuracy": "0.8",
            "translation_cost": "0.2",
            "model_cost": "1.0",
            "total_cost": "1.2",
        }
        for policy in ("router", "always_direct", "always_via_english")
        for q_id in ("q1", "q2")
    ]
    assert _router_observation_errors(observations) == []
    assert "identical eligible q_ids" in _router_observation_errors(observations[:-1])[0]
    savings = [
        {
            "baseline": baseline,
            "mean_cost_saved": "0.1",
            "ci_low": "0.0",
            "ci_high": "0.2",
            "n_questions": "2",
            "accuracy_matched": "true",
            "bootstrap_unit": "q_id",
        }
        for baseline in ("always_direct", "always_via_english")
    ]
    assert _router_savings_errors(savings) == []


def test_malformed_csv_and_jsonl_are_reported(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("model,prompt_id\na\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        _csv(csv_path, {"model", "prompt_id", "lang"})

    json_path = tmp_path / "bad.jsonl"
    json_path.write_text(json.dumps({"ok": True}) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON at line 2"):
        _jsonl(json_path)
