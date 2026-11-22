"""Offline tests for release-evidence validation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.done_check import _csv, _jsonl, _measurement_errors, _provenance


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
                labels.append({
                    "model": model,
                    "prompt_id": prompt_id,
                    "lang": lang,
                    "rep": rep,
                    "parsed_answer": "4",
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


def test_malformed_csv_and_jsonl_are_reported(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("model,prompt_id\na\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        _csv(csv_path, {"model", "prompt_id", "lang"})

    json_path = tmp_path / "bad.jsonl"
    json_path.write_text(json.dumps({"ok": True}) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON at line 2"):
        _jsonl(json_path)
