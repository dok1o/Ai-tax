import csv
import json
from pathlib import Path

from src.collect import (
    DEFAULT_MODELS,
    ClientFailure,
    Completion,
    FakeClient,
    run_collection,
)


def prompt() -> dict[str, str]:
    return {
        "prompt_id": "q1-kk",
        "q_id": "q1",
        "lang": "kk",
        "prompt": "2 қосу 2 неше болады?",
        "answer_type": "integer",
        "gold_answer": "4",
    }


def test_cache_hit_skips_second_call_and_preserves_one_csv_row(tmp_path: Path) -> None:
    client = FakeClient()
    response_path = tmp_path / "responses.csv"
    raw_dir = tmp_path / "raw"

    first = run_collection(
        [prompt()], client, models=(DEFAULT_MODELS[0],), raw_dir=raw_dir,
        responses_path=response_path, sleeper=lambda _: None,
    )
    second = run_collection(
        [prompt()], client, models=(DEFAULT_MODELS[0],), raw_dir=raw_dir,
        responses_path=response_path, sleeper=lambda _: None,
    )

    assert len(client.calls) == 1
    assert len(first) == len(second) == 1
    assert len((raw_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    with response_path.open(encoding="utf-8", newline="") as source:
        assert len(list(csv.DictReader(source))) == 1


def test_missing_provider_usage_preserves_the_answer_on_restart(tmp_path: Path) -> None:
    model = DEFAULT_MODELS[0]
    client = FakeClient({(model, "q1-kk", "kk", 0): [Completion("4", input_tokens=3)]})
    path = tmp_path / "responses.csv"
    for _ in range(2):
        run_collection([prompt()], client, (model,), raw_dir=tmp_path, responses_path=path)
    with path.open(encoding="utf-8", newline="") as source:
        row, = csv.DictReader(source)
    assert len(client.calls) == 1
    assert (row["response"], row["input_tokens"], row["output_tokens"], row["usage_available"]) == ("4", "3", "", "false")


def test_retries_transient_failure_with_backoff(tmp_path: Path) -> None:
    key = (DEFAULT_MODELS[0], "q1-kk", "kk", 0)
    client = FakeClient(
        {key: [ClientFailure("temporary", True), Completion("4", "fixture-v1", 5, 1)]}
    )
    delays: list[float] = []

    result = run_collection(
        [prompt()], client, models=(DEFAULT_MODELS[0],), raw_dir=tmp_path / "raw",
        responses_path=tmp_path / "responses.csv", sleeper=delays.append,
    )

    assert len(client.calls) == 2
    assert delays == [0.1]
    assert len(result) == 1


def test_terminal_failure_is_logged_without_retry(tmp_path: Path) -> None:
    key = (DEFAULT_MODELS[0], "q1-kk", "kk", 0)
    client = FakeClient({key: [ClientFailure("bad request", False)]})
    raw_dir = tmp_path / "raw"

    result = run_collection(
        [prompt()], client, models=(DEFAULT_MODELS[0],), raw_dir=raw_dir,
        responses_path=tmp_path / "responses.csv", sleeper=lambda _: None,
    )

    attempts = [
        json.loads(line)
        for line in (raw_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert client.calls == [key]
    assert result == []
    assert attempts[0]["outcome"] == "failed"
    assert attempts[0]["client_type"] == "fixture"


def test_attempt_log_has_required_metadata_and_marks_fixture(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    run_collection(
        [prompt()], FakeClient(), models=(DEFAULT_MODELS[0],), raw_dir=raw_dir,
        responses_path=tmp_path / "responses.csv", sleeper=lambda _: None,
    )
    attempt = json.loads(
        (raw_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    assert {
        "model", "model_version", "called_at", "temperature", "input_tokens",
        "output_tokens", "latency_ms", "outcome",
    }.issubset(attempt)
    assert attempt["client_type"] == "fixture"
    assert attempt["usage_status"] == "reported"


def test_default_client_is_fake_and_does_not_open_network(tmp_path: Path, monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("unexpected network access")

    monkeypatch.setattr("src.collect.urlopen", fail_if_called)
    rows = run_collection(
        [prompt()], None, models=DEFAULT_MODELS, raw_dir=tmp_path / "raw",
        responses_path=tmp_path / "responses.csv", sleeper=lambda _: None,
    )

    assert len(rows) == 2
    assert all(str(row["model"]).startswith("fixture/") for row in rows)
