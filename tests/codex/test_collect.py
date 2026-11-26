import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Self
from urllib.error import HTTPError

import pytest

from src.collect import (
    DEFAULT_MODELS,
    ClientFailure,
    Completion,
    FakeClient,
    GeminiClient,
    OllamaClient,
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
    assert delays == [1.0]
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


class JsonResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        return self.body


def test_ollama_sends_bounded_nonstreaming_request_and_reads_digest(monkeypatch) -> None:
    model = "qwen2.5:3b"
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        if request.full_url.endswith("/api/tags"):
            return JsonResponse({"models": [{"name": model, "digest": "sha256:abc"}]})
        return JsonResponse({
            "message": {"content": "B"},
            "prompt_eval_count": 19,
            "eval_count": 1,
        })

    monkeypatch.setattr("src.collect.urlopen", fake_urlopen)
    completion = OllamaClient().complete(model, {**prompt(), "rep": "1"}, 0.5)

    chat_request = calls[0][0]
    body = json.loads(chat_request.data)
    assert chat_request.full_url == "http://localhost:11434/api/chat"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.5, "seed": 1, "num_predict": 48}
    assert completion == Completion("B", "sha256:abc", 19, 1)
    assert calls[1][0].full_url.endswith("/api/tags")


def test_ollama_missing_digest_or_usage_is_unknown(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/api/tags"):
            return JsonResponse({"models": []})
        return JsonResponse({"message": {"content": "A"}})

    monkeypatch.setattr("src.collect.urlopen", fake_urlopen)
    completion = OllamaClient().complete("local-model", prompt(), 0.5)

    assert completion.model_version == "unknown"
    assert completion.input_tokens is None
    assert completion.output_tokens is None


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (503, True), (400, False)])
def test_ollama_http_failures_are_classified(status: int, retryable: bool, monkeypatch) -> None:
    def fail(request, timeout):
        raise HTTPError(request.full_url, status, "failure", None, None)

    monkeypatch.setattr("src.collect.urlopen", fail)
    with pytest.raises(ClientFailure) as raised:
        OllamaClient().complete("model", prompt(), 0.5)
    assert raised.value.retryable is retryable


class FakeGenerateConfig:
    def __init__(self, **values) -> None:
        self.values = values


class FakeGeminiModels:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeGeminiClient:
    def __init__(self, outcomes) -> None:
        self.models = FakeGeminiModels(outcomes)


def fake_gemini_types():
    return SimpleNamespace(
        GenerateContentConfig=FakeGenerateConfig,
        ThinkingConfig=FakeGenerateConfig,
        ThinkingLevel=SimpleNamespace(MINIMAL="minimal", LOW="low"),
    )


def gemini_result():
    return SimpleNamespace(
        text="A",
        model_version="gemini-3.6-flash-001",
        usage_metadata=SimpleNamespace(
            prompt_token_count=19,
            candidates_token_count=1,
            thoughts_token_count=4,
        ),
    )


def test_gemini_uses_rep_seed_thinking_level_and_separate_usage() -> None:
    now = [0.0]
    delays = []
    client = FakeGeminiClient([gemini_result(), gemini_result()])

    def sleep(delay: float) -> None:
        delays.append(delay)
        now[0] += delay

    gemini = GeminiClient(
        rpm=8,
        client=client,
        types_module=fake_gemini_types(),
        sleeper=sleep,
        clock=lambda: now[0],
    )
    first = gemini.complete("gemini-3.6-flash", {**prompt(), "rep": "0"}, 0.5)
    second = gemini.complete("gemini-3.6-flash", {**prompt(), "rep": "1"}, 0.5)

    configs = [call["config"].values for call in client.models.calls]
    assert [config["seed"] for config in configs] == [0, 1]
    assert all(config["temperature"] == 0.5 for config in configs)
    assert all(config["max_output_tokens"] == 48 for config in configs)
    assert configs[0]["thinking_config"].values == {"thinking_level": "minimal"}
    assert (first.input_tokens, first.output_tokens, first.thinking_tokens) == (19, 1, 4)
    assert second.model_version == "gemini-3.6-flash-001"
    assert delays == [7.5]


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (503, True), (400, False)])
def test_gemini_status_errors_are_retryable_only_for_429_and_5xx(
    status: int, retryable: bool
) -> None:
    error = RuntimeError("fake API error")
    error.code = status
    gemini = GeminiClient(
        client=FakeGeminiClient([error]), types_module=fake_gemini_types()
    )

    with pytest.raises(ClientFailure) as raised:
        gemini.complete("gemini-3.6-flash", prompt(), 0.5)
    assert raised.value.retryable is retryable


def test_gemini_thinking_tokens_are_saved_in_append_only_raw_log(tmp_path: Path) -> None:
    client = FakeGeminiClient([gemini_result()])
    gemini = GeminiClient(client=client, types_module=fake_gemini_types())
    raw_dir = tmp_path / "raw"
    response_path = tmp_path / "responses.csv"

    rows = run_collection(
        [prompt()], gemini, models=("gemini-3.6-flash",), raw_dir=raw_dir,
        responses_path=response_path, temperature=0.5, sleeper=lambda _: None,
    )
    attempt = json.loads((raw_dir / "attempts.jsonl").read_text(encoding="utf-8"))
    cached = json.loads((raw_dir / "responses.jsonl").read_text(encoding="utf-8"))

    assert len(rows) == 1
    assert attempt["thinking_tokens"] == cached["thinking_tokens"] == 4
    with response_path.open(encoding="utf-8", newline="") as source:
        assert "thinking_tokens" not in csv.DictReader(source).fieldnames


def test_gemini_429_retry_uses_backoff_and_rpm_throttle(tmp_path: Path) -> None:
    now = [0.0]
    delays = []

    def sleep(delay: float) -> None:
        delays.append(delay)
        now[0] += delay

    error = RuntimeError("fake rate limit")
    error.code = 429
    client = FakeGeminiClient([error, gemini_result()])
    gemini = GeminiClient(
        rpm=8,
        client=client,
        types_module=fake_gemini_types(),
        sleeper=sleep,
        clock=lambda: now[0],
    )
    run_collection(
        [prompt()], gemini, models=("gemini-3.6-flash",),
        raw_dir=tmp_path / "raw", responses_path=tmp_path / "responses.csv",
        temperature=0.5, sleeper=sleep,
    )

    assert len(client.models.calls) == 2
    assert delays == [1.0, 6.5]
