"""Collect cached model responses; fake mode is the safe default.

Ollama input_tokens include the model's constant chat-template overhead.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.common.config import DEMO_DIR, PROMPTS_PATH, RAW_DIR, RESPONSES_PATH

RESPONSE_COLUMNS = (
    "model",
    "prompt_id",
    "lang",
    "rep",
    "q_id",
    "model_version",
    "called_at",
    "temperature",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "response",
    "usage_available",
)
DEFAULT_MODELS = ("fixture/model-a", "fixture/model-b")
DEFAULT_TEMPERATURE = 0.5
MAX_OUTPUT_TOKENS = 48
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_ATTEMPTS = 3
BASE_BACKOFF_SECONDS = 1.0
Key = tuple[str, str, str, int]


@dataclass(frozen=True)
class Completion:
    response: str
    model_version: str = "unknown"
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None


class ClientFailure(Exception):
    def __init__(self, reason: str, retryable: bool) -> None:
        super().__init__(reason)
        self.retryable = retryable


class Client(Protocol):
    def complete(
        self, model: str, prompt: Mapping[str, str], temperature: float
    ) -> Completion: ...


class FakeClient:
    """Deterministic fixture client with optional per-key scripted outcomes."""

    def __init__(
        self,
        scripts: Mapping[Key, Sequence[Completion | ClientFailure]] | None = None,
    ) -> None:
        self.scripts = {key: list(outcomes) for key, outcomes in (scripts or {}).items()}
        self.calls: list[Key] = []

    def complete(
        self, model: str, prompt: Mapping[str, str], temperature: float
    ) -> Completion:
        del temperature
        key = (model, prompt["prompt_id"], prompt["lang"], int(prompt.get("rep", "0")))
        self.calls.append(key)
        outcomes = self.scripts.get(key, [])
        if outcomes:
            outcome = outcomes.pop(0)
            if isinstance(outcome, ClientFailure):
                raise outcome
            return outcome

        answer = prompt["gold_answer"]
        q_number = int("".join(char for char in prompt["q_id"] if char.isdigit()) or "0")
        language_stride = {"kk": 3, "ru": 5, "en": 7}.get(prompt["lang"], 11)
        if model == "fixture/model-b" and q_number % language_stride == 0:
            answer = _wrong_fixture_answer(prompt.get("answer_type", "short_text"), answer)
        return Completion(
            response=answer,
            model_version="fixture-v1",
            input_tokens=len(prompt["prompt"].split()),
            output_tokens=len(answer.split()),
        )


def _wrong_fixture_answer(answer_type: str, gold_answer: str) -> str:
    if answer_type == "integer":
        return str(int(gold_answer) + 1)
    if answer_type in {"decimal", "number"}:
        return str(float(gold_answer) + 1.0)
    if answer_type == "mcq":
        return "B" if gold_answer == "A" else "A"
    if answer_type == "boolean":
        return "false" if gold_answer == "true" else "true"
    return "fixture answer"


class RealClient:
    """OpenAI-compatible client; instantiated only by an explicit --live flag."""

    def __init__(self) -> None:
        self.base_url = os.environ.get("TAX_API_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("TAX_API_KEY", "")
        if not self.base_url or not self.api_key:
            raise ValueError("Live mode requires TAX_API_BASE_URL and TAX_API_KEY environment variables.")

    def complete(
        self, model: str, prompt: Mapping[str, str], temperature: float
    ) -> Completion:
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt["prompt"]}],
                "temperature": temperature,
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.load(response)
        except HTTPError as error:
            raise ClientFailure(f"HTTP status {error.code}", error.code == 429 or error.code >= 500) from None
        except (URLError, TimeoutError):
            raise ClientFailure("network error", True) from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ClientFailure("invalid JSON response", False) from None

        try:
            content = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage") or {}
            if not isinstance(usage, dict):
                raise TypeError("Invalid usage object")
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            model_version = str(payload.get("model") or "unknown")
        except (KeyError, IndexError, TypeError, ValueError):
            raise ClientFailure("response or usage metadata unavailable", False) from None
        if not isinstance(content, str) or any(
            count is not None and (type(count) is not int or count < 0)
            for count in (input_tokens, output_tokens)
        ):
            raise ClientFailure("response or usage metadata invalid", False)
        return Completion(content, model_version, input_tokens, output_tokens)


def _nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


class OllamaClient:
    """Non-streaming Ollama REST client for explicit live collection only."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._digests: dict[str, str] = {}

    def _model_digest(self, model: str) -> str:
        if model in self._digests:
            return self._digests[model]
        digest = "unknown"
        request = Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=10) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, UnicodeDecodeError):
            self._digests[model] = digest
            return digest
        tags = payload.get("models", []) if isinstance(payload, dict) else []
        if isinstance(tags, list):
            for tag in tags:
                if not isinstance(tag, dict):
                    continue
                if model in {tag.get("name"), tag.get("model")}:
                    value = tag.get("digest")
                    if isinstance(value, str) and value:
                        digest = value
                    break
        self._digests[model] = digest
        return digest

    def complete(
        self, model: str, prompt: Mapping[str, str], temperature: float
    ) -> Completion:
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt["prompt"]}],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "seed": int(prompt.get("rep", "0")),
                    "num_predict": MAX_OUTPUT_TOKENS,
                },
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                payload = json.load(response)
        except HTTPError as error:
            raise ClientFailure(
                f"Ollama HTTP status {error.code}",
                error.code == 429 or error.code >= 500,
            ) from None
        except (URLError, TimeoutError, OSError):
            raise ClientFailure("Ollama connection failed", True) from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ClientFailure("Ollama returned invalid JSON", False) from None

        try:
            content = payload["message"]["content"]
            if not isinstance(content, str):
                raise TypeError
            input_tokens = _nonnegative_int(payload.get("prompt_eval_count"))
            output_tokens = _nonnegative_int(payload.get("eval_count"))
        except (KeyError, TypeError):
            raise ClientFailure("Ollama response is missing message content", False) from None
        return Completion(
            content,
            self._model_digest(model),
            input_tokens,
            output_tokens,
        )


class GeminiClient:
    """Google Gen AI client; requires explicit live mode and a GEMINI_API_KEY."""

    def __init__(
        self,
        rpm: int = 8,
        *,
        client: Any | None = None,
        types_module: Any | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        api_error_type: type[Exception] | tuple[type[Exception], ...] | None = None,
    ) -> None:
        if rpm < 1:
            raise ValueError("rpm must be at least 1")
        self._interval = 60.0 / rpm
        self._sleeper = sleeper
        self._clock = clock
        self._last_call: float | None = None
        self._api_error_type = api_error_type or Exception

        if client is not None and types_module is not None:
            self._client = client
            self._types = types_module
            return
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Gemini live mode requires GEMINI_API_KEY.")
        try:
            from google import genai
            from google.genai import types
            from google.genai.errors import APIError
        except ImportError:
            raise RuntimeError("Install google-genai for Gemini live mode.") from None
        self._client = client or genai.Client(api_key=api_key)
        self._types = types_module or types
        self._api_error_type = api_error_type or APIError

    def _wait_for_slot(self) -> None:
        now = self._clock()
        if self._last_call is not None:
            delay = self._interval - (now - self._last_call)
            if delay > 0:
                self._sleeper(delay)
                now = self._clock()
        self._last_call = now

    def _thinking_config(self, model: str) -> Any | None:
        config_type = getattr(self._types, "ThinkingConfig", None)
        if config_type is None:
            return None
        model_id = model.lower()
        if model_id.startswith("gemini-2.5-"):
            try:
                return config_type(thinking_budget=0)
            except (TypeError, ValueError):
                return None
        levels = (
            ("gemini-3.8-flash", "LOW"),
            ("gemini-3.7-flash", "LOW"),
            ("gemini-3.6-flash", "MINIMAL"),
            ("gemini-3.5-flash-lite", "MINIMAL"),
            ("gemini-3.5-flash", "MINIMAL"),
            ("gemini-3.1-flash-lite", "MINIMAL"),
            ("gemini-3.1-pro", "LOW"),
            ("gemini-3-flash", "MINIMAL"),
            ("gemini-3-pro", "LOW"),
        )
        for prefix, level in levels:
            if model_id.startswith(prefix):
                enum = getattr(self._types, "ThinkingLevel", None)
                value = getattr(enum, level, level.lower()) if enum else level.lower()
                try:
                    return config_type(thinking_level=value)
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _status_code(error: Exception) -> int | None:
        response = getattr(error, "response", None)
        for source in (error, response):
            if source is None:
                continue
            for name in ("code", "status_code"):
                value = getattr(source, name, None)
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
                if isinstance(value, str) and value.isdigit():
                    return int(value)
        return None

    def complete(
        self, model: str, prompt: Mapping[str, str], temperature: float
    ) -> Completion:
        self._wait_for_slot()
        config_values: dict[str, object] = {
            "temperature": temperature,
            "seed": int(prompt.get("rep", "0")),
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
        thinking_config = self._thinking_config(model)
        if thinking_config is not None:
            config_values["thinking_config"] = thinking_config
        config = self._types.GenerateContentConfig(**config_values)
        try:
            result = self._client.models.generate_content(
                model=model,
                contents=prompt["prompt"],
                config=config,
            )
        except self._api_error_type as error:
            status = self._status_code(error)
            retryable = status == 429 or (status is not None and status >= 500)
            reason = f"Gemini HTTP status {status}" if status else "Gemini request failed"
            raise ClientFailure(reason, retryable) from None

        usage = getattr(result, "usage_metadata", None)
        try:
            content = result.text
        except (AttributeError, ValueError):
            content = None
        return Completion(
            content if isinstance(content, str) else "",
            getattr(result, "model_version", None) or "unknown",
            _nonnegative_int(getattr(usage, "prompt_token_count", None)),
            _nonnegative_int(getattr(usage, "candidates_token_count", None)),
            _nonnegative_int(getattr(usage, "thoughts_token_count", None)),
        )



def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _key_from_record(record: Mapping[str, object]) -> Key:
    return (
        str(record["model"]),
        str(record["prompt_id"]),
        str(record["lang"]),
        int(record["rep"]),
    )


def _append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_responses(records: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda row: _key_from_record(row))
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=RESPONSE_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for record in ordered:
            writer.writerow({column: record[column] for column in RESPONSE_COLUMNS})


def load_prompts(path: Path = PROMPTS_PATH) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"prompt_id", "q_id", "lang", "prompt", "gold_answer"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Prompt CSV is missing required contract columns.")
        return list(reader)


def run_collection(
    prompts: Sequence[Mapping[str, str]],
    client: Client | None = None,
    models: Sequence[str] = DEFAULT_MODELS,
    *,
    repeats: int = 1,
    temperature: float = DEFAULT_TEMPERATURE,
    raw_dir: Path = RAW_DIR,
    responses_path: Path = RESPONSES_PATH,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[dict[str, object]]:
    """Collect missing observations; raw cache and attempt log are append-only."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    selected_client = client or FakeClient()
    cache_path = raw_dir / "responses.jsonl"
    attempt_path = raw_dir / "attempts.jsonl"
    cache: dict[Key, dict[str, object]] = {}
    for record in _read_jsonl(cache_path):
        if "response" in record:
            record.setdefault("usage_available", "true" if all(record.get(k) is not None for k in ("input_tokens", "output_tokens")) else "false")
            cache.setdefault(_key_from_record(record), record)

    client_type = "fixture" if isinstance(selected_client, FakeClient) else "live"
    for prompt in prompts:
        for model in models:
            for rep in range(repeats):
                key: Key = (model, prompt["prompt_id"], prompt["lang"], rep)
                if key in cache:
                    continue
                call_prompt = {**prompt, "rep": str(rep)}
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    called_at = _utc_now()
                    started = perf_counter()
                    try:
                        completion = selected_client.complete(
                            model, call_prompt, temperature
                        )
                    except ClientFailure as error:
                        latency = max(0.0, (perf_counter() - started) * 1000)
                        retry = error.retryable and attempt < MAX_ATTEMPTS
                        _append_jsonl(
                            attempt_path,
                            {
                                "model": model,
                                "prompt_id": prompt["prompt_id"],
                                "lang": prompt["lang"],
                                "rep": rep,
                                "q_id": prompt["q_id"],
                                "model_version": "unknown",
                                "called_at": called_at,
                                "temperature": temperature,
                                "input_tokens": None,
                                "output_tokens": None,
                                "thinking_tokens": None,
                                "usage_status": "unavailable",
                                "latency_ms": latency,
                                "outcome": "retry" if retry else "failed",
                                "attempt": attempt,
                                "client_type": client_type,
                                "error": str(error),
                            },
                        )
                        if retry:
                            delay = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                            sleeper(delay)
                            continue
                        break

                    latency = max(0.0, (perf_counter() - started) * 1000)
                    usage_known = (
                        completion.input_tokens is not None
                        and completion.output_tokens is not None
                        and completion.input_tokens >= 0
                        and completion.output_tokens >= 0
                    )
                    success = bool(completion.response.strip())
                    record: dict[str, object] = {
                        "model": model,
                        "prompt_id": prompt["prompt_id"],
                        "lang": prompt["lang"],
                        "rep": rep,
                        "q_id": prompt["q_id"],
                        "model_version": completion.model_version or "unknown",
                        "called_at": called_at,
                        "temperature": temperature,
                        "input_tokens": completion.input_tokens,
                        "output_tokens": completion.output_tokens,
                        "thinking_tokens": completion.thinking_tokens,
                        "usage_available": "true" if usage_known else "false",
                        "usage_status": "reported" if usage_known else "unavailable",
                        "latency_ms": latency,
                        "outcome": "success" if success else "failed",
                        "attempt": attempt,
                        "client_type": client_type,
                        "response": completion.response,
                    }
                    _append_jsonl(attempt_path, record)
                    if success:
                        _append_jsonl(cache_path, record)
                        cache[key] = record
                    break

    requested_models = set(models)
    requested_prompts = {(row["prompt_id"], row["lang"]) for row in prompts}
    responses = [
        record for (model, prompt_id, lang, rep), record in cache.items()
        if model in requested_models and (prompt_id, lang) in requested_prompts and rep < repeats
    ]
    _write_responses(responses, responses_path)
    return responses


def _models_for_live() -> tuple[str, str]:
    model_a = os.environ.get("TAX_MODEL_A", "").strip()
    model_b = os.environ.get("TAX_MODEL_B", "").strip()
    if not model_a or not model_b:
        raise ValueError("Live mode requires TAX_MODEL_A and TAX_MODEL_B environment variables.")
    return model_a, model_b


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROMPTS_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--provider",
        choices=("ollama", "gemini", "openai-compatible"),
        default="ollama",
    )
    parser.add_argument("--models", nargs="+", help="Model identifiers to collect")
    parser.add_argument("--rpm", type=int, default=8, help="Gemini requests per minute")
    args = parser.parse_args(argv)
    if args.live and args.provider != "openai-compatible" and not args.models:
        parser.error("--live requires --models for Ollama and Gemini.")
    if args.provider == "gemini" and args.rpm < 1:
        parser.error("--rpm must be at least 1.")
    if not args.live:
        client: Client = FakeClient()
        models = tuple(args.models or DEFAULT_MODELS)
    elif args.provider == "ollama":
        client = OllamaClient()
        models = tuple(args.models or ())
    elif args.provider == "gemini":
        client = GeminiClient(rpm=args.rpm)
        models = tuple(args.models or ())
    else:
        client = RealClient()
        models = tuple(args.models or _models_for_live())
    records = run_collection(
        load_prompts(args.input),
        client,
        models,
        repeats=args.repeats,
        temperature=args.temperature,
        raw_dir=args.raw_dir or (RAW_DIR if args.live else RAW_DIR / "fixture"),
        responses_path=args.output or (RESPONSES_PATH if args.live else DEMO_DIR / "responses_fixture.csv"),
    )
    print(f"Collected {len(records)} successful observations ({'live' if args.live else 'fixture'}).")


if __name__ == "__main__":
    main()
