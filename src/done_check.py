"""Read-only evidence checks for the release gates in DONE.md."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.config import ROOT

GATES = (
    "DATA",
    "MEASUREMENT",
    "STATISTICS",
    "MODELS",
    "ROUTER",
    "APP",
    "REPRODUCIBILITY",
    "INTEGRITY",
    "PRESENTATION",
    "MERGE",
)
PROMPT_FIELDS = {
    "prompt_id",
    "q_id",
    "lang",
    "category",
    "prompt",
    "answer_type",
    "gold_answer",
    "verified_by_native",
}
RESPONSE_FIELDS = {
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
}
LOG_FIELDS = {
    "model",
    "model_version",
    "called_at",
    "temperature",
    "input_tokens",
    "output_tokens",
    "latency_ms",
}
CALL_KEY_FIELDS = {"model", "prompt_id", "lang", "rep"}
LANGS = {"kk", "ru", "en"}


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    evidence: str
    detail: str


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _pending_confirmations(root: Path, thresholds: tuple[str, ...]) -> list[str]:
    release = root / "RELEASE.md"
    if not release.is_file():
        return list(thresholds)
    text = release.read_text(encoding="utf-8").casefold()
    return [
        item
        for item in thresholds
        if not re.search(rf"\b{item}\b[^\n]*(confirmed|approved|accepted|signed off)", text)
    ]


def _csv(path: Path, required: set[str] | None = None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("missing CSV header")
        missing = sorted((required or set()) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("no data rows")
    return rows


def _jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at line {line_number}") from error
            if not isinstance(item, dict):
                raise TypeError(f"line {line_number} is not a JSON object")
            entries.append(item)
    if not entries:
        raise ValueError("no JSONL entries")
    return entries


def _result(name: str, evidence: Path | str, detail: str = "") -> GateResult:
    if isinstance(evidence, Path):
        evidence = _display_path(evidence)
    return GateResult(name, not detail, str(evidence), detail or "all automated checks passed")


def _read_failure(name: str, paths: list[Path], check: Any) -> GateResult:
    evidence = ", ".join(_result(name, path).evidence for path in paths)
    try:
        detail = check()
    except OSError as error:
        detail = f"cannot read evidence: {_display_path(Path(error.filename))}" if error.filename else str(error)
    except (ValueError, KeyError, TypeError) as error:
        detail = str(error)
    return _result(name, evidence, detail or "")


def _prompts_gate(root: Path) -> GateResult:
    path = root / "data" / "prompts.csv"
    release = root / "RELEASE.md"

    def check() -> str:
        from data.validate_prompts import (
            PROMPT_COLUMNS,
            load_banned_terms,
            validate_rows,
        )

        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != PROMPT_COLUMNS:
                return "prompts.csv header differs from SCHEMA v1"
            rows = list(reader)
        summary = validate_rows(rows, load_banned_terms(root / "data" / "banned_topics.txt"))
        issues: list[str] = []
        if summary["n_questions"] < 60:
            issues.append(f"D1 requires 60 q_ids; found {summary['n_questions']}")
        categories = summary["per_category"]
        if len(categories) != 6 or any(count != 10 for count in categories.values()):
            issues.append("D1 requires 10 q_ids in each of six categories")
        kk_rows = [row for row in rows if row["lang"] == "kk"]
        verified = sum(row["verified_by_native"] == "true" for row in kk_rows)
        if len(kk_rows) < 60 or verified < len(kk_rows):
            issues.append(f"D2 requires all kk rows verified; found {verified}/{len(kk_rows)}")
        if not release.is_file():
            issues.append("RELEASE.md missing; D1/D2 human confirmation is pending")
        else:
            pending = _pending_confirmations(root, ("D1", "D2"))
            if pending:
                issues.append(f"human confirmation pending for {', '.join(pending)}")
        return "; ".join(issues)

    return _read_failure("DATA", [path, release], check)


def _provenance(records: list[dict[str, Any]]) -> str:
    kinds: set[str] = set()
    recorded = False
    markers = ("fixture", "example", "fake", "stub", "synthetic")
    for record in records:
        values = [str(record.get(key, "")) for key in ("provenance", "source", "client", "model")]
        marked = any(any(marker in value.casefold() for marker in markers) for value in values)
        source_values = [
            str(record.get(key, "")).strip().casefold()
            for key in ("provenance", "source", "client")
            if record.get(key)
        ]
        recorded = recorded or bool(source_values)
        if marked:
            kinds.add("fixture")
        elif any(value in {"real", "live", "provider", "api"} for value in source_values):
            kinds.add("real")
    if kinds == {"fixture"}:
        return "fixture-only observations are not release evidence"
    if kinds == {"real", "fixture"}:
        return "mixed real and fixture provenance; separate before release"
    if not kinds:
        return "provenance is not recorded in observation metadata"
    if not recorded:
        return "provenance is not recorded in observation metadata"
    return ""


def _measurement_errors(
    prompts: list[dict[str, str]],
    responses: list[dict[str, str]],
    labels: list[dict[str, str]],
    tokens: list[dict[str, str]],
    attempts: list[dict[str, Any]],
    raw_responses: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    prompt_keys = {(row["prompt_id"], row["lang"], row["q_id"]) for row in prompts}
    qids = {row["q_id"] for row in prompts}
    response_keys: set[tuple[str, str, str, str]] = set()
    cells: dict[tuple[str, str], dict[str, set[str]]] = {}
    for row in responses:
        key = (row["model"], row["prompt_id"], row["lang"], row["rep"])
        if key in response_keys:
            errors.append(f"duplicate response key {key}")
        response_keys.add(key)
        if (row["prompt_id"], row["lang"], row["q_id"]) not in prompt_keys:
            errors.append(f"response does not match prompts.csv: {key}")
        try:
            rep = int(row["rep"])
            if rep < 0:
                raise ValueError
        except ValueError:
            errors.append(f"invalid repetition for {key}")
        cell = cells.setdefault((row["q_id"], row["lang"]), {})
        cell.setdefault(row["model"], set()).add(row["rep"])
    for q_id in qids:
        for lang in LANGS:
            model_reps = cells.get((q_id, lang), {})
            if len(model_reps) < 2 or any(len(reps) < 2 for reps in model_reps.values()):
                errors.append(f"coverage below 2 models x 2 repeats: q_id={q_id}, lang={lang}")

    label_keys = {
        (row["model"], row["prompt_id"], row["lang"], row["rep"]): row
        for row in labels
    }
    if set(label_keys) != response_keys:
        errors.append("labels.csv keys do not match responses.csv")
    counts: dict[tuple[str, str], list[int]] = {}
    for key, row in label_keys.items():
        cell = counts.setdefault((key[0], key[2]), [0, 0])
        cell[1] += 1
        if not row["parsed_answer"].strip():
            cell[0] += 1
        if row["is_correct"] not in {"true", "false"}:
            errors.append(f"invalid is_correct for {key}")
    for (model, lang), (unparsed, total) in counts.items():
        if unparsed / total >= 0.05:
            errors.append(f"M1 unparsed rate is not <5% for {model}/{lang}: {unparsed}/{total}")

    token_prompt_ids = {row["prompt_id"] for row in tokens}
    if not {row["prompt_id"] for row in prompts}.issubset(token_prompt_ids):
        errors.append("tokens.csv does not cover every prompt_id")
    for row in tokens:
        try:
            word_count, token_count = int(row["word_count"]), int(row["token_count"])
            fertility = float(row["fertility"])
            if word_count <= 0 or token_count < 0 or abs(fertility - token_count / word_count) > 1e-6:
                raise ValueError
        except ValueError:
            errors.append(f"invalid token count/fertility for {row.get('prompt_id', '?')}")

    attempt_keys: set[tuple[str, str, str, str]] = set()
    attempt_status: dict[tuple[str, str, str, str], str] = {}
    for entry in attempts:
        missing = (LOG_FIELDS | CALL_KEY_FIELDS) - entry.keys()
        if missing:
            errors.append(f"attempt log missing metadata fields: {', '.join(sorted(missing))}")
            continue
        if "outcome" not in entry and "status" not in entry:
            errors.append("attempt log is missing outcome/status")
        attempt_key = tuple(str(entry[key]) for key in ("model", "prompt_id", "lang", "rep"))
        attempt_keys.add(attempt_key)
        attempt_status[attempt_key] = str(entry.get("outcome", entry.get("status", ""))).casefold()
        if not str(entry["called_at"]).endswith("Z"):
            errors.append(f"attempt timestamp is not UTC ISO-8601: {entry['model']}")
        if any(not str(entry[field]).strip() for field in LOG_FIELDS):
            errors.append(f"attempt metadata is blank: {entry['model']}")
    successful = {
        (str(row.get("model", "")), str(row.get("prompt_id", "")), str(row.get("lang", "")), str(row.get("rep", "")))
        for row in raw_responses
    }
    if not response_keys.issubset(successful):
        errors.append("responses.csv lacks a matching successful raw call")
    if not response_keys.issubset(attempt_keys):
        errors.append("responses.csv lacks a matching logged attempt")
    if any(attempt_status.get(key) not in {"success", "succeeded", "ok"} for key in response_keys):
        errors.append("responses.csv matches an attempt not marked successful")
    if successful - attempt_keys:
        errors.append("successful raw calls are missing from the attempt log")
    provenance_error = _provenance(responses + attempts + raw_responses)
    if provenance_error:
        errors.append(provenance_error)
    return errors


def _measurement_gate(root: Path) -> GateResult:
    paths = [
        root / "data" / "prompts.csv",
        root / "data" / "tokens.csv",
        root / "data" / "responses.csv",
        root / "data" / "labels.csv",
        root / "data" / "raw" / "attempts.jsonl",
        root / "data" / "raw" / "responses.jsonl",
    ]

    def check() -> str:
        for path in paths:
            if not path.is_file():
                missing = f"missing evidence: {_display_path(path)}"
                pending = _pending_confirmations(root, ("M1",))
                return "; ".join([missing, f"human confirmation pending for {', '.join(pending)}"] if pending else [missing])
        prompts = _csv(paths[0], PROMPT_FIELDS)
        responses = _csv(paths[2], RESPONSE_FIELDS)
        labels = _csv(paths[3], {"model", "prompt_id", "lang", "rep", "parsed_answer", "is_correct"})
        tokens = _csv(paths[1], {"prompt_id", "word_count", "token_count", "fertility"})
        attempts, raw = _jsonl(paths[4]), _jsonl(paths[5])
        errors = _measurement_errors(prompts, responses, labels, tokens, attempts, raw)
        return "; ".join(errors[:5]) if errors else ""

    return _read_failure("MEASUREMENT", paths, check)


def _table_gate(name: str, root: Path) -> GateResult:
    path = root / "data" / "metrics.csv"
    report = root / "notebooks" / "REPORT.md"

    def check() -> str:
        rows = _csv(path)
        columns = set(rows[0])
        required = {"model", "lang", "accuracy", "ci_low", "ci_high", "n_questions", "n_observations"}
        if not required.issubset(columns):
            return f"metrics.csv lacks statistical fields: {', '.join(sorted(required - columns))}"
        for row in rows:
            try:
                low, high, accuracy = float(row["ci_low"]), float(row["ci_high"]), float(row["accuracy"])
                n_q, n_obs = int(row["n_questions"]), int(row["n_observations"])
                if not (0 <= low <= accuracy <= high <= 1 and n_q > 0 and n_obs > 0):
                    raise ValueError
            except ValueError:
                return "invalid accuracy, CI, or sample size in metrics.csv"
        if not report.is_file():
            return "notebooks/REPORT.md missing comparison family, exact paired test and Holm evidence"
        text = report.read_text(encoding="utf-8").casefold()
        if "holm" not in text or "mcnemar" not in text or "question" not in text:
            return "report must document q_id-paired exact McNemar and Holm correction"
        if "95%" not in text:
            return "report must state 95% q_id-bootstrap confidence intervals"
        if "preliminary" not in text:
            return "report must identify small-sample estimates as preliminary"
        pending = _pending_confirmations(root, ("S1",))
        if pending:
            return "human confirmation pending for S1"
        return ""

    return _read_failure(name, [path, report], check)


def _artifact_gate(name: str, root: Path, folder: str, required: tuple[str, ...]) -> GateResult:
    directory = root / folder

    def check() -> str:
        if not directory.is_dir():
            return f"missing evidence directory: {_display_path(directory)}"
        files = [path for path in directory.rglob("*") if path.is_file()]
        if not files:
            return f"no saved evidence under {_display_path(directory)}"
        names = " ".join(path.name.casefold() for path in files)
        missing = [term for term in required if term not in names]
        if missing:
            return f"missing evidence files for: {', '.join(missing)}"
        for path in files:
            if path.suffix.lower() == ".csv":
                _csv(path)
        return ""

    return _read_failure(name, [directory], check)


def _models_gate(root: Path) -> GateResult:
    paths = [root / "models" / "token_cost", root / "models" / "error_risk"]

    def check() -> str:
        for directory in paths:
            if not directory.is_dir():
                return f"missing evidence directory: {_display_path(directory)}"
            files = [path for path in directory.rglob("*.csv") if path.is_file()]
            if not files:
                return f"no model CSV evidence under {_display_path(directory)}"
            folds = [row for path in files for row in _csv(path) if "q_id" in row and "split" in row]
            grouped: dict[str, dict[str, set[str]]] = {}
            for row in folds:
                grouped.setdefault(row.get("fold", "0"), {}).setdefault(row["split"].casefold(), set()).add(row["q_id"])
            if not grouped or any(train & test for parts in grouped.values() for train in [parts.get("train", set())] for test in [parts.get("test", parts.get("eval", set()))]):
                return f"grouped q_id train/evaluation split missing or leaking under {_display_path(directory)}"
            headers = [set(_csv(path)[0]) for path in files]
            all_columns = set.union(*headers)
            if not ({"baseline", "metric"} <= all_columns or {"baseline_value", "model_value"} <= all_columns):
                return f"training baseline/model comparison missing under {_display_path(directory)}"
        return ""

    return _read_failure("MODELS", paths, check)


def _router_gate(root: Path) -> GateResult:
    path = root / "models" / "router"
    gate = _artifact_gate("ROUTER", root, "models/router", ("observation", "decision", "curve", "saving"))
    if not gate.passed:
        return gate
    files = list(path.rglob("*.csv"))
    names = set().union(*(set(_csv(file)[0]) for file in files))
    required = {"q_id", "route", "cost", "accuracy"}
    if not required.issubset(names):
        return _result("ROUTER", path, f"router CSVs lack fields: {', '.join(sorted(required - names))}")
    return gate


def _release_text_gate(name: str, root: Path, checks: tuple[str, ...]) -> GateResult:
    path = root / "RELEASE.md"

    def check() -> str:
        text = path.read_text(encoding="utf-8").casefold()
        missing = [item for item in checks if item not in text]
        if missing:
            return f"RELEASE.md lacks recorded evidence: {', '.join(missing)}"
        return ""

    return _read_failure(name, [path], check)


def _app_gate(root: Path) -> GateResult:
    demo = root / "data" / "demo"
    release = root / "RELEASE.md"
    required_languages = ("kk", "ru", "en")

    def check() -> str:
        if not demo.is_dir() or not any(path.is_file() for path in demo.rglob("*")):
            return "saved demo inputs missing under data/demo"
        text = release.read_text(encoding="utf-8").casefold()
        for item in (*required_languages, "15 seconds", "external networking disabled", "credentials unset"):
            if item not in text:
                return f"RELEASE.md lacks offline app evidence: {item}"
        if "example data" not in text or "real-only" not in text:
            return "RELEASE.md lacks both fixture-banner and real-only smoke evidence"
        pending = _pending_confirmations(root, ("A1",))
        if pending:
            return "human confirmation pending for A1"
        return ""

    return _read_failure("APP", [demo, release], check)


def _repro_gate(root: Path) -> GateResult:
    path = root / "RELEASE.md"

    def check() -> str:
        text = path.read_text(encoding="utf-8").casefold()
        required = (
            "python 3.11",
            "python run.py check",
            "seed = 42",
            "rebuild",
            "input version",
            "keys, counts and values",
        )
        missing = [item for item in required if item not in text]
        return f"RELEASE.md lacks rebuild evidence: {', '.join(missing)}" if missing else ""

    return _read_failure("REPRODUCIBILITY", [path], check)


def _integrity_gate(root: Path, git: Any) -> GateResult:
    release = root / "RELEASE.md"
    banned = root / "data" / "banned_topics.txt"
    evidence = [release, banned]

    def check() -> str:
        text = release.read_text(encoding="utf-8").casefold()
        required = ("citation", "opened", "model version", "history", "secret scan", "provenance")
        missing = [item for item in required if item not in text]
        if missing:
            return f"RELEASE.md lacks integrity evidence: {', '.join(missing)}"
        env = git(["log", "--all", "--full-history", "--format=%H", "--", ".env", ":(glob)**/.env"], root)
        if env.returncode != 0:
            return "git history scan for .env failed"
        if env.stdout.strip():
            return "tracked .env exists in reachable Git history"
        commits = git(["rev-list", "--all"], root)
        if commits.returncode != 0:
            return "cannot enumerate reachable Git history for secret scan"
        pattern = r"(sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})"
        for commit in commits.stdout.splitlines():
            scan = git(["grep", "-I", "-E", pattern, commit, "--"], root)
            if scan.returncode == 0:
                return f"possible credential signature found in reachable commit {commit}; line redacted"
            if scan.returncode not in (1,):
                return "Git secret scan encountered an error"
        return ""

    return _read_failure("INTEGRITY", evidence, check)


def _presentation_gate(root: Path) -> GateResult:
    path = root / "RELEASE.md"

    def check() -> str:
        text = path.read_text(encoding="utf-8").casefold()
        required = ("claim-to-file", "rehearsal", "judge q&a", "human sign-off")
        missing = [item for item in required if item not in text]
        return f"human presentation evidence pending: {', '.join(missing)}" if missing else ""

    return _read_failure("PRESENTATION", [path], check)


def _git(args: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)


def _merge_gate(root: Path) -> GateResult:
    def check() -> str:
        branches = _git(["branch", "--list", "main", "dev"], root)
        if branches.returncode != 0 or not {"main", "dev"}.issubset(set(branches.stdout.split())):
            return "main/dev refs missing; human merge and reviewed-tip evidence pending"
        tags = _git(["tag", "--list"], root)
        if tags.returncode != 0 or not tags.stdout.strip():
            return "release tag missing; human merge/sign-off pending"
        for tag in tags.stdout.splitlines():
            ancestor = _git(["merge-base", "--is-ancestor", "dev", tag], root)
            released = _git(["merge-base", "--is-ancestor", tag, "main"], root)
            if ancestor.returncode == 0 and released.returncode == 0:
                return ""
        return "no release tag contains the reviewed dev tip"

    return _read_failure("MERGE", [root / ".git"], check)


def evaluate(root: Path = ROOT) -> list[GateResult]:
    """Return all release gates without modifying the worktree or Git refs."""
    return [
        _prompts_gate(root),
        _measurement_gate(root),
        _table_gate("STATISTICS", root),
        _models_gate(root),
        _router_gate(root),
        _app_gate(root),
        _repro_gate(root),
        _integrity_gate(root, _git),
        _presentation_gate(root),
        _merge_gate(root),
    ]


def main() -> int:
    results = evaluate()
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.name} | {status} | {result.evidence} | {result.detail}")
    return int(any(not result.passed for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
