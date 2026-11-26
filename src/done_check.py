"""Read-only evidence checks for the release gates in DONE.md."""

from __future__ import annotations

import csv
import json
import math
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
        if not all(term in text for term in ("holm", "mcnemar", "question", "prespecified", "repetition", "bootstrap", "comparison family")):
            return "report must document prespecified q_id-paired exact McNemar, Holm, and bootstrap family"
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


def _fold_errors(rows: list[dict[str, str]]) -> list[str]:
    partitions: dict[str, dict[str, set[str]]] = {}
    training = {"train", "training"}
    evaluation = {"test", "eval", "evaluation", "valid", "validation", "heldout", "held-out"}
    for row in rows:
        split = row["split"].strip().casefold().replace("_", "-")
        label = "train" if split in training else "eval" if split in evaluation else "unknown"
        if label == "unknown":
            return [f"unknown fold split label {row['split']!r}"]
        fold = row.get("fold", "0")
        partitions.setdefault(fold, {}).setdefault(label, set()).add(row["q_id"])
    if not partitions:
        return ["fold assignments are missing"]
    errors: list[str] = []
    for fold, parts in partitions.items():
        if "train" not in parts or "eval" not in parts:
            errors.append(f"fold {fold} is missing train or evaluation rows")
        elif overlap := parts["train"] & parts["eval"]:
            errors.append(f"fold {fold} leaks q_ids: {', '.join(sorted(overlap)[:3])}")
    return errors


def _model_comparison_errors(
    rows: list[dict[str, str]], expected_metric: str
) -> list[str]:
    fields = {
        "metric",
        "n",
        "baseline",
        "model",
        "baseline_value",
        "model_value",
        "signed_difference",
    }
    comparisons = [row for row in rows if fields.issubset(row)]
    if not comparisons:
        return ["held-out comparison must show n, baseline, model and signed difference"]
    metric_aliases = {
        "mae": {"mae", "token_count_mae", "token-count-mae"},
        "brier": {"brier", "brier_score"},
    }[expected_metric]
    matches = [
        row
        for row in comparisons
        if row["metric"].casefold() in metric_aliases
    ]
    if not matches:
        return [f"held-out {expected_metric} comparison is missing"]
    errors: list[str] = []
    for row in matches:
        try:
            n = int(row["n"])
            values = [
                float(row["baseline_value"]),
                float(row["model_value"]),
                float(row["signed_difference"]),
            ]
            if n <= 0 or not all(math.isfinite(value) for value in values):
                raise ValueError
        except ValueError:
            errors.append(f"invalid {expected_metric} comparison row")
            continue
        baseline = row["baseline"].casefold()
        if "training" not in baseline:
            errors.append(f"{expected_metric} baseline is not identified as training-derived")
        if expected_metric == "mae" and "mean" not in baseline:
            errors.append("token MAE baseline must be the training per-language/tokenizer mean")
        if expected_metric == "brier" and not any(word in baseline for word in ("error", "rate")):
            errors.append("Brier baseline must be the training error rate")
        if not row["model"].strip():
            errors.append(f"{expected_metric} model name is missing")
    return errors


def _feature_manifest_errors(path: Path | None) -> list[str]:
    if path is None:
        return ["pre-call feature manifest is missing"]
    rows = _csv(path, {"feature", "availability"})
    if any(row["availability"].casefold().replace("_", "-") not in {"pre-call", "before-call"} for row in rows):
        return ["feature manifest contains a feature unavailable before the call"]
    return []


def _models_gate(root: Path) -> GateResult:
    paths = [root / "models" / "token_cost", root / "models" / "error_risk"]

    def check() -> str:
        for directory, metric in zip(paths, ("mae", "brier"), strict=True):
            if not directory.is_dir():
                return f"missing evidence directory: {_display_path(directory)}"
            files = [path for path in directory.rglob("*.csv") if path.is_file()]
            if not files:
                return f"no model CSV evidence under {_display_path(directory)}"
            csv_rows = [_csv(path) for path in files]
            folds = [
                row
                for rows in csv_rows
                for row in rows
                if {"q_id", "split"}.issubset(row)
            ]
            errors = _fold_errors(folds)
            if errors:
                return f"{errors[0]} under {_display_path(directory)}"
            manifest = next((path for path in files if path.name == "feature_manifest.csv"), None)
            errors = _feature_manifest_errors(manifest)
            if errors:
                return f"{errors[0]} under {_display_path(directory)}"
            errors = _model_comparison_errors(
                [row for rows in csv_rows for row in rows], metric
            )
            if errors:
                return f"{errors[0]} under {_display_path(directory)}"
        return ""

    return _read_failure("MODELS", paths, check)


def _router_gate(root: Path) -> GateResult:
    path = root / "models" / "router"
    gate = _artifact_gate("ROUTER", root, "models/router", ("observation", "decision", "curve", "saving"))
    if not gate.passed:
        return gate
    files = list(path.rglob("*.csv"))
    try:
        rows_by_path = {file: _csv(file) for file in files}
    except (OSError, ValueError) as error:
        return _result("ROUTER", path, str(error))
    observations = [
        row
        for rows in rows_by_path.values()
        for row in rows
        if {"q_id", "accuracy", "eligible"}.issubset(row)
        and ("policy" in row or "baseline" in row)
    ]
    observation_errors = _router_observation_errors(observations)
    if observation_errors:
        return _result("ROUTER", path, observation_errors[0])
    savings_files = [file for file in files if "saving" in file.name.casefold()]
    savings_rows = [row for file in savings_files for row in rows_by_path[file]]
    savings_errors = _router_savings_errors(savings_rows)
    if savings_errors:
        return _result("ROUTER", path, savings_errors[0])
    report_path = path / "REPORT.md"
    if not report_path.is_file():
        report_path = root / "RELEASE.md"
    try:
        report = report_path.read_text(encoding="utf-8").casefold()
    except OSError as error:
        return _result("ROUTER", report_path, f"router report unavailable: {error}")
    if "threshold" not in report or "training" not in report:
        return _result("ROUTER", report_path, "threshold selection on training questions is not documented")
    if "bootstrap" not in report or "q_id" not in report or "translation" not in report:
        return _result("ROUTER", report_path, "q_id-bootstrap and translation-cost evidence is missing")
    if any(float(row["ci_low"]) <= 0 <= float(row["ci_high"]) for row in savings_rows) and not any(
        term in report for term in ("includes zero", "crosses zero", "null result", "no savings")
    ):
        return _result("ROUTER", report_path, "CI includes zero but the null outcome is not reported")
    if any(row["accuracy_matched"].casefold() == "false" for row in savings_rows) and not any(
        term in report for term in ("cannot match accuracy", "accuracy not matched", "unmatched accuracy")
    ):
        return _result("ROUTER", report_path, "unmatched held-out accuracy outcome is not reported")
    return gate


def _policy_name(value: str) -> str | None:
    normalized = value.casefold().replace("-", "_").replace(" ", "_")
    if "always_direct" in normalized:
        return "always_direct"
    if "always_via_english" in normalized:
        return "always_via_english"
    if "router" in normalized or "frozen_policy" in normalized:
        return "router"
    return None


def _router_observation_errors(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["per-q_id router observations with eligibility and policy are missing"]
    required = {"q_id", "accuracy", "eligible", "translation_cost", "model_cost", "total_cost"}
    if any(not required.issubset(row) for row in rows):
        return ["router observations must include per-leg cost components and eligibility"]
    qids_by_policy: dict[str, set[str]] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        policy = _policy_name(row.get("policy") or row.get("baseline", ""))
        if policy is None:
            return [f"unknown router policy {row.get('policy', row.get('baseline', ''))!r}"]
        if row["eligible"].casefold() not in {"true", "false"}:
            return [f"invalid eligibility mark for q_id {row['q_id']}"]
        if row["eligible"].casefold() == "false":
            continue
        key = (policy, row["q_id"])
        if key in seen:
            return [f"duplicate eligible router result for {policy}/{row['q_id']}"]
        seen.add(key)
        qids_by_policy.setdefault(policy, set()).add(row["q_id"])
        try:
            accuracy = float(row["accuracy"])
            translation = float(row["translation_cost"])
            model = float(row["model_cost"])
            total = float(row["total_cost"])
            if not 0 <= accuracy <= 1 or min(translation, model, total) < 0:
                raise ValueError
            if not all(math.isfinite(value) for value in (accuracy, translation, model, total)):
                raise ValueError
            if not math.isclose(total, translation + model, rel_tol=1e-6, abs_tol=1e-9):
                return [f"total cost omits a translation/model leg for {policy}/{row['q_id']}"]
        except ValueError:
            return [f"invalid router accuracy/cost for {policy}/{row['q_id']}"]
    required_policies = {"router", "always_direct", "always_via_english"}
    if set(qids_by_policy) != required_policies:
        return ["router observations must include router, always_direct and always_via_english"]
    qid_sets = list(qids_by_policy.values())
    if any(qids != qid_sets[0] for qids in qid_sets[1:]):
        return ["router policies are not evaluated on identical eligible q_ids"]
    return []


def _router_savings_errors(rows: list[dict[str, str]]) -> list[str]:
    fields = {
        "baseline",
        "mean_cost_saved",
        "ci_low",
        "ci_high",
        "n_questions",
        "accuracy_matched",
        "bootstrap_unit",
    }
    if not rows or any(not fields.issubset(row) for row in rows):
        return ["savings table must report paired mean, q_id-bootstrap CI, n and accuracy match"]
    found: set[str] = set()
    for row in rows:
        baseline = _policy_name(row["baseline"])
        if baseline not in {"always_direct", "always_via_english"}:
            continue
        found.add(baseline)
        try:
            mean = float(row["mean_cost_saved"])
            low, high = float(row["ci_low"]), float(row["ci_high"])
            n = int(row["n_questions"])
            if not all(math.isfinite(value) for value in (mean, low, high)) or not low <= mean <= high or n <= 0:
                raise ValueError
        except ValueError:
            return [f"invalid paired savings/CI row for {baseline}"]
        if row["bootstrap_unit"].casefold() not in {"q_id", "qid", "question_id"}:
            return [f"savings CI for {baseline} is not bootstrapped by q_id"]
        if row["accuracy_matched"].casefold() not in {"true", "false"}:
            return [f"accuracy match outcome missing for {baseline}"]
    if found != {"always_direct", "always_via_english"}:
        return ["paired savings missing for always_direct or always_via_english"]
    return []


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
        for item in (*required_languages, "external networking disabled", "credentials unset", "interactive", "populated"):
            if item not in text:
                return f"RELEASE.md lacks offline app evidence: {item}"
        timing = re.search(r"cold start[^\n]{0,100}?(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b", text)
        if timing is None:
            return "RELEASE.md lacks a measured cold-start time"
        if float(timing.group(1)) >= 15:
            return f"A1 cold-start threshold is 15 seconds; measured {timing.group(1)} seconds"
        if "example data" not in text or "real-only" not in text:
            return "RELEASE.md lacks both fixture-banner and real-only smoke evidence"
        pending = _pending_confirmations(root, ("A1",))
        if pending:
            return "human confirmation pending for A1"
        return ""

    return _read_failure("APP", [demo, release], check)


def _repro_gate(root: Path) -> GateResult:
    path = root / "RELEASE.md"
    comparison_path = root / "notebooks" / "rebuild_comparison.csv"

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
        if missing:
            return f"RELEASE.md lacks rebuild evidence: {', '.join(missing)}"
        rows = _csv(
            comparison_path,
            {"artifact", "keys_match", "counts_match", "values_match", "input_version", "command"},
        )
        expected = {"measurement", "statistics", "token_cost", "error_risk", "router"}
        by_artifact = {row["artifact"].casefold(): row for row in rows}
        if not expected.issubset(by_artifact):
            return f"rebuild comparison lacks artifacts: {', '.join(sorted(expected - by_artifact.keys()))}"
        for artifact in expected:
            row = by_artifact[artifact]
            if not row["input_version"].strip() or not row["command"].strip():
                return f"rebuild comparison lacks command/input version for {artifact}"
            if any(row[column].casefold() != "true" for column in ("keys_match", "counts_match", "values_match")):
                return f"rebuild comparison differs from release tables for {artifact}"
        return ""

    return _read_failure("REPRODUCIBILITY", [path, comparison_path], check)


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
        if not re.search(r"https?://\S+", text) or not re.search(r"20\d{2}-\d{2}-\d{2}", text):
            return "citation evidence must include a source URL and opened date"
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
