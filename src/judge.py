"""Grade responses into the frozen v1 labels schema."""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.common.config import LABELS_PATH, PROMPTS_PATH, RESPONSES_PATH

LABEL_COLUMNS = (
    "model",
    "prompt_id",
    "lang",
    "rep",
    "q_id",
    "parsed_answer",
    "is_correct",
)
STATUS_COLUMNS = (
    "model",
    "prompt_id",
    "lang",
    "rep",
    "q_id",
    "status",
    "data_source",
)
STATUS_PATH = LABELS_PATH.with_name("label_status.csv")
NUMERIC_TOLERANCE = 0.01
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
CHOICE_PATTERN = re.compile(r"(?<!\w)([a-d])(?!\w)", re.IGNORECASE)
BOOLEAN_WORDS = {
    "true": True,
    "yes": True,
    "да": True,
    "иә": True,
    "ия": True,
    "false": False,
    "no": False,
    "нет": False,
    "жоқ": False,
}


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def extract_choice(value: str) -> str | None:
    choices = {match.upper() for match in CHOICE_PATTERN.findall(value)}
    return next(iter(choices)) if len(choices) == 1 else None


def _parse_number(value: str) -> float | None:
    match = NUMBER_PATTERN.search(value.replace("\u00a0", " "))
    if match is None:
        return None
    try:
        return float(match.group().replace(",", "."))
    except ValueError:
        return None


def _parse_boolean(value: str) -> bool | None:
    words = set(re.findall(r"[^\W_]+", normalize_text(value), flags=re.UNICODE))
    found = {BOOLEAN_WORDS[word] for word in words if word in BOOLEAN_WORDS}
    return next(iter(found)) if len(found) == 1 else None


def grade_answer(
    response: str, gold_answer: str, answer_type: str, lang: str = "en"
) -> tuple[str, bool, str]:
    """Return normalized answer, correctness and graded/unparsed status."""
    del lang  # Boolean aliases are accepted in all three project languages.
    if answer_type == "mcq":
        parsed = extract_choice(response)
        if parsed is None:
            return "", False, "unparsed"
        return parsed, parsed == normalize_text(gold_answer).upper(), "graded"

    if answer_type in {"integer", "decimal", "number"}:
        parsed_number = _parse_number(response)
        expected_number = _parse_number(gold_answer)
        if parsed_number is None or expected_number is None:
            return "", False, "unparsed"
        if answer_type == "integer" and abs(parsed_number - round(parsed_number)) > 1e-9:
            return "", False, "unparsed"
        parsed = str(int(parsed_number)) if parsed_number.is_integer() else str(parsed_number)
        return parsed, abs(parsed_number - expected_number) <= NUMERIC_TOLERANCE, "graded"

    if answer_type == "boolean":
        parsed_boolean = _parse_boolean(response)
        if parsed_boolean is None or gold_answer not in {"true", "false"}:
            return "", False, "unparsed"
        parsed = "true" if parsed_boolean else "false"
        return parsed, parsed == gold_answer, "graded"

    if answer_type in {"short", "short_text"}:
        parsed = normalize_text(response)
        expected = normalize_text(gold_answer)
        if not parsed:
            return "", False, "unparsed"
        return parsed, parsed == expected, "graded"

    return "", False, "unparsed"


def grade_rows(
    prompts: Sequence[Mapping[str, str]], responses: Sequence[Mapping[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    prompt_by_id = {prompt["prompt_id"]: prompt for prompt in prompts}
    if len(prompt_by_id) != len(prompts):
        raise ValueError("prompts contain duplicate prompt_id values")

    labels: list[dict[str, str]] = []
    statuses: list[dict[str, str]] = []
    for response in responses:
        prompt = prompt_by_id.get(response["prompt_id"])
        if prompt is None:
            raise ValueError(f"No prompt for response {response['prompt_id']}.")
        if prompt["q_id"] != response["q_id"] or prompt["lang"] != response["lang"]:
            raise ValueError(f"Response key mismatch for {response['prompt_id']}.")
        parsed, correct, status = grade_answer(
            response["response"],
            prompt["gold_answer"],
            prompt["answer_type"],
            prompt["lang"],
        )
        values = {
            "model": response["model"],
            "prompt_id": response["prompt_id"],
            "lang": response["lang"],
            "rep": str(int(response["rep"])),
            "q_id": response["q_id"],
        }
        labels.append(
            {
                **values,
                "parsed_answer": parsed,
                "is_correct": "true" if correct else "false",
            }
        )
        statuses.append(
            {
                **values,
                "status": status,
                "data_source": "fixture"
                if response["model"].startswith("fixture/")
                else "external",
            }
        )
    return labels, statuses


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _write_csv(
    rows: Sequence[Mapping[str, str]], columns: Sequence[str], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_judge(
    prompts_path: Path = PROMPTS_PATH,
    responses_path: Path = RESPONSES_PATH,
    labels_path: Path = LABELS_PATH,
    status_path: Path = STATUS_PATH,
) -> tuple[int, int]:
    prompts = _read_csv(prompts_path)
    responses = _read_csv(responses_path)
    labels, statuses = grade_rows(prompts, responses)
    _write_csv(labels, LABEL_COLUMNS, labels_path)
    _write_csv(statuses, STATUS_COLUMNS, status_path)
    return len(labels), sum(row["status"] == "unparsed" for row in statuses)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    parser.add_argument("--responses", type=Path, default=RESPONSES_PATH)
    parser.add_argument("--output", type=Path, default=LABELS_PATH)
    parser.add_argument("--status-output", type=Path, default=STATUS_PATH)
    args = parser.parse_args(argv)
    total, unparsed = run_judge(
        args.prompts, args.responses, args.output, args.status_output
    )
    print(f"Graded {total} responses; unparsed: {unparsed}.")


if __name__ == "__main__":
    main()
