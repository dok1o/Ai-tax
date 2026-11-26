"""Export a Kazakh review worksheet and provide a human-run v1 import."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from data.validate_prompts import (
    PROMPT_COLUMNS,
    load_banned_terms,
    validate_csv,
    validate_rows,
)
from src.common.config import DATA_DIR, PROMPTS_PATH

REVIEW_PATH = DATA_DIR / "native_review.csv"
REVIEW_COLUMNS = (
    "q_id",
    "category",
    "prompt_kk",
    "prompt_ru",
    "gold_answer",
    "ok",
    "correction",
)
Row = Mapping[str, str]


def read_prompt_rows(path: Path = PROMPTS_PATH) -> list[dict[str, str]]:
    """Read and validate a frozen v1 prompt CSV."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != PROMPT_COLUMNS:
            raise ValueError("prompts.csv header must exactly match SCHEMA v1")
        rows = list(reader)
    validate_rows(rows, load_banned_terms())
    return rows


def build_review_rows(prompts: Iterable[Row]) -> list[dict[str, str]]:
    """Join Kazakh and Russian prompts with one Kazakh gold per question."""
    grouped: dict[str, dict[str, Row]] = defaultdict(dict)
    order: list[str] = []
    for row in prompts:
        q_id = row["q_id"]
        if q_id not in grouped:
            order.append(q_id)
        lang = row["lang"]
        if lang in grouped[q_id]:
            raise ValueError(f"Duplicate {lang} row for q_id {q_id}")
        grouped[q_id][lang] = row

    result: list[dict[str, str]] = []
    for q_id in order:
        languages = grouped[q_id]
        if not {"kk", "ru", "en"}.issubset(languages):
            raise ValueError(f"Question {q_id} is missing a localized prompt")
        kk, ru = languages["kk"], languages["ru"]
        if kk["category"] != ru["category"]:
            raise ValueError(f"Question {q_id} has inconsistent categories")
        result.append(
            {
                "q_id": q_id,
                "category": kk["category"],
                "prompt_kk": kk["prompt"],
                "prompt_ru": ru["prompt"],
                "gold_answer": kk["gold_answer"],
                "ok": "",
                "correction": "",
            }
        )
    if not result:
        raise ValueError("No prompts to review")
    return result


def write_review_csv(rows: Iterable[Row], path: Path = REVIEW_PATH) -> int:
    """Write the extra review worksheet without changing a v1 CSV."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing review worksheet: {path}")
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row_list)
    return len(row_list)


def read_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != REVIEW_COLUMNS:
            raise ValueError("Review worksheet header does not match the required columns")
        return list(reader)


def apply_review(
    prompts: Iterable[Row],
    reviews: Iterable[Row],
    banned_terms: Iterable[str] = (),
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Apply validated marks to a copy, preserving all unmarked rows verbatim."""
    original_rows = [dict(row) for row in prompts]
    candidate = [dict(row) for row in original_rows]
    review_rows = [dict(row) for row in reviews]
    by_qid: dict[str, dict[str, str]] = {}
    for review in review_rows:
        missing = set(REVIEW_COLUMNS) - review.keys()
        if missing:
            raise ValueError(f"Review row is missing fields: {', '.join(sorted(missing))}")
        q_id = review["q_id"]
        if q_id in by_qid:
            raise ValueError(f"Duplicate review q_id {q_id}")
        by_qid[q_id] = review
        if review["ok"].strip() not in {"", "0", "1"}:
            raise ValueError(f"Invalid ok mark for q_id {q_id}; use 1, 0 or blank")

    qids = {row["q_id"] for row in candidate}
    unknown = sorted(set(by_qid) - qids)
    if unknown:
        raise ValueError(f"Unknown review q_id(s): {', '.join(unknown)}")

    kk_by_qid = {row["q_id"]: row for row in candidate if row["lang"] == "kk"}
    reviewed = 0
    corrected = 0
    for q_id, review in by_qid.items():
        correction = review["correction"].strip()
        marked = review["ok"].strip() == "1" or bool(correction)
        if not marked:
            continue
        if not review["gold_answer"].strip():
            raise ValueError(f"Marked review row has empty gold_answer for q_id {q_id}")
        row = kk_by_qid.get(q_id)
        if row is None:
            raise ValueError(f"No Kazakh prompt row for q_id {q_id}")
        reviewed += 1
        before = (row["prompt"], row["gold_answer"])
        if correction:
            row["prompt"] = correction
        row["gold_answer"] = review["gold_answer"]
        row["verified_by_native"] = "true"
        if before != (row["prompt"], row["gold_answer"]):
            corrected += 1

    validate_rows(candidate, banned_terms)
    counts = {
        "reviewed": reviewed,
        "corrected": corrected,
        "unchanged": len(candidate) - corrected,
    }
    return candidate, counts


def import_review_file(path: Path, prompts_path: Path = PROMPTS_PATH) -> dict[str, int]:
    """HUMAN-RUN ONLY: validate a candidate, then atomically replace prompts.csv."""
    prompts = read_prompt_rows(prompts_path)
    reviews = read_review_rows(path)
    candidate, counts = apply_review(prompts, reviews, load_banned_terms())
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=prompts_path.parent,
            suffix=".csv.tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(
                temporary,
                fieldnames=PROMPT_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(candidate)
        validate_csv(temporary_path)
        os.replace(temporary_path, prompts_path)
        validate_csv(prompts_path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("export", help="Write the Kazakh review worksheet")
    import_parser = actions.add_parser("import", help="HUMAN-RUN ONLY: apply a reviewed worksheet")
    import_parser.add_argument("file", type=Path, help="Completed native_review.csv")
    args = parser.parse_args(argv)
    try:
        if args.action == "export":
            rows = build_review_rows(read_prompt_rows())
            count = write_review_csv(rows)
            print(f"Wrote {count} review rows to {REVIEW_PATH}")
        else:
            counts = import_review_file(args.file)
            validate_csv(PROMPTS_PATH)
            print(", ".join(f"{key}={value}" for key, value in counts.items()))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Native review failed: {error}\n")


if __name__ == "__main__":
    main()
