"""Build and validate the frozen v2 trilingual prompt CSV."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from src.common.config import DATA_DIR, PROMPTS_PATH

DRAFT_PATH = DATA_DIR / "prompts_draft_old.csv"
BANNED_PATH = DATA_DIR / "banned_topics.txt"
LANGS = ("kk", "ru", "en")
ANSWER_TYPES = {"mcq", "number", "short"}
PROMPT_COLUMNS = (
    "prompt_id",
    "q_id",
    "lang",
    "category",
    "prompt",
    "answer_type",
    "gold_answer",
    "verified_by_native",
)
REQUIRED_COLUMNS = tuple(column for column in PROMPT_COLUMNS if column != "verified_by_native")
Row = Mapping[str, str]


def load_banned_terms(path: Path = BANNED_PATH) -> tuple[str, ...]:
    """Load one banned-topic keyword per line."""
    return tuple(
        line.strip().casefold()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _validate_gold(answer_type: str, gold: str, q_id: str) -> None:
    if answer_type == "mcq" and gold not in {"A", "B", "C", "D"}:
        raise ValueError(f"Question {q_id} requires a choice letter A-D.")
    if answer_type == "number":
        try:
            value = float(gold)
        except ValueError as error:
            raise ValueError(f"Question {q_id} has a non-numeric answer.") from error
        if not math.isfinite(value):
            raise ValueError(f"Question {q_id} needs a finite number.")


def validate_rows(
    rows: Iterable[Row], banned_terms: Iterable[str] = ()
) -> dict[str, object]:
    """Validate v2 rows, allowing localized short-text answers per the contract."""
    row_list = list(rows)
    if not row_list:
        raise ValueError("Prompt CSV has no data rows.")

    prompt_ids: set[str] = set()
    q_langs: set[tuple[str, str]] = set()
    groups: dict[str, list[Row]] = defaultdict(list)
    terms = tuple(term.casefold() for term in banned_terms)

    for row in row_list:
        missing = [
            column
            for column in REQUIRED_COLUMNS
            if not (row.get(column) or "").strip()
        ]
        if missing:
            raise ValueError(f"Row has empty required field(s): {', '.join(missing)}.")
        if row["lang"] not in LANGS:
            raise ValueError(f"Unsupported language {row['lang']!r}.")
        if row["answer_type"] not in ANSWER_TYPES:
            raise ValueError(f"Unsupported answer_type {row['answer_type']!r}.")
        if row["verified_by_native"] not in {"0", "1"}:
            raise ValueError("verified_by_native must be 0 or 1 in v2.")
        if row["prompt_id"] in prompt_ids:
            raise ValueError(f"Duplicate prompt_id {row['prompt_id']}.")
        prompt_ids.add(row["prompt_id"])

        key = (row["q_id"], row["lang"])
        if key in q_langs:
            raise ValueError(f"Duplicate q_id/language pair {key}.")
        q_langs.add(key)
        _validate_gold(row["answer_type"], row["gold_answer"], row["q_id"])
        prompt_text = row["prompt"].casefold()
        if any(term in prompt_text for term in terms):
            raise ValueError(f"Question {row['q_id']} contains a banned-topic keyword.")
        groups[row["q_id"]].append(row)

    categories: Counter[str] = Counter()
    for q_id, group in groups.items():
        if {row["lang"] for row in group} != set(LANGS):
            raise ValueError(f"Question {q_id} must have kk, ru and en rows.")
        if len({row["category"] for row in group}) != 1:
            raise ValueError(f"Question {q_id} has inconsistent categories.")
        if len({row["answer_type"] for row in group}) != 1:
            raise ValueError(f"Question {q_id} has inconsistent answer types.")
        answer_type = group[0]["answer_type"]
        if answer_type != "short" and len(
            {row["gold_answer"] for row in group}
        ) != 1:
            raise ValueError(f"Question {q_id} has inconsistent gold answers.")
        categories[group[0]["category"]] += 1

    verified_share = sum(row["verified_by_native"] == "1" for row in row_list) / len(
        row_list
    )
    return {
        "n_questions": len(groups),
        "n_rows": len(row_list),
        "per_category": dict(sorted(categories.items())),
        "verified_share": verified_share,
    }


def build_prompts(
    draft_path: Path = DRAFT_PATH, output_path: Path = PROMPTS_PATH
) -> None:
    """Expand the 60-row draft into the three localized v2 rows per question."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing prompts: {output_path}.")
    rows: list[dict[str, str]] = []
    with draft_path.open(encoding="utf-8-sig", newline="") as source:
        if source.readline().startswith("#"):
            pass
        else:
            source.seek(0)
        for draft in csv.DictReader(source):
            if draft["verified_by_native"] != "0":
                raise ValueError("Draft verified_by_native must stay 0 until native review.")
            for lang in LANGS:
                rows.append(
                    {
                        "prompt_id": f"{draft['q_id']}-{lang}",
                        "q_id": draft["q_id"],
                        "lang": lang,
                        "category": draft["category"],
                        "prompt": draft[f"prompt_{lang}"],
                        "answer_type": draft["answer_type"],
                        "gold_answer": draft[f"gold_{lang}"],
                        "verified_by_native": "0",
                    }
                )
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=PROMPT_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_csv(path: Path = PROMPTS_PATH) -> dict[str, object]:
    """Validate the exact v2 header and all prompt rows."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != PROMPT_COLUMNS:
            raise ValueError("prompts.csv header must exactly match SCHEMA v2.")
        summary = validate_rows(reader, load_banned_terms())
    if summary["n_questions"] != 60 or summary["n_rows"] != 180:
        raise ValueError("The prompt dataset must contain 60 questions and 180 rows.")
    categories = summary["per_category"]
    if len(categories) != 6 or any(count != 10 for count in categories.values()):
        raise ValueError("The prompt dataset must contain 10 questions in each of 6 categories.")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Build prompts.csv from the draft.")
    parser.add_argument("--input", type=Path, default=PROMPTS_PATH)
    args = parser.parse_args(argv)
    try:
        if args.build:
            build_prompts()
        summary = validate_csv(args.input)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Prompt validation failed: {error}\n")
    categories = ", ".join(
        f"{category}={count}"
        for category, count in summary["per_category"].items()
    )
    print(
        f"Validated {summary['n_questions']} questions / {summary['n_rows']} rows; "
        f"categories: {categories}; verified share: {summary['verified_share']:.3f}"
    )


if __name__ == "__main__":
    main()
