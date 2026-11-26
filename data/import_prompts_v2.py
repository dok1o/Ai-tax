"""Convert the proposed multilingual dataset into the frozen SCHEMA v2 format."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from data.validate_prompts import (
    PROMPT_COLUMNS,
    load_banned_terms,
    validate_csv,
    validate_rows,
)
from src.common.config import DATA_DIR, PROMPTS_PATH

SOURCE_PATH = DATA_DIR / "incoming" / "prompts_v2_proposed.csv"
SOURCE_COLUMNS = (
    "q_id",
    "category",
    "lang",
    "prompt",
    "gold_answer",
    "answer_type",
    "verified_by_native",
)
CATEGORY_MAP = {
    "Математика и логика": "math_logic",
    "Повседневные ситуации": "everyday_situations",
    "Здоровье и образование (базовое)": "health_education_basics",
    "Казахстан: история и культура": "kazakhstan_history_culture",
    "Наука": "science",
    "География и история мира": "world_geography_history",
}
Row = Mapping[str, str]


def read_source(path: Path = SOURCE_PATH) -> list[dict[str, str]]:
    """Read the source CSV and require its declared columns and flag values."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError("Source header does not match the T-014 input columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Source dataset has no rows")
    if any(row["verified_by_native"] != "0" for row in rows):
        raise ValueError("T-014 source verification flags must remain 0")
    return rows


def convert_rows(rows: Iterable[Row]) -> list[dict[str, str]]:
    """Map category labels and add localized prompt IDs without changing values."""
    converted: list[dict[str, str]] = []
    for row in rows:
        try:
            category = CATEGORY_MAP[row["category"]]
        except KeyError as error:
            raise ValueError(f"Unmapped source category: {row.get('category', '')}") from error
        if row["verified_by_native"] != "0":
            raise ValueError("T-014 must never promote native verification flags")
        converted.append(
            {
                "prompt_id": f"{row['q_id']}-{row['lang']}",
                "q_id": row["q_id"],
                "lang": row["lang"],
                "category": category,
                "prompt": row["prompt"],
                "answer_type": row["answer_type"],
                "gold_answer": row["gold_answer"],
                "verified_by_native": row["verified_by_native"],
            }
        )
    if not converted:
        raise ValueError("No source rows to convert")
    return converted


def import_dataset(
    input_path: Path = SOURCE_PATH,
    output_path: Path = PROMPTS_PATH,
) -> dict[str, object]:
    """Validate a complete candidate before atomically replacing prompts.csv."""
    candidate = convert_rows(read_source(input_path))
    validate_rows(candidate, load_banned_terms())
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
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
        summary = validate_csv(temporary_path)
        os.replace(temporary_path, output_path)
        return summary
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=PROMPTS_PATH)
    args = parser.parse_args()
    try:
        summary = import_dataset(args.input, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Dataset import failed: {error}\n")
    print(
        f"Imported {summary['n_questions']} questions / {summary['n_rows']} rows; "
        f"verified share: {summary['verified_share']:.3f}"
    )


if __name__ == "__main__":
    main()
