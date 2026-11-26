"""Verify the imported SCHEMA v2 dataset and Kazakh UTF-8 round trip."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from data.import_prompts_v2 import CATEGORY_MAP, SOURCE_PATH
from data.validate_prompts import DRAFT_PATH, PROMPT_COLUMNS
from src.common.config import PROMPTS_PATH

EXPECTED_LANGS = {"kk", "ru", "en"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        assert tuple(reader.fieldnames or ()) == PROMPT_COLUMNS
        return list(reader)


def test_imported_dataset_has_sixty_complete_trilingual_questions() -> None:
    assert not PROMPTS_PATH.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = _read_rows(PROMPTS_PATH)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["q_id"]].append(row)
    assert len(rows) == 180
    assert len(groups) == 60
    assert all({row["lang"] for row in group} == EXPECTED_LANGS for group in groups.values())
    assert all(len({row["gold_answer"] for row in group}) == 1 for group in groups.values())
    assert all(row["prompt_id"] == f"{row['q_id']}-{row['lang']}" for row in rows)


def test_source_flags_and_category_mapping_are_preserved() -> None:
    with SOURCE_PATH.open(encoding="utf-8-sig", newline="") as source:
        source_rows = list(csv.DictReader(source))
    output_rows = _read_rows(PROMPTS_PATH)
    source_by_key = {(row["q_id"], row["lang"]): row for row in source_rows}
    for row in output_rows:
        source = source_by_key[(row["q_id"], row["lang"])]
        assert row["verified_by_native"] == source["verified_by_native"] == "0"
        assert row["category"] == CATEGORY_MAP[source["category"]]
        assert row["answer_type"] == source["answer_type"]
        assert row["gold_answer"] == source["gold_answer"]
    assert {row["verified_by_native"] for row in output_rows} == {"0"}


def test_validator_draft_default_tracks_archived_filename() -> None:
    assert DRAFT_PATH.name == "prompts_draft_old.csv"


def test_russian_category_labels_map_to_project_slugs() -> None:
    assert set(CATEGORY_MAP.values()) == {
        "math_logic",
        "everyday_situations",
        "health_education_basics",
        "kazakhstan_history_culture",
        "science",
        "world_geography_history",
    }


def test_kazakh_special_letters_survive_utf8_csv_round_trip(tmp_path: Path) -> None:
    sentence = "ә ғ қ ң ө ұ ү һ і"
    path = tmp_path / "kazakh.csv"
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=("text",), lineterminator="\n")
        writer.writeheader()
        writer.writerow({"text": sentence})
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8", newline="") as source:
        assert next(csv.DictReader(source))["text"] == sentence
