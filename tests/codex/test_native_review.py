"""Temporary-fixture tests for the native review worksheet and merge logic."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import pytest

from data.validate_prompts import PROMPT_COLUMNS
from src.native_review import (
    REVIEW_COLUMNS,
    apply_review,
    build_review_rows,
    write_review_csv,
)


def prompt_group(q_id: str, short_text: bool = False) -> list[dict[str, str]]:
    answers = {"kk": "солтүстік", "ru": "север", "en": "north"} if short_text else {
        "kk": "4",
        "ru": "4",
        "en": "4",
    }
    prompts = {"kk": "2 қосу 2 неше болады?", "ru": "Сколько будет 2 плюс 2?", "en": "What is 2 plus 2?"}
    return [
        {
            "prompt_id": f"{q_id}-{lang}",
            "q_id": q_id,
            "lang": lang,
            "category": "math_logic",
            "prompt": prompts[lang],
            "answer_type": "short_text" if short_text else "integer",
            "gold_answer": answers[lang],
            "verified_by_native": "false",
        }
        for lang in ("kk", "ru", "en")
    ]


def worksheet(q_id: str, **changes: str) -> dict[str, str]:
    row = {
        "q_id": q_id,
        "category": "math_logic",
        "prompt_kk": "2 қосу 2 неше болады?",
        "prompt_ru": "Сколько будет 2 плюс 2?",
        "gold_answer": "4",
        "ok": "",
        "correction": "",
    }
    row.update(changes)
    return row


def test_export_exact_header_join_and_blank_review_fields(tmp_path: Path) -> None:
    rows = build_review_rows(prompt_group("q1"))
    path = tmp_path / "native_review.csv"
    assert write_review_csv(rows, path) == 1
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        assert tuple(reader.fieldnames or ()) == REVIEW_COLUMNS
        exported = next(reader)
    assert exported["prompt_kk"] == "2 қосу 2 неше болады?"
    assert exported["prompt_ru"] == "Сколько будет 2 плюс 2?"
    assert exported["gold_answer"] == "4"
    assert exported["ok"] == exported["correction"] == ""


def test_export_refuses_to_overwrite_human_work(tmp_path: Path) -> None:
    path = tmp_path / "native_review.csv"
    path.write_text("completed worksheet", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_review_csv(build_review_rows(prompt_group("q1")), path)
    assert path.read_text(encoding="utf-8") == "completed worksheet"


def test_ok_correction_and_unmarked_rows_preserve_v1_data() -> None:
    prompts = prompt_group("q1") + prompt_group("q2", short_text=True) + prompt_group("q3")
    prompts[-3]["prompt"] = "Existing native correction"
    prompts[-3]["verified_by_native"] = "true"
    original = deepcopy(prompts)
    reviews = [
        worksheet("q1", ok="1"),
        worksheet(
            "q2",
            gold_answer="солтүстік бағыт",
            correction="Төрттен кейінгі санды атаңыз.",
        ),
        worksheet("q3", gold_answer="edited but unmarked"),
    ]

    candidate, counts = apply_review(prompts, reviews)
    rows = {row["prompt_id"]: row for row in candidate}
    assert rows["q1-kk"]["verified_by_native"] == "true"
    assert rows["q2-kk"]["prompt"] == "Төрттен кейінгі санды атаңыз."
    assert rows["q2-kk"]["gold_answer"] == "солтүстік бағыт"
    assert rows["q2-kk"]["verified_by_native"] == "true"
    assert rows["q3-kk"] == original[-3]
    assert [row["prompt_id"] for row in candidate] == [row["prompt_id"] for row in original]
    assert all(
        rows[f"{q_id}-{lang}"] == original[index * 3 + lang_index]
        for index, q_id in enumerate(("q1", "q2", "q3"))
        for lang_index, lang in enumerate(("kk", "ru", "en"))
        if q_id == "q3" or (q_id == "q1" and lang != "kk")
    )
    assert counts["reviewed"] == 2
    assert counts["corrected"] == 1


@pytest.mark.parametrize("reviews, message", [
    ([worksheet("q1"), worksheet("q1", ok="1")], "Duplicate review q_id"),
    ([worksheet("unknown", ok="1")], "Unknown review q_id"),
])
def test_duplicate_or_unknown_review_does_not_mutate_input(
    reviews: list[dict[str, str]], message: str
) -> None:
    prompts = prompt_group("q1")
    original = deepcopy(prompts)
    with pytest.raises(ValueError, match=message):
        apply_review(prompts, reviews)
    assert prompts == original


def test_review_only_updates_v1_boolean_and_exact_prompt_columns() -> None:
    candidate, _ = apply_review(prompt_group("q1"), [worksheet("q1", ok="1")])
    assert tuple(candidate[0]) == PROMPT_COLUMNS
    assert {row["verified_by_native"] for row in candidate} == {"false", "true"}
