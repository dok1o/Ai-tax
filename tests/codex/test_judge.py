from pathlib import Path

import pytest

from src.judge import grade_answer, grade_rows


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("C", "C"), ("(d).", "D"), ("The answer is B.", "B")],
)
def test_extracts_choice_letter_variants(answer: str, expected: str) -> None:
    parsed, correct, status = grade_answer(answer, expected, "mcq")
    assert (parsed, correct, status) == (expected, True, "graded")


def test_numbers_allow_tolerance_and_units() -> None:
    parsed, correct, status = grade_answer("3.141 meters", "3.14", "decimal")
    assert (parsed, correct, status) == ("3.141", True, "graded")
    assert grade_answer("5 apples", "5", "integer")[1] is True


def test_boolean_answers_accept_russian_and_kazakh() -> None:
    assert grade_answer("Да, верно", "true", "boolean", "ru")[1] is True
    assert grade_answer("жоқ", "false", "boolean", "kk")[1] is True


def test_short_text_uses_normalized_exact_match() -> None:
    parsed, correct, status = grade_answer("  H₂O! ", "H2O", "short_text")
    assert (parsed, correct, status) == ("h2o", True, "graded")


def test_unparsed_answer_is_empty_in_v1_and_marked_in_extra_status() -> None:
    prompts = [
        {
            "prompt_id": "q1-en",
            "q_id": "q1",
            "lang": "en",
            "answer_type": "integer",
            "gold_answer": "4",
        }
    ]
    responses = [
        {
            "model": "fixture/model-a",
            "prompt_id": "q1-en",
            "lang": "en",
            "rep": "0",
            "q_id": "q1",
            "response": "I do not know",
        }
    ]

    labels, statuses = grade_rows(prompts, responses)

    assert labels[0]["parsed_answer"] == ""
    assert labels[0]["is_correct"] == "false"
    assert statuses[0]["status"] == "unparsed"
    assert statuses[0]["data_source"] == "fixture"


def test_labels_follow_v1_columns_and_mismatched_join_is_rejected(tmp_path: Path) -> None:
    prompts = [
        {
            "prompt_id": "q1-en",
            "q_id": "q1",
            "lang": "en",
            "answer_type": "integer",
            "gold_answer": "4",
        }
    ]
    responses = [
        {
            "model": "fixture/model-a",
            "prompt_id": "q1-en",
            "lang": "en",
            "rep": "0",
            "q_id": "q1",
            "response": "4",
        }
    ]
    labels, _ = grade_rows(prompts, responses)
    assert list(labels[0]) == [
        "model", "prompt_id", "lang", "rep", "q_id", "parsed_answer", "is_correct"
    ]
    responses[0]["q_id"] = "other"
    with pytest.raises(ValueError, match="key mismatch"):
        grade_rows(prompts, responses)
