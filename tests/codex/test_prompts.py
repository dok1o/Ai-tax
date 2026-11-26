from copy import deepcopy

import pytest

from data.validate_prompts import PROMPT_COLUMNS, build_prompts, main, validate_rows


def valid_rows() -> list[dict[str, str]]:
    return [
        {
            "prompt_id": "q1-kk",
            "q_id": "q1",
            "lang": "kk",
            "category": "math_logic",
            "prompt": "2 қосу 2 неше болады?",
            "answer_type": "number",
            "gold_answer": "4",
            "verified_by_native": "0",
        },
        {
            "prompt_id": "q1-ru",
            "q_id": "q1",
            "lang": "ru",
            "category": "math_logic",
            "prompt": "Сколько будет 2 плюс 2?",
            "answer_type": "number",
            "gold_answer": "4",
            "verified_by_native": "0",
        },
        {
            "prompt_id": "q1-en",
            "q_id": "q1",
            "lang": "en",
            "category": "math_logic",
            "prompt": "What is 2 plus 2?",
            "answer_type": "number",
            "gold_answer": "4",
            "verified_by_native": "0",
        },
    ]


def test_accepts_valid_trilingual_fixture() -> None:
    summary = validate_rows(valid_rows())
    assert summary["n_questions"] == 1
    assert summary["n_rows"] == 3
    assert summary["verified_share"] == 0


def test_rejects_missing_language() -> None:
    with pytest.raises(ValueError, match="must have kk, ru and en"):
        validate_rows(valid_rows()[:2])


def test_rejects_inconsistent_category() -> None:
    rows = valid_rows()
    rows[1]["category"] = "science"
    with pytest.raises(ValueError, match="inconsistent categories"):
        validate_rows(rows)


def test_rejects_inconsistent_gold_answer() -> None:
    rows = valid_rows()
    rows[1]["gold_answer"] = "5"
    with pytest.raises(ValueError, match="inconsistent gold answers"):
        validate_rows(rows)


def test_rejects_answer_type_outside_v2() -> None:
    rows = valid_rows()
    rows[0]["answer_type"] = "unsupported"
    with pytest.raises(ValueError, match="Unsupported answer_type"):
        validate_rows(rows)


def test_rejects_unparseable_integer_gold() -> None:
    rows = valid_rows()
    rows[0]["gold_answer"] = "four"
    with pytest.raises(ValueError, match="non-numeric"):
        validate_rows(rows)


def test_rejects_empty_required_cell() -> None:
    rows = valid_rows()
    rows[0]["prompt"] = ""
    with pytest.raises(ValueError, match="empty required field"):
        validate_rows(rows)


def test_rejects_duplicate_prompt_id() -> None:
    rows = valid_rows()
    rows[1]["prompt_id"] = rows[0]["prompt_id"]
    with pytest.raises(ValueError, match="Duplicate prompt_id"):
        validate_rows(rows)


def test_rejects_duplicate_question_language_pair() -> None:
    rows = valid_rows()
    duplicate = deepcopy(rows[0])
    duplicate["prompt_id"] = "q1-kk-copy"
    rows.append(duplicate)
    with pytest.raises(ValueError, match="Duplicate q_id/language"):
        validate_rows(rows)


def test_rejects_banned_topic_keyword() -> None:
    rows = valid_rows()
    rows[2]["prompt"] = "Who is the president?"
    with pytest.raises(ValueError, match="banned-topic"):
        validate_rows(rows, ("president",))


def test_cli_returns_nonzero_for_invalid_csv(tmp_path) -> None:
    input_path = tmp_path / "invalid.csv"
    input_path.write_text(",".join(PROMPT_COLUMNS) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main(["--input", str(input_path)])

    assert error.value.code == 1


def test_builder_refuses_to_overwrite_existing_native_marks(tmp_path) -> None:
    output_path = tmp_path / "prompts.csv"
    original = b"existing native corrections"
    output_path.write_bytes(original)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_prompts(tmp_path / "missing-draft.csv", output_path)

    assert output_path.read_bytes() == original
