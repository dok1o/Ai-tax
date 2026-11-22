import csv

import pytest

from src import tokens
from src.common.config import DEMO_DIR


class FakeTiktokenEncoding:
    def encode(
        self, text: str, disallowed_special: tuple[str, ...] = ()
    ) -> list[str]:
        assert disallowed_special == ()
        return list(text)


class FakeOpenTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def test_build_rows_covers_languages_whitespace_and_is_repeatable() -> None:
    prompts = [
        {
            "prompt_id": "kk-1",
            "q_id": "q1",
            "lang": "kk",
            "prompt": "Сәлем,\tәлем!",
        },
        {
            "prompt_id": "ru-1",
            "q_id": "q1",
            "lang": "ru",
            "prompt": "Привет,   мир!",
        },
        {
            "prompt_id": "en-1",
            "q_id": "q1",
            "lang": "en",
            "prompt": "Hello,\tworld!",
        },
    ]
    encoders = {
        tokens.TIKTOKEN_ENCODING: FakeTiktokenEncoding().encode,
        tokens.SUBWORD_TOKENIZER_ID: FakeOpenTokenizer().encode,
    }

    rows = tokens.build_token_rows(prompts, encoders)

    assert len(rows) == len(prompts) * len(encoders)
    assert [row["lang"] for row in rows] == ["kk", "kk", "ru", "ru", "en", "en"]
    pairs = (rows[i : i + 2] for i in range(0, len(rows), 2))
    for prompt, pair in zip(prompts, pairs):
        word_count = len(prompt["prompt"].split())
        expected_counts = [len(prompt["prompt"]), word_count]
        for row, expected_count in zip(pair, expected_counts):
            assert row["token_count"] == expected_count
            assert row["word_count"] == word_count
            assert row["fertility"] == expected_count / word_count
    assert tokens.build_token_rows(prompts, encoders) == rows


def test_load_encoders_pins_open_revision_and_matches_library_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    def get_encoding(name: str) -> FakeTiktokenEncoding:
        calls.append(("tiktoken", name))
        return FakeTiktokenEncoding()

    def from_pretrained(
        repo: str, revision: str, use_fast: bool
    ) -> FakeOpenTokenizer:
        calls.append((repo, (revision, use_fast)))
        return FakeOpenTokenizer()

    monkeypatch.setattr(tokens.tiktoken, "get_encoding", get_encoding)
    monkeypatch.setattr(
        tokens.AutoTokenizer, "from_pretrained", staticmethod(from_pretrained)
    )
    encoders = tokens.load_encoders()

    assert calls == [
        ("tiktoken", tokens.TIKTOKEN_ENCODING),
        (
            tokens.SUBWORD_TOKENIZER_REPO,
            (tokens.SUBWORD_TOKENIZER_REVISION, True),
        ),
    ]
    text = "Қазақша  text!"
    assert len(encoders[tokens.TIKTOKEN_ENCODING](text)) == len(text)
    assert len(encoders[tokens.SUBWORD_TOKENIZER_ID](text)) == len(text.split())


def test_run_writes_contract_order_and_rejects_empty_prompt(tmp_path) -> None:
    prompts_path = tmp_path / "prompts.csv"
    output_path = tmp_path / "tokens.csv"
    with prompts_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=("prompt_id", "q_id", "lang", "prompt")
        )
        writer.writeheader()
        writer.writerow(
            {
                "prompt_id": "kk-1",
                "q_id": "q1",
                "lang": "kk",
                "prompt": "Сәлем",
            }
        )
    encoders = {"test-encoder": lambda text: [text]}

    tokens.run(prompts_path, output_path, encoders)

    with output_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        assert reader.fieldnames == list(tokens.TOKEN_COLUMNS)
        assert len(list(reader)) == 1
    with prompts_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=("prompt_id", "q_id", "lang", "prompt")
        )
        writer.writeheader()
        writer.writerow(
            {
                "prompt_id": "empty",
                "q_id": "q2",
                "lang": "en",
                "prompt": " \t ",
            }
        )
    with pytest.raises(ValueError, match="must contain a word"):
        tokens.run(prompts_path, output_path, encoders)


def test_read_text_lang_csv_without_ids_generates_contract_ids(tmp_path) -> None:
    input_path = tmp_path / "phrases.csv"
    input_path.write_text(
        "lang,text,source\nkk,Сәлем әлем,sample\nen,Hello world,sample\n",
        encoding="utf-8",
    )

    prompts = tokens.read_prompts(input_path)

    assert prompts == [
        {
            "prompt_id": "input-0001",
            "q_id": "input-0001",
            "lang": "kk",
            "prompt": "Сәлем әлем",
        },
        {
            "prompt_id": "input-0002",
            "q_id": "input-0002",
            "lang": "en",
            "prompt": "Hello world",
        },
    ]


def test_demo_sample_has_12_trilingual_example_rows() -> None:
    sample_path = DEMO_DIR / "sample_phrases.csv"
    with sample_path.open(encoding="utf-8", newline="") as source:
        sample_rows = list(csv.DictReader(source))
    prompts = tokens.read_prompts(sample_path)
    question_ids = {row["q_id"] for row in prompts}

    assert len(sample_rows) == len(prompts) == 12
    assert all(row["is_example"] == "true" for row in sample_rows)
    assert {row["lang"] for row in prompts} == {"kk", "ru", "en"}
    assert question_ids == {"ex-01", "ex-02", "ex-03", "ex-04"}
    assert all(
        sum(row["q_id"] == q_id for row in prompts) == 3
        for q_id in question_ids
    )
