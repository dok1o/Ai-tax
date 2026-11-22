import csv

import pytest

from src import tokens


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
        tokens.OPEN_TOKENIZER_ID: FakeOpenTokenizer().encode,
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

    def from_pretrained(repo: str, revision: str) -> FakeOpenTokenizer:
        calls.append((repo, revision))
        return FakeOpenTokenizer()

    monkeypatch.setattr(tokens.tiktoken, "get_encoding", get_encoding)
    monkeypatch.setattr(
        tokens.AutoTokenizer, "from_pretrained", staticmethod(from_pretrained)
    )
    encoders = tokens.load_encoders()

    assert calls == [
        ("tiktoken", tokens.TIKTOKEN_ENCODING),
        (tokens.OPEN_TOKENIZER_REPO, tokens.OPEN_TOKENIZER_REVISION),
    ]
    text = "Қазақша  text!"
    assert len(encoders[tokens.TIKTOKEN_ENCODING](text)) == len(text)
    assert len(encoders[tokens.OPEN_TOKENIZER_ID](text)) == len(text.split())


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
