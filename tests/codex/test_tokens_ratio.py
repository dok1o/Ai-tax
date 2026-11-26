from src.tokens import build_ratio_rows


def test_tokens_vs_en_uses_same_question_and_tokenizer() -> None:
    rows = [
        {
            "prompt_id": "q1-kk",
            "q_id": "q1",
            "lang": "kk",
            "tokenizer": "test-tokenizer",
            "word_count": 2,
            "token_count": 6,
            "fertility": 3.0,
        },
        {
            "prompt_id": "q1-ru",
            "q_id": "q1",
            "lang": "ru",
            "tokenizer": "test-tokenizer",
            "word_count": 3,
            "token_count": 9,
            "fertility": 3.0,
        },
        {
            "prompt_id": "q1-en",
            "q_id": "q1",
            "lang": "en",
            "tokenizer": "test-tokenizer",
            "word_count": 4,
            "token_count": 12,
            "fertility": 3.0,
        },
    ]

    ratios = build_ratio_rows(rows)

    assert [row["tokens_vs_en"] for row in ratios] == [0.5, 0.75, 1.0]
