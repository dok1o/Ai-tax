"""Count prompt tokens with one named API encoding and one pinned open tokenizer."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import tiktoken
from transformers import AutoTokenizer

from src.common.config import PROMPTS_PATH, TOKENS_PATH

TIKTOKEN_ENCODING = "cl100k_base"
SUBWORD_TOKENIZER_REPO = "Qwen/Qwen2.5-0.5B"
SUBWORD_TOKENIZER_REVISION = "5796c1d2033cb6257f44cdffa28957680359236d"
SUBWORD_TOKENIZER_ID = f"{SUBWORD_TOKENIZER_REPO}@{SUBWORD_TOKENIZER_REVISION}"
SUPPORTED_LANGS = frozenset({"kk", "ru", "en"})
TOKEN_COLUMNS = (
    "prompt_id",
    "q_id",
    "lang",
    "tokenizer",
    "word_count",
    "token_count",
    "fertility",
)
Encoder = Callable[[str], Sequence[object]]


def load_encoders() -> dict[str, Encoder]:
    """Load the configured tokenizers and return their no-special-token encoders."""
    encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    subword_tokenizer = AutoTokenizer.from_pretrained(
        SUBWORD_TOKENIZER_REPO,
        revision=SUBWORD_TOKENIZER_REVISION,
        use_fast=True,
    )
    return {
        TIKTOKEN_ENCODING: lambda text: encoding.encode(
            text, disallowed_special=()
        ),
        SUBWORD_TOKENIZER_ID: lambda text: subword_tokenizer.encode(
            text, add_special_tokens=False
        ),
    }


def build_token_rows(
    prompts: Iterable[Mapping[str, str]], encoders: Mapping[str, Encoder]
) -> list[dict[str, str | int | float]]:
    """Build schema-ordered metrics, counting words by whitespace splitting."""
    rows: list[dict[str, str | int | float]] = []
    for prompt in prompts:
        text = prompt["prompt"]
        word_count = len(text.split())
        if word_count == 0:
            raise ValueError(f"Prompt {prompt['prompt_id']} must contain a word.")
        for tokenizer_id, encode in encoders.items():
            token_count = len(encode(text))
            rows.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "q_id": prompt["q_id"],
                    "lang": prompt["lang"],
                    "tokenizer": tokenizer_id,
                    "word_count": word_count,
                    "token_count": token_count,
                    "fertility": token_count / word_count,
                }
            )
    return rows


def read_prompts(path: Path = PROMPTS_PATH) -> list[dict[str, str]]:
    """Read any CSV with text and lang, or normalize contract prompt columns."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or ())
        text_column = "text" if "text" in columns else "prompt"
        if text_column not in columns or "lang" not in columns:
            raise ValueError("Input CSV must contain 'text' and 'lang' columns.")

        prompts = []
        for row_number, row in enumerate(reader, start=1):
            prompt_id = (row.get("prompt_id") or "").strip()
            q_id = (row.get("q_id") or "").strip()
            lang = (row.get("lang") or "").strip()
            if not prompt_id:
                prompt_id = f"input-{row_number:04d}"
            if not q_id:
                q_id = prompt_id
            if lang not in SUPPORTED_LANGS:
                raise ValueError(
                    f"Input row {row_number} has unsupported lang {lang!r}."
                )
            prompts.append(
                {
                    "prompt_id": prompt_id,
                    "q_id": q_id,
                    "lang": lang,
                    "prompt": row.get(text_column) or "",
                }
            )
        return prompts


def write_token_rows(
    rows: Iterable[Mapping[str, str | int | float]], path: Path = TOKENS_PATH
) -> None:
    """Write token metrics in the frozen CSV column order."""
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=TOKEN_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def run(
    input_path: Path = PROMPTS_PATH,
    output_path: Path = TOKENS_PATH,
    encoders: Mapping[str, Encoder] | None = None,
) -> None:
    """Generate token metrics for every localized prompt."""
    prompts = read_prompts(input_path)
    selected_encoders = encoders if encoders is not None else load_encoders()
    rows = build_token_rows(prompts, selected_encoders)
    write_token_rows(rows, output_path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Count prompt tokens by language.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROMPTS_PATH,
        help="Input CSV with text and lang.",
    )
    parser.add_argument("--output", type=Path, default=TOKENS_PATH)
    args = parser.parse_args(argv)
    run(args.input, args.output)


if __name__ == "__main__":
    main()
