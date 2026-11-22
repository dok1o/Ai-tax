import csv
from pathlib import Path

import pytest

from src.common.config import (
    LABELS_PATH,
    PROMPTS_PATH,
    RESPONSES_PATH,
    SCHEMA_PATH,
    TOKENS_PATH,
)


@pytest.mark.parametrize(
    "path", [PROMPTS_PATH, RESPONSES_PATH, TOKENS_PATH, LABELS_PATH]
)
def test_contract_header(path: Path) -> None:
    contract = SCHEMA_PATH.read_text(encoding="utf-8")
    section = contract.split(f"## {path.name}\n", 1)[1].split("\n## ", 1)[0]
    columns = [
        row.split("|")[1].strip()
        for row in section.splitlines()
        if row.startswith("| ")
    ][2:]
    assert columns and len(columns) == len(set(columns))
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as source:
            assert next(csv.reader(source), None) == columns
