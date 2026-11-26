import csv
from pathlib import Path

import pytest

from src.common.config import (
    LABELS_PATH,
    METRICS_PATH,
    PROMPTS_PATH,
    RESPONSES_PATH,
    SCHEMA_PATH,
    TOKENS_PATH,
)
from src.common.schema import METRIC_NAMES, SCHEMA_VERSION, invalid_metric_names


@pytest.mark.parametrize(
    "path", [PROMPTS_PATH, RESPONSES_PATH, TOKENS_PATH, LABELS_PATH, METRICS_PATH]
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
    fixture = path.parent / "demo" / f"{path.stem}_fixture.csv"
    for dataset in (path, fixture):
        if not dataset.exists():
            continue
        with dataset.open(encoding="utf-8-sig", newline="") as source:
            assert next(csv.reader(source), None) == columns


def test_contract_version() -> None:
    header = SCHEMA_PATH.read_text(encoding="utf-8").splitlines()[0]
    assert header == f"# CSV contract v{SCHEMA_VERSION}"


def test_metric_enum_matches_contract() -> None:
    contract = SCHEMA_PATH.read_text(encoding="utf-8")
    row = next(line for line in contract.splitlines() if line.startswith("| metric |"))
    for name in METRIC_NAMES:
        assert name in row


@pytest.mark.parametrize(
    "path", [METRICS_PATH, METRICS_PATH.parent / "demo" / "metrics_fixture.csv"]
)
def test_metric_names(path: Path) -> None:
    if path.exists():
        assert invalid_metric_names(path) == []


def test_unknown_metric_name_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("metric,model\naccuracy_strict,m\nf1,m\n", encoding="utf-8")
    assert invalid_metric_names(path) == ["f1"]
