"""Value checks for the CSV contract in data/SCHEMA.md."""

import csv
from pathlib import Path

SCHEMA_VERSION = "2.1"
ACCURACY_METRICS = frozenset({"accuracy", "accuracy_strict", "accuracy_parsed"})
METRIC_NAMES = ACCURACY_METRICS | {"mcnemar_exact"}


def invalid_metric_names(path: Path) -> list[str]:
    """Return metric values in a metrics CSV that the contract does not admit."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        return sorted({row["metric"] for row in csv.DictReader(source)} - METRIC_NAMES)
