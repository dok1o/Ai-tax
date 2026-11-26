"""Keep the ARENA tests hermetic.

Real contract CSVs are hidden for the whole session, so every resolver falls
through to the committed fixtures in ``models/fixtures/`` and reports them as
fixtures. Without this the suite changes behaviour as soon as codex lands real
data in ``data/``, and mixing a real ``prompts.csv`` with fixture tokens, labels
or routes joins on identifiers that do not exist in both files.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from src import features

KINDS = ("PROMPTS_PATH", "TOKENS_PATH", "LABELS_PATH", "ROUTES_PATH")
ABSENT_DIR = features.MODELS_DIR / "absent_inputs"


@pytest.fixture(autouse=True, scope="session")
def hidden_real_inputs() -> Iterator[None]:
    """Point the contract paths at files that do not exist."""
    originals = {name: getattr(features, name) for name in KINDS}
    for name in KINDS:
        setattr(features, name, ABSENT_DIR / f"{name.lower()}.csv")
    yield
    for name, value in originals.items():
        setattr(features, name, value)
