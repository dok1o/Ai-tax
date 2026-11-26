"""Keep the ARENA tests hermetic.

Real contract CSVs are hidden for the whole session, so every resolver falls
through to the committed fixtures in ``models/fixtures/`` and reports them as
fixtures. This keeps the expected sample sizes stable: the assertions in this
suite are about the committed fixture set, not about whatever lands in ``data/``.

Provenance mixing is no longer the reason for it. ``resolve_input`` now resolves
a prompts file that matches the table it is joined with, so a real
``data/prompts.csv`` next to fixture labels, tokens or routes can no longer break
training; ``tests/arena/test_review_fixes.py`` pins that. The suite passes both
with and without this pinning.
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
