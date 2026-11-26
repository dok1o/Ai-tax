"""Keep the ARENA tests hermetic.

Real contract CSVs are hidden for the whole session, so every resolver falls
through to the committed fixtures in ``models/fixtures/`` and reports them as
fixtures. This keeps the expected sample sizes stable: the assertions in this
suite are about the committed fixture set, not about whatever lands in ``data/``.

Provenance mixing is no longer the reason for it. ``resolve_input`` now resolves
a prompts file that matches the table it is joined with, so a real
``data/prompts.csv`` next to fixture labels, tokens or routes can no longer break
training; ``tests/arena/test_review_fixes.py`` pins that.

Since the real T-004 observations landed in ``data/`` and the real artifacts in
``models/``, the session also runs the dashboard in its documented example mode
(``LT_FORCE_EXAMPLE=1``): that is the mode in which ``app.logic`` resolves the
fixture tables *and* the fixture-labelled artifacts, so an assertion about the
committed fixture set cannot be overturned by whatever a real run leaves behind.
Tests that need the real-data path build their own state explicitly.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from src import features

# Set at import time, before any fixture is created: the session-scoped fixtures
# in this suite resolve artifacts while they are being built, so example mode has
# to be on already. It keeps `resolve_artifact` and `app.logic.candidate` on the
# committed `*_fixture.*` files after a real run has written artifacts next to
# them.
os.environ.setdefault("LT_FORCE_EXAMPLE", "1")

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


@pytest.fixture(autouse=True, scope="session")
def example_mode() -> Iterator[None]:
    """Undo the example mode after the session, for whoever runs next."""
    yield
    os.environ.pop("LT_FORCE_EXAMPLE", None)
