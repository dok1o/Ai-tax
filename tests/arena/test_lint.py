"""Lint regression test for the ARENA-owned files.

`make check` runs ruff with the repository defaults, which miss line length and
a few bug-prone patterns that a reviewer does flag. This test pins the stricter
selection for the files ARENA owns, so the same findings cannot come back.

Ignored on purpose: E402, because `app/main.py` and the fixture generator put the
repository root on `sys.path` before importing project modules; C420, because
`{name: np.full(...) for name in models}` must build one array per key and
`dict.fromkeys` would share a single object.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARENA_PATHS = (
    "src/features.py",
    "src/train.py",
    "src/router.py",
    "app",
    "tests/arena",
    "models/fixtures",
)
SELECTION = "E,F,W,I,UP,B,SIM,PLW,ARG,RET,ISC"
IGNORE = "E402,C420"


def test_arena_files_pass_the_stricter_ruff_selection() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            SELECTION,
            "--ignore",
            IGNORE,
            "--output-format",
            "concise",
            *ARENA_PATHS,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if "No module named ruff" in completed.stderr:
        pytest.skip("ruff is not installed in this interpreter")
    assert completed.returncode == 0, completed.stdout[-3000:]
