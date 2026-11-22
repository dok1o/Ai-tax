"""Compatibility shim: ``make dashboard`` runs ``app/dashboard.py``.

The T-006 page lives in ``app/main.py``; this file only starts it so the
existing Makefile target keeps working until Astra repoints it.
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("main.py")), run_name="__main__")
