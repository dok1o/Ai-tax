"""Compatibility shim: ``python run.py dashboard`` runs ``app/dashboard.py``.

The T-011 page lives in ``app/main.py``; this file only starts it so the shared
runner keeps working without a change outside the ARENA-owned paths.
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("main.py")), run_name="__main__")
