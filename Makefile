PYTHON ?= python

.PHONY: check tokens dashboard

check:
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pytest -q
ifneq ($(wildcard data/validate_prompts.py),)
	$(PYTHON) -m data.validate_prompts
endif

tokens:
	$(PYTHON) -c "from pathlib import Path; import sys; Path('src/tokens.py').stat().st_size or sys.exit('Token counting is pending T-001.')"
	$(PYTHON) -m src.tokens

dashboard:
	$(PYTHON) -c "from pathlib import Path; import sys; Path('app/dashboard.py').stat().st_size or sys.exit('The offline dashboard is pending T-006.')"
	$(PYTHON) -m streamlit run app/dashboard.py
