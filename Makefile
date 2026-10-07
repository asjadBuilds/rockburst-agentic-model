# Rockburst agentic assessment — common targets
.PHONY: setup extract train evaluate pipeline test assess-examples llm-compare

PYTHON ?= .venv/bin/python
CLI = $(PYTHON) scripts/rockburst_cli.py

setup:
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

extract:
	$(CLI) extract

train:
	$(CLI) train

evaluate:
	$(CLI) evaluate

pipeline: extract train evaluate

test:
	.venv/bin/pytest -q

assess-examples:
	$(CLI) assess --no 1
	$(CLI) assess --no 20
	$(CLI) assess --no 146
	$(CLI) assess --no 244
