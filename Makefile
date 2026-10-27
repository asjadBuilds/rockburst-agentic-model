# Rockburst agentic assessment — common targets
.PHONY: setup preprocess extract train evaluate sensitivities examples figures notebook test tests archive package pipeline assess-examples llm-compare

PYTHON ?= .venv/bin/python
CLI = $(PYTHON) scripts/rockburst_cli.py

setup:
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

extract:
	$(CLI) extract

preprocess: extract

train:
	$(CLI) train

evaluate:
	$(CLI) evaluate

sensitivities:
	$(PYTHON) scripts/triplet_label_sensitivity.py

examples:
	$(PYTHON) scripts/generate_examples.py

figures:
	$(PYTHON) scripts/generate_figures.py

notebook:
	.venv/bin/jupyter nbconvert --to notebook --execute notebooks/end_to_end_demo.ipynb --inplace

archive:
	$(PYTHON) scripts/archive_stale_artefacts.py

package:
	bash scripts/package_delivery.sh

pipeline: extract train evaluate

test:
	.venv/bin/pytest -q

tests: test

assess-examples:
	$(CLI) assess --no 1
	$(CLI) assess --no 20
	$(CLI) assess --no 146
	$(CLI) assess --no 244
