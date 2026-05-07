.PHONY: test secrets figures adversarial-summary public-artifacts check

PYTHON ?= .venv/bin/python

test:
	$(PYTHON) -m pytest -q

secrets:
	$(PYTHON) check_no_secrets.py

adversarial-summary:
	$(PYTHON) summarize_adversarial.py

figures:
	$(PYTHON) plot_results.py \
		--summary reports/gt_bench_results.json \
		--robustness reports/robustness_results.json \
		--adversarial reports/adversarial_results.json \
		--out-dir reports/figures

public-artifacts: adversarial-summary figures

check: test secrets public-artifacts
	git diff --check
