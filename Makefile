.PHONY: test secrets figures adversarial-summary seed-sweep-summary suite-data suite-baselines public-artifacts check

PYTHON ?= .venv/bin/python

test:
	$(PYTHON) -m pytest -q

secrets:
	$(PYTHON) check_no_secrets.py

adversarial-summary:
	$(PYTHON) summarize_adversarial.py

seed-sweep-summary:
	$(PYTHON) summarize_seed_sweep.py

figures:
	$(PYTHON) plot_results.py \
		--summary reports/gt_bench_results.json \
		--robustness reports/robustness_results.json \
		--adversarial reports/adversarial_results.json \
		--seed-sweep reports/seed_sweep_results.json \
		--out-dir reports/figures

suite-data:
	$(PYTHON) generate_benchmark_suite.py \
		--train-per-family 200 \
		--val-per-family 10 \
		--test-per-family 50 \
		--seed 20260511 \
		--out-dir data/suite

suite-baselines: suite-data
	$(PYTHON) run_suite_baselines.py \
		--gold data/suite/test.jsonl \
		--train data/suite/train.jsonl \
		--pred-dir predictions/suite_baselines \
		--out-json reports/suite_results.json \
		--out-md reports/suite_results.md \
		--figure reports/figures/suite_smoke_accuracy.svg

public-artifacts: adversarial-summary seed-sweep-summary figures suite-baselines

check: test secrets public-artifacts
	git diff --check
