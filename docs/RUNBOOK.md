# Reproducible Experiment Runbook

## Environment

On Windows PowerShell, from the project directory:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py --experiment quick
```

The quick run is expected to finish in about half a minute on CPU. It validates
the end-to-end FL, fading, RIS, outage, metric logging, and plotting paths.

## Final-study sequence

Run each experiment separately so its logs and figures are easy to inspect:

```powershell
.\venv\Scripts\python.exe main.py --experiment baseline --seed 42
.\venv\Scripts\python.exe main.py --experiment joint --seed 42
.\venv\Scripts\python.exe main.py --experiment aircomp --seed 42
```

Run M2 with the main CIFAR-10 configuration and repeat it across independent
seeds. CPU hosts may parallelize those independent seed runs; CUDA hosts stay
serial unless GPU memory and throughput have been measured explicitly:

```powershell
.\venv\Scripts\python.exe run_full_evaluation.py --experiments joint --seeds 42 123 456 --workers 1
# CPU-only alternative after checking available memory:
.\venv\Scripts\python.exe run_full_evaluation.py --experiments joint --seeds 42 123 456 --workers 3
```

M2 reserves `data.validation_fraction` of the training examples for scheduler
utility feedback. That subset is not assigned to clients; the test set remains
for final evaluation only. Inspect `results/multi_seed/seed_<seed>/exp2/` for
`exp2_config.json`, the summary, per-round CSVs, and figures. Review
`docs/M2_METHODOLOGY.md` for assumptions, feasibility handling, baselines, and
the interpretation of the learning-utility and participation-debt signals.

The revised M2 and CPU process pool need no additional third-party package or
dataset. The primary study uses the existing TorchVision CIFAR-10 dataset.

Repeat with seeds 7 and 2026. Before each repeat, set a distinct
`system.results_dir` in a copy of `config/config.yaml` to preserve all raw logs.
For final studies, use a CUDA device when available. CPU execution of the default
200-round CIFAR-10 configuration is intentionally slow.

## Evidence checklist

* Preserve the per-round CSV and summary JSON files.
* Record the exact configuration used for each seed.
* Build tables from the seed-level summaries; report mean plus standard deviation.
* Keep quick-test outputs separate from final-study outputs.
* If a PDF is open on Windows, the plotting helper writes a timestamped replacement
  rather than terminating the completed experiment.
