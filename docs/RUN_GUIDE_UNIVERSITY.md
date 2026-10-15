# RIS Assisted Wireless Federated Learning Run Guide

## Purpose and hardware fit

This guide runs the revised M2 joint client-selection and RIS design experiment
from the `tejas/m2-methodology` branch, created from `main`. The university
request specifies Ubuntu 20.04 or 22.04 LTS,
Python 3.10 or 3.11, CUDA 11.8 or CUDA 12.x with cuDNN, a GPU with 16–24 GB
VRAM, and 32–64 GB RAM. Those are suitable target resources for full CIFAR-10
experiments. MNIST on the fast configuration is for checking the installation
and execution path only.

The simulation does not directly measure GPU memory, client energy, or real
radio latency. It uses the GPU mainly for model training; wireless channels,
RIS phases, scheduling, and energy/latency estimates are simulated.

## 1. Clone the repository and select the implementation branch

Open a terminal in Ubuntu and run:

```bash
git clone --branch main --single-branch https://github.com/DHYEYDUMASIA24022006/RIS-Assisted-Wireless-Federated-Learning.git
cd RIS-Assisted-Wireless-Federated-Learning
git fetch origin
git switch tejas/m2-methodology
```

## 2. Check system and Python versions

```bash
lsb_release -a
python3 --version
nvidia-smi
```

Install Git, Python virtual-environment support, and system libraries if
needed:

```bash
sudo apt update
sudo apt install -y git python3-venv python3-dev build-essential
```

Create and activate the project environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 3. Install PyTorch for the installed CUDA driver

Use the official PyTorch installation selector to choose a PyTorch build for
Linux, pip, Python, and the CUDA runtime available in the selector. The
university's CUDA 11.8 or CUDA 12.x requirement refers to a compatible software
stack; the CUDA toolkit and PyTorch wheel must match the driver. Install the
matching `torch` and `torchvision` pair first. Then install the project's other
dependencies:

```bash
python -m pip install -r requirements.txt
```

If pip proposes replacing the CUDA-enabled PyTorch wheel, reinstall the
selected matching `torch`/`torchvision` pair afterward. Confirm CUDA access:

```bash
python -c "import torch; print('torch', torch.__version__); print('CUDA build', torch.version.cuda); print('available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The M2 change adds no packages and needs no new dataset. The code uses the
existing PyTorch, NumPy, SciPy, Matplotlib, Seaborn, PyYAML, CVXPY, tqdm, pandas,
and scikit-learn stack. The run downloads MNIST or CIFAR-10 through the existing
dataset loader on first use.

## 4. Review the M2 configuration

The branch includes the revised scheduler and the matching implementation
notes in `docs/M2_METHODOLOGY.md`. The changes add starvation-age scheduling,
seeded tie-breaking, validation of selection inputs, refill after final RIS
feasibility checks, and correct experiment-specific collection in multi-seed
runs. No manual source copying is needed after checking out this branch.

The branch sets these fields in `config/config.yaml` under `selection`:

```yaml
  learning_utility_weight: 0.65
  participation_debt_weight: 0.20
  starvation_age_weight: 0.15
  starvation_age_cap_rounds: 100
  utility_ema_momentum: 0.8
  utility_validation_samples: 512
```

These weights are normalized by the scheduler. `config/config_fast.yaml` is a
CPU MNIST smoke configuration and skips the M2 budget sweeps. Keep `max_participants`,
`latency_budget_ms`, `energy_budget_mj`, and compute/energy assumptions visible
in the saved experiment configuration.

## 5. Run a fast MNIST smoke run

From the repository root, activate `.venv`, then run the M2 experiment using
the fast config:

```bash
python main.py --experiment joint --config config/config_fast.yaml
```

This confirms package imports, dataset loading, training, output creation, and
the joint scheduler path. It is not final evidence. The fast config skips the
M2 budget sweeps and reports only the fixed-budget comparison.

## 6. Run the primary M2 evaluation

Use CIFAR-10 and the intended university GPU configuration:

```bash
python main.py --experiment joint --config config/config.yaml --seed 42
```

The repository's `main.py` CLI may vary slightly by branch; if `--seed` is not
accepted, set `system.seed` in the YAML config and rerun. Inspect the printed
device at startup and ensure it reports CUDA. A full run may take substantial
time because each strategy trains local models and the M2 experiment includes
budget sweeps unless disabled.

For a reproducible multi-seed evaluation:

```bash
python run_full_evaluation.py --config config/config.yaml --experiments joint --seeds 42 123 456 --workers 1
```

Keep CUDA multi-seed runs serial (`--workers 1`) to avoid multiple processes
competing for GPU memory. To parallelize independent seeds on CPU, configure
`system.device: cpu` and use a modest process count, for example:

```bash
python run_full_evaluation.py --config config/config_fast.yaml --experiments joint --seeds 42 123 456 --workers 3
```

Each process uses Python's `ProcessPoolExecutor`; this is process pooling over
independent seeds. Local clients inside a single federated round are not
parallelized by this setting. The scheduler remains sequential at the server.

## 7. Outputs and interpretation

Look under the configured `system.results_dir` (normally `results/`):

- `exp2/`: M2 per-strategy CSV/JSON logs, summary JSON, and plots.
- `multi_seed/seed_<seed>/`: isolated experiment outputs per seed.
- `multi_seed/aggregated_results.json`: numeric means and standard deviations
  for the requested experiments only.

Review at least final and best test accuracy, rounds to target, successful
uploads, idle-round fraction, Jain fairness over successful uploads, mean and
maximum modeled latency, mean modeled energy, outage, and resource-constraint
violations. Confirm the proposed scheduler reports zero modeled constraint
violations. Report CPU cycles/sample, assumed CPU frequency, transmit power,
bandwidth, noise, model size, local epochs, and energy coefficients beside the
results. State clearly that values are modeled, not measured on real clients.

For a defensible comparison, retain config files and logs, use the same client
partitions/seeds for each strategy, include multiple seeds, and compare with
random and channel-only baselines. Add ablations removing utility, upload debt,
and starvation age before making claims about which mechanism helps. Treat
short MNIST output as a smoke check, not journal evidence.

## 8. Troubleshooting

- **`torch.cuda.is_available()` is false:** check `nvidia-smi`, driver support,
  and that the installed Torch wheel is the CUDA build selected for the host.
- **Out of GPU memory:** reduce batch size or local epochs, use the MNIST fast
  config for debugging, or set one worker. Do not use concurrent CUDA seed
  processes for the full experiments.
- **No clients selected:** the current budgets may make every client
  infeasible. Inspect the modeled rate, latency, energy, model size, and budget
  values. Do not loosen budgets silently; report any change.
- **CIFAR-10 download fails:** check outbound network access from the compute
  node or place the dataset where the existing TorchVision loader can read it.
- **Results seem unexpectedly high or low:** verify dataset split and Dirichlet
  alpha, seeds, evaluation rounds, communication noise, and whether the run used
  the intended config.
