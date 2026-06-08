<div align="center">

[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-D22128?logo=apache&logoColor=white)](LICENSE.md)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/XXXX.XXXXX)

**Repository:** `https://github.com/fabiospagliardi/efficient_llm_safety_benchmarking`

</div>

# Efficient Safety Benchmarking via Item Response Theory

This repository applies **Item Response Theory (IRT)** — a psychometrics framework that models per-item difficulty and discrimination alongside latent model safety ability — to compress safety benchmarks without sacrificing ranking quality.

Two approaches are implemented:

- **Adaptive (Fluid Benchmarking):** Dynamically selects items per model via Maximum Fisher Information (computerized adaptive testing). Reduces evaluation cost by ≥80% while maintaining Spearman's ρ ≥ 0.90 with the full benchmark, and by up to **99.9%** on AIR-Bench 2024.
- **Static subset selection:** Extracts a fixed informative subset reusable across all models without per-model adaptation. Up to **99.8%** cost savings on AIR-Bench 2024; 80–92% across other benchmarks.

**Evaluated on six safety benchmarks:** AIR-Bench 2024 · Anthropic Red Team · HarmBench · SafetyBench · SimpleSafety · WMDP (Bio)

---

## Repository Structure

```
efficient_llm_safety_benchmarking/
├── subset_selection/           # Main module — item selection and evaluation
│   ├── select_subset.py        #   Entry point: fit IRT, run all methods, evaluate
│   ├── irt.py                  #   IRT fitting wrapper (calls irt-fit/ via subprocess)
│   ├── split.py                #   JSONL I/O, k-fold and random-sample splits
│   ├── fisher.py               #   Fisher information computation + FisherMethod class
│   ├── disco.py                #   DISCO entropy ranking + DiscoSubsetMethod class
│   ├── anchor_point_benchmark.py  # Anchor Point (K-Medoids) + AnchorPointMethod class
│   ├── fluid_benchmarking.py   #   Adaptive (per-model) CAT + FluidBenchmarkingMethod
│   ├── evaluation.py           #   Ability/accuracy estimators, per-K sweep
│   ├── utils.py                #   Item ranking helpers, RandomMethod class
│   └── input_files_paths.sh    #   Shell variables for supported benchmark paths
│
├── irt-fit/                    # IRT model fitting (py-irt / Pyro backend)
│   ├── fit_irt_model.py        #   CLI: fit 2PL model, writes parameters + abilities CSVs
│   └── two_param_logistic.py   #   2PL model definition for py-irt
│
├── fluid_benchmarking/         # Fluid (adaptive) benchmarking engine
│   ├── engine.py               #   run_fluid_benchmarking() — MFI-based CAT loop
│   ├── estimators.py           #   ability_estimate() — MAP/MLE Newton-Raphson
│   ├── irt_utils.py            #   sigmoid_stable, fisher_information
│   ├── config.py               #   HuggingFace dataset repo config
│   ├── datasets.py             #   Dataset loading utilities
│   └── evaluation.py          #   Evaluation helpers
│
└── pyproject.toml
```

---

## Installation

```bash
git clone https://github.com/fabiospagliardi/efficient_llm_safety_benchmarking.git
cd efficient_llm_safety_benchmarking
pip install -e .
```

**Requirements:** Python 3.11, NumPy, SciPy, pandas, tqdm, torch, pyro, py-irt, kmedoids.

---

## Input Format

The pipeline expects a JSONL file where each line is one model's full evaluation result:

```json
{"subject_id": "model-name", "responses": {"question_id_1": 1, "question_id_2": 0, ...}}
```

- `subject_id`: string identifier for the model
- `responses`: dict mapping question IDs to binary scores (1 = correct / safe, 0 = incorrect / unsafe)
- Every record must contain the same set of question IDs (rectangular matrix)

Supported benchmark paths are in `subset_selection/input_files_paths.sh`.

---

## Entry Point: `subset_selection/select_subset.py`

This script fits a 2PL IRT model on training models, selects item subsets using multiple methods, and evaluates each method on held-out models across K values (log-spaced from 1 to `--max_items`).

### Basic usage

```bash
# 5-fold cross-validation
python -m subset_selection.select_subset \
  --benchmark helm_airbench2024 \
  --input data/irt_fit_input_data/helm_airbench2024_irt_data.jsonl \
  --n_folds 5 \
  --max_items 500

# Random subsampling (100 train/test splits, 10 held-out models each)
python -m subset_selection.select_subset \
  --benchmark helm_airbench2024 \
  --input data/irt_fit_input_data/helm_airbench2024_irt_data.jsonl \
  --n_samples 100 \
  --test_size 10 \
  --max_items 500

# Include adaptive (fluid) benchmarking in addition to static methods
python -m subset_selection.select_subset \
  --benchmark helm_airbench2024 \
  --input data/irt_fit_input_data/helm_airbench2024_irt_data.jsonl \
  --n_folds 5 \
  --fluid

# Dry run: print dataset stats and exit
python -m subset_selection.select_subset \
  --benchmark helm_airbench2024 \
  --input data/irt_fit_input_data/helm_airbench2024_irt_data.jsonl \
  --dry_run
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--benchmark` | required | Name tag used in output filenames |
| `--input` | required | Path (relative to repo root) to the input JSONL |
| `--n_folds` | `None` | K-fold cross-validation over models. Mutually exclusive with `--n_samples` |
| `--n_samples` | `None` | Number of random train/test splits. Mutually exclusive with `--n_folds` |
| `--test_size` | `10` | Held-out models per split when using `--n_samples` |
| `--max_items` | `3000` | Maximum subset size to evaluate |
| `--min_variance` | `0.01` | Drop items with empirical variance p(1−p) below this threshold |
| `--n_quantiles` | `4` | Difficulty quantiles for `*_bquant` Fisher methods |
| `--n_points` | `20` | Number of K values (log-spaced) in the sweep |
| `--no_static` | flag | Skip all static methods (use with `--fluid` to run only adaptive) |
| `--fluid` | flag | Also run fluid (adaptive, per-model) benchmarking |
| `--dry_run` | flag | Print model/item counts and exit |

---

## Methods

All static methods are fit on training models and evaluated on held-out models at each K.

| Method name | Description |
|---|---|
| `all_items` | Baseline: full item set |
| `total_fisher` | Items ranked by summed Fisher information across training models |
| `marginal_fisher` | Greedy selection minimising Σ 1/√TIF — accounts for item redundancy |
| `total_fisher_bquant` | `total_fisher` with round-robin interleaving across b-difficulty quantiles |
| `marginal_fisher_bquant` | `marginal_fisher` with round-robin interleaving across b-difficulty quantiles |
| `disco` | Items ranked by binary entropy of empirical difficulty (DISCO) |
| `random` | Random ordering — null baseline |
| `anchor_point` | K-Medoids on Pearson-correlation distances (Vivek et al., 2024) |
| `fluid_benchmarking` | Per-model adaptive selection via maximum Fisher information (requires `--fluid`) |

---

## Outputs

Output files are written to the working directory, named by a run tag of the form:
`{benchmark}_{split_tag}_{n_quantiles}bquant_minvar{min_variance}_step{n_points}`

### Performance CSV (`*_{max_items}items_performance.csv`)

One row per (fold, method, K, model). Columns:

| Column | Description |
|---|---|
| `fold` | Fold or sample index |
| `method` | Method name (see table above) |
| `K` | Number of items in the subset |
| `model` | Model identifier |
| `split` | `"train"` or `"test"` |
| `ability` | IRT ability estimate (θ) |
| `ability_std` | Standard error: 1/√(Σ Fisher information) |
| `accuracy` | Mean accuracy on the K-item subset |
| `accuracy_std` | Binomial SE: √(p(1−p)/K) |
| `ap_pred` | APW (Anchor Point Weighted) score (anchor_point method only) |

### Items CSV (`*_items.csv`)

One row per item per fold. Columns include IRT parameters (`a`, `b`), `total_fisher`, `marginal_fisher_contribution`, per-method rank columns (`{method}_rank`), and anchor point inclusion flags (`anchor_B_{B}`).

---

## IRT Fitting

`irt-fit/fit_irt_model.py` fits a 2PL model using [py-irt](https://github.com/nd-ball/py-irt) (Pyro/PyTorch backend) and writes two CSVs to `irt-fit/fit_results/`:

- `*_parameters.csv`: item IDs, discrimination (`a`), difficulty (`b`), posterior scales
- `*_abilities.csv`: model IDs, fitted abilities, posterior scales

`subset_selection/irt.py` calls this script as a subprocess, passing a temporary JSONL file for each fold. Results are read back from `irt-fit/fit_results/` after the subprocess completes.

---

## Analysis

`subset_selection/subset_selection_plots.ipynb` contains correlation curve analysis comparing methods across K values.

---