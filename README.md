[![DOI](https://zenodo.org/badge/681966574.svg)](https://doi.org/10.5281/zenodo.13884362)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

# EvoFed — Evolutionary Federated Learning

**EvoFed** is a JAX-based research framework that replaces the traditional gradient-aggregation step in Federated Learning with **Evolution Strategies (ES)**. Instead of averaging client gradients, the server runs an ES loop that treats client model updates as fitness signals, enabling communication-efficient and privacy-preserving model training across distributed clients.

---

## Features

- **Evolutionary aggregation** — replaces FedAvg's parameter averaging with ES optimizers (OpenES, PGPE, CMA-ES)
- **Multiple baselines** — FedAvg, FedAvg + quantization, FedAvg + sparsification
- **Communication compression** — built-in sparsification and quantization utilities
- **IID & Non-IID splits** — configurable data heterogeneity across clients
- **Vision benchmarks** — MNIST, FashionMNIST, and CIFAR-10 out of the box
- **Experiment tracking** — first-class [Weights & Biases](https://wandb.ai) integration
- **Hardware-accelerated** — fully JIT-compiled via JAX with GPU/TPU support

---

## Installation

### Requirements

- Python ≥ 3.9
- CUDA 12 (for GPU acceleration)

### Step 1 — Install JAX with GPU support

```bash
pip install -U "jax[cuda12]"
```

> For CPU-only or other CUDA versions see the [JAX installation guide](https://github.com/google/jax#installation).

### Step 2 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — (Optional) Install the package locally

```bash
pip install -e .
```

---

## Quick Start

### Run EvoFed on FashionMNIST

```bash
python evofed.py --config configs/Vision-FMNIST/evofed.yaml
```

### Run the FedAvg baseline

```bash
python fedavg.py --config configs/Vision-FMNIST/fedavg.yaml
```

### Run on MNIST or CIFAR-10

```bash
python evofed.py --config configs/Vision-MNIST/evofed.yaml
python evofed.py --config configs/Vision-CIFAR10/evofed.yaml
```

### Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/Vision-FMNIST/evofed.yaml` | Path to YAML config file |
| `--seed` | `0` | Random seed for reproducibility |
| `--log-dir` | `/tmp/gym/` | Directory for experiment logs |

---

## Configuration

All hyperparameters are controlled by YAML files in `configs/`. Below is an annotated excerpt from `configs/Vision-FMNIST/evofed.yaml`:

```yaml
# Federated learning settings
n_rounds: 1000          # Total communication rounds
n_clients: 5            # Number of federated clients
dist: "NON-IID"         # Data distribution: IID | NON-IID
batch_size: 512

# Local training (SGD)
lr: 0.087
momentum: 0.907

# Neural network (CNN)
network_name: "CNN"

# Evolution strategy
strategy: "OpenES"      # OpenES | PGPE | CMA_ES
pop_size: 128           # ES population size
sigma_init: 0.35        # Initial noise scale

# Fitness shaping
maximize: true
z_score: true
w_decay: 0.015

# Communication compression
percentage: 0.0         # Sparsification ratio (0 = no sparsification)
```

Pre-built configs are available for every dataset/algorithm combination under `configs/`.

---

## Algorithms

| Script | Algorithm | Description |
|---|---|---|
| `evofed.py` | **EvoFed** | ES-based federated aggregation (single GPU) |
| `evofed_parallel.py` | **EvoFed Parallel** | Multi-device parallel variant |
| `evofed_partitioning.py` | **EvoFed Partitioning** | Client partitioning variant |
| `fedavg.py` | **FedAvg** | Standard Federated Averaging baseline |
| `fedavg_quantization.py` | **FedAvg-Q** | FedAvg with quantized updates |
| `fedavg_sparse.py` | **FedAvg-S** | FedAvg with sparsified updates |
| `pbge.py` | **PBGE** | Population-Based Gradient Estimation |
| `pbge_partitioning.py` | **PBGE Partitioning** | PBGE with client partitioning |
| `bp.py` | **BP** | Pure backpropagation (centralized) reference |

### Supported evolution strategies

| Strategy | Key |
|---|---|
| OpenAI Evolution Strategies | `OpenES` |
| Parameter-Exploring Policy Gradients | `PGPE` |
| Covariance Matrix Adaptation ES | `CMA_ES` |

---

## Project Structure

```
EvoFL/
├── configs/                    # YAML hyperparameter configs
│   ├── Vision-MNIST/
│   ├── Vision-FMNIST/
│   └── Vision-CIFAR10/
├── backprop/
│   └── sl.py                   # Supervised learning utilities (data loading, train/eval)
├── utils/
│   ├── evo.py                  # ES strategy factory helpers
│   ├── helpers.py              # Config loading & misc utilities
│   └── models.py               # Neural network definitions
├── evosax/                     # Bundled evosax ES library (JAX)
├── evofed.py                   # EvoFed main entry point
├── evofed_parallel.py          # Parallel EvoFed
├── evofed_partitioning.py      # Partitioning EvoFed
├── fedavg.py                   # FedAvg baseline
├── fedavg_quantization.py      # FedAvg + quantization
├── fedavg_sparse.py            # FedAvg + sparsification
├── pbge.py                     # PBGE algorithm
├── pbge_partitioning.py        # PBGE + partitioning
├── bp.py                       # Backprop (centralized) baseline
├── quantization.py             # Quantization utilities
├── sparsification.py           # Sparsification utilities
├── args.py                     # CLI argument parsing
├── requirements.txt
└── setup.py
```

---

## Citation

If you use EvoFed in your research, please cite:

```bibtex
@software{evofed2024,
  author  = {Rahimi, Mahi},
  title   = {{EvoFed}: Evolutionary Federated Learning},
  year    = {2024},
  doi     = {10.5281/zenodo.13884362},
  url     = {https://github.com/mahi97/EvoFL}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

