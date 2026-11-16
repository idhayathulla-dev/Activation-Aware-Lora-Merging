# Activation-Distribution-Aware LoRA Merging

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Autonomous AI/ML research engineering framework for investigating whether using input activation distributions to transform LoRA updates prior to multi-adapter alignment and merging preserves task-specific behavior better than conventional weight-space merging under equal rank and parameter constraints.

---

## 1. Central Research Question

> **"Can activation-distribution-aware low-rank alignment and merging preserve task-specific behavior better than conventional weight-space LoRA merging under an equal rank and parameter budget?"**

Existing LoRA merging algorithms (KnOTS, CtM, TIES, Task Arithmetic) optimize or truncate updates under an isotropic weight-space Frobenius norm:
$$\min_{\widehat{\Delta W}} \|\Delta W - \widehat{\Delta W}\|_F^2$$

However, in actual inference, linear layers compute $y = (W_0 + \Delta W) x$. The expected functional output error is:
$$\mathbb{E}_{x \sim \mathcal{D}}[\|\Delta W x - \widehat{\Delta W} x\|_2^2] = \text{Tr}\left((\Delta W - \widehat{\Delta W}) C_x (\Delta W - \widehat{\Delta W})^T\right) = \|(\Delta W - \widehat{\Delta W}) C_x^{1/2}\|_F^2$$
where $C_x = \mathbb{E}[x x^T]$ is the uncentered second moment of layer input activations.

This project implements and tests **Activation-Distribution-Aware KnOTS Merging**: transforming task updates into canonical activation coordinates via $C_\lambda^{1/2}$, performing joint SVD alignment across task updates, merging in the aligned space, and reconstructing an exact rank-$R$ LoRA adapter via $C_\lambda^{-1/2}$.

---

## 2. Project Directory Structure

```text
activation-aware-lora-merging/
├── README.md                          # Project documentation and reproduction guide
├── requirements.txt                   # Tested Python dependencies
├── pyproject.toml                     # Package configuration
├── .gitignore                         # Version control ignore rules
│
├── configs/                           # Experiment and model configurations
│   ├── base.yaml                      # Base model (Qwen2.5-0.5B-Instruct) settings
│   ├── training.yaml                  # LoRA training hyperparameters (BoolQ, GSM8K, ARC-Easy)
│   ├── calibration.yaml               # Activation calibration settings
│   └── experiments.yaml               # Merging methods & evaluation matrix
│
├── notebooks/                         # Self-contained Google Colab notebook
│   └── activation_aware_lora_merging.ipynb
│
├── research/                          # Scientific documentation & literature audit
│   ├── literature_review.md           # Taxonomy & literature survey (KnOTS, TIES, SVD-LLM, etc.)
│   ├── mathematical_formulation.md    # Derivations, proofs, and numerical invariants
│   ├── experiment_protocol.md         # Data splits, evaluation protocol, and metrics
│   └── research_log.md                # Ongoing experiment and engineering log
│
├── src/                               # Modular source code
│   ├── __init__.py
│   ├── model_loader.py                # Model and tokenizer loading with fallbacks
│   ├── dataset_loader.py              # HF dataset loading & split partitioning
│   ├── data_preprocessing.py          # Task prompt formatting (BoolQ, GSM8K, ARC-Easy)
│   ├── adapter_training.py            # Sequential LoRA training
│   ├── provenance_audit.py            # Checkpoint verification & update norm audit
│   ├── activation_collector.py        # Forward hook activation accumulator
│   ├── covariance_estimator.py        # Task-balanced uncentered second moments
│   ├── matrix_utils.py                # Symmetric eigendecomposition & square roots
│   ├── merging/                       # Model merging algorithms
│   │   ├── __init__.py
│   │   ├── task_arithmetic.py         # Baseline 1: Task Arithmetic
│   │   ├── ties.py                    # Baseline 2: TIES-Merging
│   │   ├── knots.py                   # Baseline 3: Weight-space KnOTS
│   │   └── activation_aware_knots.py  # Proposed: Activation-aware KnOTS
│   ├── evaluation.py                  # Multi-task evaluation harness
│   ├── metrics.py                     # Accuracy, GSM8K numeric EM, degradation
│   └── experiment_runner.py           # Experiment runner orchestrating all conditions
│
├── tests/                             # Comprehensive unit & integration tests
│   ├── test_covariance.py             # Numerical stability & PSD checks
│   ├── test_matrix_transforms.py      # Square root & isometric equivalence tests
│   ├── test_lora_updates.py           # LoRA extraction and factorization tests
│   ├── test_provenance.py             # Checkpoint auditing unit tests
│   └── test_merging.py                # Merging algorithms unit tests
│
├── scripts/                           # Reproducibility CLI scripts
│   ├── run_preflight.py               # Hardware & environment audit
│   ├── train_adapters.py              # Train LoRA adapters
│   ├── audit_adapters.py              # Audit adapter provenance
│   ├── collect_activations.py         # Collect calibration activations
│   ├── run_baselines.py               # Run baseline merges
│   ├── run_proposed_method.py         # Run activation-aware merges
│   └── evaluate_all.py                # Run full evaluation suite
│
├── results/                           # Experimental outputs
│   ├── raw/                           # Raw JSON predictions
│   ├── processed/                     # Covariances and aggregated metrics
│   ├── figures/                       # Plots and charts
│   └── reports/                       # Formal markdown and JSON audit reports
│
└── logs/                              # Execution logs
```

---

## 3. Quickstart & Environment Setup

### 3.1 Local Environment (CPU / Development)
For unit testing, synthetic matrix validation, and pipeline verification:

```bash
cd activation-aware-lora-merging
python -m pip install -r requirements.txt
python scripts/run_preflight.py
pytest tests/ -v
```

### 3.2 Google Colab (GPU Training & Full Benchmark)
For fine-tuning adapters on `Qwen/Qwen2.5-0.5B-Instruct` and executing the full multi-task benchmark:
1. Open `notebooks/activation_aware_lora_merging.ipynb` in Google Colab.
2. Select a GPU runtime (T4, V100, or A100).
3. Execute all cells sequentially.

---

## 4. Experimental Matrix

Every merged model is evaluated on all three tasks under an identical rank budget ($R \in \{4, 8, 16\}$):

1. **Base Model (Zero-shot)**: Untouched base checkpoint.
2. **Individual Adapters**: Adapter A (BoolQ), Adapter B (GSM8K), Adapter C (ARC-Easy).
3. **Task Arithmetic**: Weight-space linear sum with uniform task weighting.
4. **TIES-Merging**: Weight-space trim (top 80%), elect sign, disjoint merge.
5. **KnOTS + Task Arithmetic**: Joint SVD alignment in weight space + linear average.
6. **KnOTS + TIES**: Joint SVD alignment in weight space + TIES merge.
7. **Activation-Aware KnOTS + Task Arithmetic (Proposed)**: Activation covariance square root transform + KnOTS + Task Arithmetic.
8. **Activation-Aware KnOTS + TIES (Proposed)**: Activation covariance square root transform + KnOTS + TIES.
