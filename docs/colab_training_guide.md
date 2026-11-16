# Google Colab GPU Execution & Multi-Adapter Training Guide

This guide details the exact procedure for training authentic, production-grade LoRA adapters on **Google Colab (CUDA GPU: T4, V100, or A100)** with real Hugging Face datasets.

---

## 1. Environment & Hardware Requirements

| Requirement | Specification |
|---|---|
| **Platform** | Google Colab (Standard T4 16GB GPU or Colab Pro A100/V100) |
| **VRAM Note** | The 6 GB VRAM threshold is an initial lower-bound preflight heuristic. Real peak memory depends on dynamic sequence lengths (up to 512 tokens), activation caching, and precision. A standard T4 (15–16 GB) is strongly recommended. |
| **CUDA Driver** | $\ge 12.1$ |
| **Python** | $\ge 3.10$ |
| **Base Model** | `Qwen/Qwen2.5-0.5B-Instruct` |
| **Precision** | `bfloat16` (or `float16` on T4) |
| **Tasks** | `google/boolq`, `openai/gsm8k` (main), `allenai/ai2_arc` (ARC-Easy) |
| **Target Commit SHA** | `c7c6d62d3f8ebf547b71028327c9f7674960548a` |

---

## 2. Step 1: Runtime Setup & Persistent Storage in Colab

In your first Google Colab cell:

```python
# 1. Mount Google Drive for persistent artifact preservation
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/activation_aware_lora_merging_artifacts

# 2. Check GPU presence & system resources
import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemError("CUDA GPU is required! Change runtime type to T4, V100, or A100 GPU.")
print("GPU Hardware:", torch.cuda.get_device_name(0))

# 3. Clone repository and enter project directory
!git clone https://github.com/idhayathulla-dev/Activation-Aware-Lora-Merging.git
%cd Activation-Aware-Lora-Merging

# 4. Install compatible dependencies (without breaking Colab's CUDA PyTorch)
!pip install -q transformers peft datasets accelerate safetensors pyyaml pytest scipy psutil
```

---

## 3. Step 2: Authenticity Audit & Provenance Verification

Run the dataset provenance audit in **FINAL** mode. This connects directly to Hugging Face, downloads real benchmark splits, verifies disjointness, and records SHA-256 fingerprints.

```bash
# Ensure configs/experiment.yaml has mode: "final" and allow_synthetic: false
!python scripts/audit_datasets.py --config configs/experiment.yaml
```

**Expected Output:**
- `artifacts/provenance/dataset_manifest.json` (SHA-256 hashes per partition)
- `artifacts/provenance/split_integrity_report.json`
- `artifacts/provenance/split_integrity_report.md`
- Status: `ALL CHECKS PASSED (AUTHENTIC_HF)`

---

## 4. Step 3: Preflight Hardware & Configuration Check

Run the multi-adapter preflight check to confirm GPU memory and hyperparameter parity:

```bash
!python scripts/train_all_adapters.py --preflight
```

**What this verifies:**
1. GPU VRAM $\ge 6$ GB available.
2. Identical LoRA hyperparameter configuration across all tasks ($r=8, \alpha=16$, dropout=0.05, target modules `q_proj`, `v_proj`).
3. Total parameter parity: 22,528 trainable parameters per adapter ($0.0045\%$ of base model).
4. Zero leakage between train ($172$), calibration ($128$), validation ($100$), and evaluation ($200$) splits.

---

## 5. Step 4: Execute Unified Multi-Adapter Training Pipeline

Execute full automated adapter training across all three tasks using the primary unified command:

```bash
!python scripts/train_all_adapters.py \
    --config configs/training.yaml \
    --experiment-config configs/experiment.yaml \
    --device cuda \
    --backup-dir /content/drive/MyDrive/activation_aware_lora_merging_artifacts
```

### What this unified pipeline guarantees:
1. **Clean Directory Reset:** If any task was interrupted previously, its staging directory is cleanly wiped prior to training to eliminate stale or partial files.
2. **Sequential Execution:** Trains BoolQ, GSM8K, and ARC-Easy with identical hyperparameters ($r=8, \alpha=16, lr=2\times 10^{-4}$, 3 epochs, batch size 4, gradient accumulation 4).
3. **Immediate Per-Task Verification:** Directly verifies weights, dimensions, and finite numbers via `verify_single_adapter` the moment each task finishes.
4. **Immediate Google Drive Mirroring:** Backs up each verified adapter to `/content/drive/MyDrive/.../adapters/<task>/` before starting the next task.
5. **Joint Verification:** Runs cross-task divergence audit and reports overall health.

### Training Parameters (configs/training.yaml):
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Learning Rate**: $2 \times 10^{-4}$ with AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $0.01$)
- **Scheduler**: Linear warmup ($5\%$ steps) with linear decay
- **Batch Size**: 4 per device, gradient accumulation 4 (effective batch size = 16)
- **Epochs**: 3 epochs
- **Validation**: Evaluated at end of each epoch; checkpoint with lowest validation loss is automatically retained.

---

## 6. Step 5: Verify Saved Adapter Checkpoints

Run the automated verification suite to validate weights, PEFT reloading, and forward inference:

```bash
!python scripts/verify_adapters.py --device cuda
```

**Audit Checks Performed:**
1. **Required Files**: Confirms `adapter_config.json`, `adapter_model.safetensors`, `training_metadata.json`, and `loss_curve.json`.
2. **LoRA Parity**: Verifies $r=8, \alpha=16$, target modules `['q_proj', 'v_proj']`.
3. **Weight Numerical Health**: Confirms all weights are finite (no NaN, no Inf, non-zero).
4. **Cross-Task Distinctness**: Confirms adapters are mathematically distinct and divergent across tasks ($\max |\Delta W_i - \Delta W_j| > 0$).
5. **Authentic Forward Smoke Test**: Loads base model, attaches each adapter via PEFT, and verifies forward pass produces finite logits.

---

## 7. Step 6: Backup Checkpoints to Google Drive

Mirror the verified artifacts to your permanent Google Drive storage:

```python
import shutil
from pathlib import Path

persistent_dir = Path("/content/drive/MyDrive/activation_aware_lora_merging_artifacts")
for sub in ["adapters", "provenance", "experiment"]:
    src = Path(f"artifacts/{sub}")
    dst = persistent_dir / sub
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Backed up {src} -> {dst}")
print("Backup complete. Checkpoints are safely preserved in Google Drive.")
```

---

## 8. Provenance & Scientific Integrity Policy

1. **No Synthetic Fallback in Final Mode**: Synthetic data is strictly rejected when running `--config configs/experiment.yaml` in `final` mode.
2. **Zero Calibration Leakage**: Calibration examples ($128$ per task) are strictly held out for activation-distribution covariance estimation and are NEVER shown to the training optimizer or used for checkpoint selection.
3. **Reproducibility Guarantee**: Every training run logs full git commit hash, environment packages, random seeds, and per-step loss curves in `training_metadata.json`.
