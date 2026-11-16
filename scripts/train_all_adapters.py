#!/usr/bin/env python3
"""
CLI Script to train LoRA adapters for all three tasks sequentially.

Tasks:
  1. boolq
  2. gsm8k
  3. arc_easy

Supports:
  --preflight: run full system, dataset, and configuration preflight audit
  --development: permit synthetic dataset fallback for offline testing
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.train_adapter import train_single_adapter
from src.training.training_config import (
    load_experiment_config,
    load_training_config,
)
from src.training.training_utils import detect_hardware


def run_preflight_audit(
    training_config_path: str,
    experiment_config_path: str,
    allow_synthetic: bool = False
) -> bool:
    print("=" * 70)
    print(" PREFLIGHT AUDIT: MULTI-ADAPTER TRAINING SUITE ")
    print("=" * 70)

    hw = detect_hardware()
    print(f"Device Type        : {hw['device_type'].upper()}")
    print(f"Device Name        : {hw['device_name']}")
    print(f"Total RAM (GB)     : {hw['total_memory_gb']:.2f}")
    print(f"Free RAM (GB)      : {hw['free_memory_gb']:.2f}")
    print(f"CUDA Available     : {hw['cuda_available']}")

    training_cfg, lora_cfg = load_training_config(training_config_path)
    exp_cfg = load_experiment_config(experiment_config_path)

    is_dev = allow_synthetic or (exp_cfg.mode == "development")

    print(f"Configured Mode    : {exp_cfg.mode}")
    print(f"Allow Synthetic    : {is_dev}")
    print(f"Base Model         : {exp_cfg.base_model}")
    print(f"Tasks Configured   : {list(exp_cfg.tasks.keys())}")
    print(f"LoRA Rank (r)      : {lora_cfg.rank}")
    print(f"LoRA Alpha (alpha) : {lora_cfg.alpha}")
    print(f"Target Modules     : {lora_cfg.target_modules}")
    print("-" * 70)

    passed = True
    if not hw["cuda_available"] and not is_dev:
        print("[PREFLIGHT WARNING] No CUDA GPU detected and mode is 'final'.")
        print("Training on CPU in 'final' mode is prohibited to prevent invalid/fabricated claims.")
        passed = False

    # Check dry run on each task
    for task_name in ["boolq", "gsm8k", "arc_easy"]:
        print(f"\nChecking dataset and model compatibility for task '{task_name}'...")
        ok = train_single_adapter(
            task_name=task_name,
            training_config_path=training_config_path,
            experiment_config_path=experiment_config_path,
            allow_synthetic=is_dev,
            dry_run=True
        )
        if not ok:
            passed = False

    print("\n" + "=" * 70)
    if passed:
        print("PREFLIGHT STATUS: READY FOR TRAINING")
    else:
        print("PREFLIGHT STATUS: CHECKS FAILED OR REVISE FLAGS FOR OFFLINE DEV")
    print("=" * 70)
    return passed


def train_all(
    training_config_path: str = "configs/training.yaml",
    experiment_config_path: str = "configs/experiment.yaml",
    allow_synthetic: bool = False,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    device: str = None,
    preflight: bool = False,
) -> bool:
    if preflight:
        return run_preflight_audit(training_config_path, experiment_config_path, allow_synthetic)

    tasks = ["boolq", "gsm8k", "arc_easy"]
    start_all = time.time()
    results = {}

    print("=" * 70)
    print(" SEQUENTIAL THREE-TASK LoRA TRAINING PIPELINE ")
    print(f" Tasks: {', '.join(tasks)} ")
    print("=" * 70)

    for idx, task in enumerate(tasks, 1):
        print(f"\n[{idx}/3] Starting Training for Task: {task.upper()}...")
        success = train_single_adapter(
            task_name=task,
            training_config_path=training_config_path,
            experiment_config_path=experiment_config_path,
            allow_synthetic=allow_synthetic,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device_override=device
        )
        results[task] = success
        if not success:
            print(f"[ERROR] Training failed for task '{task}'. Halting pipeline.")
            return False

    elapsed = round(time.time() - start_all, 2)
    print("\n" + "=" * 70)
    print(f" ALL 3 ADAPTERS TRAINED SUCCESSFULLY IN {elapsed}s ")
    print(" Results Summary:")
    for t, res in results.items():
        print(f"  - {t:<12}: {'SUCCESS' if res else 'FAILED'}")
    print("=" * 70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA adapters for all tasks.")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training.yaml")
    parser.add_argument("--experiment-config", type=str, default="configs/experiment.yaml", help="Path to experiment.yaml")
    parser.add_argument("--development", "--allow-synthetic", action="store_true", help="Allow synthetic fallback")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Override device ('cpu' or 'cuda')")
    parser.add_argument("--preflight", action="store_true", help="Run preflight checks without training")
    args = parser.parse_args()

    ok = train_all(
        training_config_path=args.config,
        experiment_config_path=args.experiment_config,
        allow_synthetic=args.development,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        preflight=args.preflight
    )
    if not ok:
        sys.exit(1)
