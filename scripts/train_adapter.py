#!/usr/bin/env python3
"""
CLI Script to train an independent LoRA adapter for a single task.

Usage:
    python scripts/train_adapter.py --task boolq --config configs/training.yaml
    python scripts/train_adapter.py --task gsm8k --development --allow-synthetic --epochs 1
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.dataset_loader import load_task_dataset, compute_example_content_sha256
from src.model_loader import load_model_and_tokenizer
from src.training.training_config import (
    load_experiment_config,
    load_training_config,
)
from src.training.lora_trainer import LoRATrainer
from src.training.training_utils import detect_hardware, set_seed


def train_single_adapter(
    task_name: str,
    training_config_path: str = "configs/training.yaml",
    experiment_config_path: str = "configs/experiment.yaml",
    output_dir: str = None,
    allow_synthetic: bool = False,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    device_override: str = None,
    dry_run: bool = False,
) -> bool:
    print("=" * 70)
    print(f" INDEPENDENT LoRA ADAPTER TRAINING: {task_name.upper()} ")
    print("=" * 70)

    # 1. Load Configurations
    training_cfg, lora_cfg = load_training_config(training_config_path)
    exp_cfg = load_experiment_config(experiment_config_path)

    # Overrides from CLI
    if epochs is not None:
        training_cfg.num_train_epochs = epochs
    if batch_size is not None:
        training_cfg.per_device_train_batch_size = batch_size
    if learning_rate is not None:
        training_cfg.learning_rate = learning_rate

    # Target output directory
    if output_dir is None:
        output_path = Path("artifacts") / "adapters" / task_name
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check mode and synthetic allowance
    is_development = allow_synthetic or (exp_cfg.mode == "development")
    mode_str = "development" if is_development else exp_cfg.mode
    require_real = not is_development

    print(f"Task Name              : {task_name}")
    print(f"Execution Mode         : {mode_str.upper()}")
    print(f"Allow Synthetic Data   : {is_development}")
    print(f"Output Directory       : {output_path}")
    print(f"LoRA Target Modules    : {lora_cfg.target_modules}")
    print(f"LoRA Rank / Alpha      : r={lora_cfg.rank}, alpha={lora_cfg.alpha}")
    print(f"Epochs                 : {training_cfg.num_train_epochs}")
    print(f"Batch Size / GradAcc   : {training_cfg.per_device_train_batch_size} / {training_cfg.gradient_accumulation_steps}")
    print(f"Learning Rate          : {training_cfg.learning_rate}")
    print("-" * 70)

    # 2. Hardware Inspection
    hw_info = detect_hardware()
    print(f"Hardware Detected      : {hw_info['device_name']} ({hw_info['device_type'].upper()})")
    print(f"Available Memory (GB)  : {hw_info['free_memory_gb']:.2f} free / {hw_info['total_memory_gb']:.2f} total")
    if hw_info["cuda_available"]:
        print(f"CUDA Version           : {hw_info.get('cuda_version', 'N/A')}")
        target_device = "cuda" if device_override is None else device_override
    else:
        print("CUDA Warning           : No CUDA GPU detected. Running on CPU.")
        target_device = "cpu" if device_override is None else device_override
        if mode_str == "final":
            print("[ERROR] Cannot run 'final' production training on CPU without CUDA GPU!")
            print("Please execute on a CUDA-enabled GPU (e.g., Google Colab) or run with '--development --allow-synthetic'.")
            return False

    set_seed(training_cfg.seed)

    # 3. Load Task Dataset
    print(f"\n[1/4] Loading and auditing dataset partitions for {task_name}...")
    task_cfg_entry = exp_cfg.tasks.get(task_name, {})
    splits_cfg = task_cfg_entry.get("splits", {})
    calib_count = splits_cfg.get("calibration", 128)
    val_count = splits_cfg.get("validation", 100)
    eval_count = splits_cfg.get("evaluation", 200)

    try:
        split_dict, dataset_meta = load_task_dataset(
            task_name=task_name,
            calib_count=calib_count,
            val_count=val_count,
            eval_count=eval_count,
            seed=exp_cfg.seed,
            mode=mode_str,
            allow_synthetic=is_development,
            require_real_huggingface_data=require_real,
            metadata_save_dir=str(output_path)
        )
    except Exception as e:
        print(f"[FATAL] Failed to load dataset: {e}")
        return False

    # 4. Strict Split Disjointness & Isolation Verification
    print("\n[2/4] Verifying dataset isolation and strict non-leakage...")
    train_split = split_dict["train"]
    val_split = split_dict["validation"]
    calib_split = split_dict["calibration"]
    eval_split = split_dict["evaluation"]

    print(f"  Training Split Size    : {len(train_split)} examples")
    print(f"  Validation Split Size  : {len(val_split)} examples (used ONLY for early stopping / best checkpoint)")
    print(f"  Calibration Split Size : {len(calib_split)} examples (HELD OUT - strictly NOT used in training)")
    print(f"  Evaluation Split Size  : {len(eval_split)} examples (HELD OUT - strictly NOT used in training)")

    # Assert zero overlap between train and eval/calib using cryptographic hashes
    train_contents = {compute_example_content_sha256(ex, task_name) for ex in train_split}
    calib_contents = {compute_example_content_sha256(ex, task_name) for ex in calib_split}
    eval_contents = {compute_example_content_sha256(ex, task_name) for ex in eval_split}

    overlap_train_eval = train_contents.intersection(eval_contents)
    overlap_train_calib = train_contents.intersection(calib_contents)
    if overlap_train_eval:
        raise RuntimeError(f"DATA LEAKAGE: {len(overlap_train_eval)} examples overlap between train and eval splits!")
    if overlap_train_calib:
        raise RuntimeError(f"DATA LEAKAGE: {len(overlap_train_calib)} examples overlap between train and calib splits!")
    print("  Split Isolation Verification: PASSED (Zero data leakage).")

    if dry_run:
        print("\n[DRY RUN] Preflight checks passed. Exiting without training.")
        return True

    # 5. Load Base Model and Tokenizer
    print("\n[3/4] Loading base model and tokenizer...")
    model_name = exp_cfg.base_model
    prefer_mock = is_development and (not hw_info["cuda_available"])

    # 5b. Safe directory reset: restart cleanly to prevent mixing partial, interrupted, or stale artifacts
    if not dry_run and output_path.exists():
        existing_artifacts = list(output_path.glob("*"))
        if existing_artifacts:
            print(f"  [Clean Restart] Resetting '{output_path}' ({len(existing_artifacts)} stale/partial files removed to ensure clean run)...")
            import shutil
            shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
    
    model, tokenizer, _ = load_model_and_tokenizer(
        model_name_or_path=model_name,
        device=target_device,
        torch_dtype=training_cfg.torch_dtype,
        prefer_mock=prefer_mock,
        local_files_only=False
    )

    # 6. Instantiate Trainer and Train
    print("\n[4/4] Launching LoRA Trainer...")
    trainer = LoRATrainer(
        model=model,
        tokenizer=tokenizer,
        task_name=task_name,
        train_dataset=train_split,
        val_dataset=val_split,
        training_config=training_cfg,
        lora_config=lora_cfg,
        output_dir=str(output_path),
        device=target_device
    )

    train_summary = trainer.train()
    print("\n" + "=" * 70)
    print(f" Training complete for '{task_name}' ")
    print(f" Best Validation Loss : {train_summary['best_eval_loss']:.4f}")
    print(f" Duration (seconds)   : {train_summary['total_duration_sec']}")
    print(f" Saved Artifacts to   : {train_summary['output_dir']}")
    print("=" * 70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an independent LoRA adapter for a specific task.")
    parser.add_argument("--task", type=str, required=True, choices=["boolq", "gsm8k", "arc_easy"], help="Task name")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training.yaml")
    parser.add_argument("--experiment-config", type=str, default="configs/experiment.yaml", help="Path to experiment.yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--development", "--allow-synthetic", action="store_true", help="Allow synthetic fallback")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Override device ('cpu' or 'cuda')")
    parser.add_argument("--dry-run", "--preflight", action="store_true", help="Run preflight validation without training")
    args = parser.parse_args()

    success = train_single_adapter(
        task_name=args.task,
        training_config_path=args.config,
        experiment_config_path=args.experiment_config,
        output_dir=args.output_dir,
        allow_synthetic=args.development,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device_override=args.device,
        dry_run=args.dry_run
    )
    if not success:
        sys.exit(1)
