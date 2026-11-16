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
    if hw["cuda_available"]:
        print(f"GPU Total VRAM     : {hw.get('gpu_total_vram_gb', 0.0):.2f} GB")
        print(f"GPU Free VRAM      : {hw.get('gpu_free_vram_gb', 0.0):.2f} GB")
        print(f"VRAM Note          : Note: The 6 GB threshold is a lower-bound preflight heuristic.")
        print(f"                     Actual VRAM requirement depends on batching (b=4, accum=4),")
        print(f"                     dynamic sequence padding (up to 512), and precision.")
        print(f"                     Standard Colab T4 (15-16 GB) or A100 is recommended.")

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
    backup_dir: str = None,
) -> bool:
    if preflight:
        return run_preflight_audit(training_config_path, experiment_config_path, allow_synthetic)

    tasks = ["boolq", "gsm8k", "arc_easy"]
    start_all = time.time()
    results = {}

    print("=" * 70)
    print(" SEQUENTIAL THREE-TASK LoRA TRAINING PIPELINE ")
    print(f" Tasks: {', '.join(tasks)} ")
    if backup_dir:
        print(f" Immediate Backup Dir: {backup_dir}")
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

        # Immediate verification of the trained adapter
        print(f"\n[Immediate Verification] Auditing '{task}' adapter integrity...")
        from scripts.verify_adapters import verify_single_adapter
        src_task_dir = Path("artifacts/adapters") / task
        v_res = verify_single_adapter(src_task_dir, task)
        if not v_res["passed"]:
            print(f"[ERROR] Immediate verification failed for task '{task}': {v_res.get('diagnostics')}. Halting pipeline.")
            return False
        print(f"  [Verification PASSED] r={v_res.get('lora_r')}, alpha={v_res.get('lora_alpha')}, params={v_res.get('parameter_count')}, sha256={v_res.get('adapter_sha256', 'N/A')[:16]}...")

        # Immediate persistent backup after each verified task
        if backup_dir:
            import shutil
            b_path = Path(backup_dir) / "adapters" / task
            b_path.parent.mkdir(parents=True, exist_ok=True)
            if b_path.exists():
                shutil.rmtree(b_path)
            shutil.copytree(src_task_dir, b_path)
            print(f"[Backup Certified] Immediately mirrored verified '{task}' checkpoint to: {b_path}")

    elapsed = round(time.time() - start_all, 2)
    print("\n" + "=" * 70)
    print(f" ALL 3 ADAPTERS TRAINED & INDIVIDUALLY VERIFIED IN {elapsed}s ")
    print(" Results Summary:")
    for t, res in results.items():
        print(f"  - {t:<12}: {'SUCCESS' if res else 'FAILED'}")
    print("=" * 70)

    # Full joint verification across all tasks
    print("\n[Running Joint Multi-Adapter Verification & Cross-Task Divergence Check]...")
    from scripts.verify_adapters import run_adapter_verification
    full_verif_passed = run_adapter_verification(
        adapters_base_dir="artifacts/adapters",
        tasks=tasks,
        device=device or "cpu"
    )
    if not full_verif_passed:
        print("[ERROR] Joint multi-adapter verification failed!")
        return False

    # Backup full verification report to backup_dir if provided
    if backup_dir:
        import shutil
        exp_src = Path("artifacts/experiment")
        if exp_src.exists():
            exp_dst = Path(backup_dir) / "experiment"
            exp_dst.mkdir(parents=True, exist_ok=True)
            for f in exp_src.glob("*"):
                shutil.copy2(f, exp_dst / f.name)
            print(f"[Backup Certified] Mirrored experiment verification reports to: {exp_dst}")

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
    parser.add_argument("--backup-dir", type=str, default=None, help="Directory to immediately mirror completed checkpoints")
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
        preflight=args.preflight,
        backup_dir=args.backup_dir
    )
    if not ok:
        sys.exit(1)
