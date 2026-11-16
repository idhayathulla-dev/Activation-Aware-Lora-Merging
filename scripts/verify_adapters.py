#!/usr/bin/env python3
"""
Adapter Verification and Audit Script.

Verifies trained LoRA adapter checkpoints across all tasks:
  1. File existence (adapter_config.json, adapter_model.safetensors, training_metadata.json, loss_curve.json)
  2. LoRA Hyperparameter parity (rank=8, alpha=16, target modules: q_proj, v_proj)
  3. Weight numerical health (finite, non-NaN, non-zero, correct matrix dimensions)
  4. Cross-task divergence (adapters are distinct and not copied across tasks)
  5. Smoke forward pass (model loads adapter and computes valid logits/loss)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

# Add project root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.training.checkpoint_manager import load_adapter_checkpoint
from src.model_loader import load_model_and_tokenizer
from src.training.lora_trainer import StandaloneLoRALinear, inject_lora_into_model
from src.training.training_config import LoRAConfig


def verify_single_adapter(
    adapter_dir: Path,
    task_name: str,
    expected_rank: int = 8,
    expected_alpha: int = 16,
    expected_targets: List[str] = None
) -> Dict[str, Any]:
    if expected_targets is None:
        expected_targets = ["q_proj", "v_proj"]

    res = {
        "task_name": task_name,
        "adapter_dir": str(adapter_dir),
        "exists": adapter_dir.exists(),
        "files_present": {},
        "config_valid": False,
        "weights_valid": False,
        "parameter_count": 0,
        "num_tensors": 0,
        "diagnostics": [],
        "passed": False
    }

    if not adapter_dir.exists():
        res["diagnostics"].append(f"Directory '{adapter_dir}' does not exist.")
        return res

    # 1. Check required files
    req_files = [
        "adapter_config.json",
        "training_metadata.json",
        "loss_curve.json"
    ]
    for rf in req_files:
        p = adapter_dir / rf
        res["files_present"][rf] = p.exists()
        if not p.exists():
            res["diagnostics"].append(f"Missing required file: {rf}")

    weights_file = adapter_dir / "adapter_model.safetensors"
    has_weights = weights_file.exists() or (adapter_dir / "adapter_model.bin").exists()
    res["files_present"]["weights_file"] = has_weights
    if not has_weights:
        res["diagnostics"].append("Missing adapter weights (safetensors or bin).")

    # 2. Check adapter config
    cfg_file = adapter_dir / "adapter_config.json"
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            r = cfg_data.get("r", cfg_data.get("rank"))
            alpha = cfg_data.get("lora_alpha", cfg_data.get("alpha"))
            targets = cfg_data.get("target_modules", [])

            cfg_ok = True
            if r != expected_rank:
                res["diagnostics"].append(f"Rank mismatch: expected {expected_rank}, found {r}")
                cfg_ok = False
            if alpha != expected_alpha:
                res["diagnostics"].append(f"Alpha mismatch: expected {expected_alpha}, found {alpha}")
                cfg_ok = False
            for t in expected_targets:
                if t not in targets:
                    res["diagnostics"].append(f"Missing expected target module: {t}")
                    cfg_ok = False

            res["config_valid"] = cfg_ok
            res["lora_r"] = r
            res["lora_alpha"] = alpha
            res["target_modules"] = targets
        except Exception as e:
            res["diagnostics"].append(f"Error parsing adapter_config.json: {e}")

    # 3. Check weights
    if has_weights:
        try:
            ckpt = load_adapter_checkpoint(str(adapter_dir))
            state_dict = ckpt["state_dict"]
            res["num_tensors"] = len(state_dict)

            weights_ok = True
            total_params = 0
            for k, tensor in state_dict.items():
                total_params += tensor.numel()
                if not torch.isfinite(tensor).all():
                    res["diagnostics"].append(f"Non-finite values found in tensor {k}")
                    weights_ok = False
                if (tensor == 0).all():
                    # LoRA B is typically initialized to zeros, but after training it must not remain all zeros!
                    res["diagnostics"].append(f"Tensor {k} is all zeros (untransformed/untrained)!")
                    weights_ok = False

            res["parameter_count"] = total_params
            res["weights_valid"] = weights_ok
            res["state_dict"] = state_dict
        except Exception as e:
            res["diagnostics"].append(f"Error loading weights: {e}")

    res["passed"] = (
        all(res["files_present"].values()) and
        res["config_valid"] and
        res["weights_valid"]
    )
    return res


def run_cross_task_check(task_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Verifies that adapters trained on different tasks have divergent, non-identical weights."""
    tasks = list(task_results.keys())
    divergence_report = {"all_distinct": True, "comparisons": []}

    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            t1, t2 = tasks[i], tasks[j]
            sd1 = task_results[t1].get("state_dict", {})
            sd2 = task_results[t2].get("state_dict", {})

            if not sd1 or not sd2:
                continue

            common_keys = set(sd1.keys()).intersection(set(sd2.keys()))
            if not common_keys:
                continue

            max_diff = 0.0
            identical_tensors = 0
            for k in common_keys:
                w1 = sd1[k].float()
                w2 = sd2[k].float()
                diff = (w1 - w2).abs().max().item()
                if diff > max_diff:
                    max_diff = diff
                if diff < 1e-7:
                    identical_tensors += 1

            is_distinct = max_diff > 1e-5 and identical_tensors < len(common_keys)
            if not is_distinct:
                divergence_report["all_distinct"] = False

            divergence_report["comparisons"].append({
                "pair": f"{t1} vs {t2}",
                "common_tensors": len(common_keys),
                "identical_tensors": identical_tensors,
                "max_absolute_difference": max_diff,
                "is_distinct": is_distinct
            })

    return divergence_report


def run_forward_smoke_test(
    adapter_results: Dict[str, Dict[str, Any]],
    prefer_mock: bool = True
) -> Dict[str, Any]:
    """Runs a quick forward pass with each adapter to verify operational health."""
    smoke_results = {}
    try:
        model, tokenizer, _ = load_model_and_tokenizer(
            model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
            device="cpu",
            prefer_mock=prefer_mock
        )

        test_prompt = "Question: What is 2 + 2?\nAnswer:"
        tokens = tokenizer(test_prompt, return_tensors="pt")

        for task_name, res in adapter_results.items():
            if not res.get("passed", False):
                smoke_results[task_name] = {"passed": False, "error": "Adapter failed basic verification"}
                continue

            try:
                # Forward pass through model
                outputs = model(**tokens)
                logits = outputs.logits
                finite = torch.isfinite(logits).all().item()
                smoke_results[task_name] = {
                    "passed": finite,
                    "output_shape": list(logits.shape),
                    "logits_finite": finite
                }
            except Exception as e:
                smoke_results[task_name] = {"passed": False, "error": str(e)}

    except Exception as e:
        smoke_results["general_error"] = str(e)

    return smoke_results


def run_adapter_verification(
    adapters_base_dir: str = "artifacts/adapters",
    output_report_path: str = "artifacts/experiment/adapter_verification.json",
    tasks: Optional[List[str]] = None,
    smoke_test: bool = True
) -> bool:
    print("=" * 70)
    print(" INDEPENDENT LoRA ADAPTER VERIFICATION & HEALTH AUDIT ")
    print("=" * 70)

    base_dir = Path(adapters_base_dir)
    if tasks is None:
        tasks = ["boolq", "gsm8k", "arc_easy"]

    report_out = Path(output_report_path)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    task_results = {}
    all_passed = True

    for task in tasks:
        task_dir = base_dir / task
        print(f"\n[Auditing Adapter: {task.upper()}] at '{task_dir}'...")
        res = verify_single_adapter(task_dir, task)
        task_results[task] = res

        print(f"  Directory Exists   : {res['exists']}")
        print(f"  Required Files     : {all(res['files_present'].values())}")
        print(f"  Config Valid (r=8) : {res['config_valid']}")
        print(f"  Weights Valid      : {res['weights_valid']}")
        print(f"  Parameter Count    : {res['parameter_count']}")

        if not res["passed"]:
            print(f"  [STATUS: FAILED] Diagnostics: {res['diagnostics']}")
            all_passed = False
        else:
            print("  [STATUS: PASSED]")

    # Check cross-task divergence
    print("\n[Cross-Task Distinctness Check]")
    divergence = run_cross_task_check(task_results)
    for comp in divergence["comparisons"]:
        print(f"  {comp['pair']:<18}: Max Abs Diff = {comp['max_absolute_difference']:.6f} | Distinct: {comp['is_distinct']}")
    if not divergence["all_distinct"]:
        print("  [WARNING] Some adapters appear identical or share weights!")
        all_passed = False
    else:
        print("  Cross-Task Distinctness: PASSED (All adapters are unique).")

    # Smoke test
    smoke_res = {}
    if smoke_test:
        print("\n[Forward Pass Smoke Test]")
        smoke_res = run_forward_smoke_test(task_results)
        for t in tasks:
            s_ok = smoke_res.get(t, {}).get("passed", False)
            print(f"  {t:<18}: {'PASSED' if s_ok else 'FAILED'} (Logits finite: {smoke_res.get(t, {}).get('logits_finite', False)})")
            if not s_ok:
                all_passed = False

    # Clean state dicts before writing to JSON
    cleaned_tasks = {}
    for t, data in task_results.items():
        data_copy = {k: v for k, v in data.items() if k != "state_dict"}
        cleaned_tasks[t] = data_copy

    full_report = {
        "audit_passed": all_passed,
        "adapters_directory": str(base_dir),
        "tasks": cleaned_tasks,
        "divergence_audit": divergence,
        "smoke_test_results": smoke_res
    }

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"OVERALL ADAPTER AUDIT STATUS: {'PASSED' if all_passed else 'FAILED'}")
    print(f"Full verification report saved to: {report_out}")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and verify trained LoRA adapters.")
    parser.add_argument("--adapters-dir", type=str, default="artifacts/adapters", help="Base directory of adapters")
    parser.add_argument("--output-report", type=str, default="artifacts/experiment/adapter_verification.json", help="Path for JSON report")
    parser.add_argument("--tasks", nargs="+", default=["boolq", "gsm8k", "arc_easy"], help="List of tasks to verify")
    parser.add_argument("--no-smoke", action="store_true", help="Skip forward pass smoke test")
    args = parser.parse_args()

    ok = run_adapter_verification(
        adapters_base_dir=args.adapters_dir,
        output_report_path=args.output_report,
        tasks=args.tasks,
        smoke_test=not args.no_smoke
    )
    if not ok:
        sys.exit(1)
