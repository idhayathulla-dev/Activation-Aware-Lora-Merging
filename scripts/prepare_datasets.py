"""
Dataset Preparation & Verification Script (Stage D).
Partitions BoolQ, GSM8K, and ARC-Easy into strictly disjoint splits,
verifies zero data leakage, and outputs representative formatted examples.
"""

import json
from pathlib import Path

from src.dataset_loader import load_task_dataset
from src.data_preprocessing import extract_prompt_and_target, preprocess_dataset
from src.model_loader import load_model_and_tokenizer


def main():
    print("=" * 70)
    print(" STAGE D: DATASET PARTITIONING & PREPROCESSING VERIFICATION ")
    print("=" * 70)

    tasks = ["boolq", "gsm8k", "arc_easy"]
    results_dir = Path("results/processed")
    results_dir.mkdir(parents=True, exist_ok=True)

    _, tokenizer, _ = load_model_and_tokenizer(prefer_mock=True)

    summary_report = {}

    for task in tasks:
        print(f"\n[{task.upper()}] Loading and partitioning...")
        splits, meta = load_task_dataset(
            task_name=task,
            calib_count=128,
            val_count=100,
            eval_count=200,
            seed=42,
            use_mock=True,
            metadata_save_dir="results/processed"
        )

        print(f"  Total Examples     : {meta['total_examples']}")
        print(f"  Train Size         : {meta['split_sizes']['train']}")
        print(f"  Calibration Size   : {meta['split_sizes']['calibration']} (Target: 128)")
        print(f"  Validation Size    : {meta['split_sizes']['validation']}")
        print(f"  Evaluation Size    : {meta['split_sizes']['evaluation']}")
        print(f"  Calib Fingerprint  : {meta['fingerprints']['calibration']}")
        print(f"  Eval Fingerprint   : {meta['fingerprints']['evaluation']}")

        # Preprocess calibration and train examples
        train_processed = preprocess_dataset(
            splits["train"][:50], task_name=task, tokenizer=tokenizer, is_training=True
        )
        calib_processed = preprocess_dataset(
            splits["calibration"][:50], task_name=task, tokenizer=tokenizer, is_training=False
        )

        sample_ex = splits["train"][0]
        prompt, target, raw_target = extract_prompt_and_target(sample_ex, task)

        print(f"\n  --- Representative Formatted Example ({task}) ---")
        print("  [PROMPT]:")
        for line in prompt.strip().split("\n")[:4]:
            print(f"    {line}")
        if len(prompt.strip().split("\n")) > 4:
            print("    ...")
        print("  [TARGET]:")
        for line in target.strip().split("\n")[:3]:
            print(f"    {line}")
        print(f"  [RAW TARGET]: {raw_target}")

        summary_report[task] = {
            "metadata": meta,
            "sample_prompt": prompt,
            "sample_target": target,
            "sample_raw_target": str(raw_target),
            "train_stats": train_processed["statistics"],
            "calib_stats": calib_processed["statistics"],
        }

    report_path = Path("results/reports/stage_d_dataset_summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\n[Summary] Detailed dataset split audit saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
