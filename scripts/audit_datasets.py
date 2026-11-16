#!/usr/bin/env python3
"""
Dataset Provenance and Split Integrity Audit Script.

Verifies authenticity, checks split disjointness, computes content-based SHA-256 fingerprints,
and exports artifacts/provenance/ reports.
"""

import argparse
import json
import sys
from pathlib import Path
import yaml

# Add project root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.dataset_loader import load_task_dataset


def run_dataset_audit(config_path: str = "configs/experiment.yaml", allow_synthetic_override: bool = False) -> bool:
    print("=" * 70)
    print(" DATASET AUTHENTICITY & PROVENANCE INTEGRITY AUDIT ")
    print("=" * 70)

    cfg_file = Path(config_path)
    if not cfg_file.exists():
        alt_cfg = repo_root / config_path
        if alt_cfg.exists():
            cfg_file = alt_cfg
        else:
            print(f"Error: Config file '{config_path}' not found.")
            sys.exit(1)

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg.get("experiment", {})
    mode = exp_cfg.get("mode", "final")
    allow_synthetic = exp_cfg.get("allow_synthetic", False) or allow_synthetic_override
    if allow_synthetic_override:
        mode = "development"
    require_real = exp_cfg.get("require_real_huggingface_data", True) and not allow_synthetic
    seed = exp_cfg.get("seed", 42)
    artifacts_dir = repo_root / cfg.get("artifacts_dir", "artifacts")
    prov_dir = artifacts_dir / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)

    print(f"Audit Mode                  : {mode.upper()}")
    print(f"Allow Synthetic Data        : {allow_synthetic}")
    print(f"Require Real HuggingFace    : {require_real}")
    print(f"Experiment Seed             : {seed}")
    print(f"Provenance Output Directory : {prov_dir}")
    print("-" * 70)

    tasks_cfg = cfg.get("tasks", {})
    all_manifest = {}
    integrity_summary = {}
    all_passed = True

    for task_name, t_info in tasks_cfg.items():
        print(f"\n[Auditing Task: {task_name.upper()}]")
        splits_cfg = t_info.get("splits", {})
        calib_n = splits_cfg.get("calibration", 128)
        val_n = splits_cfg.get("validation", 100)
        eval_n = splits_cfg.get("evaluation", 200)

        try:
            split_data, meta = load_task_dataset(
                task_name=task_name,
                calib_count=calib_n,
                val_count=val_n,
                eval_count=eval_n,
                seed=seed,
                mode=mode,
                allow_synthetic=allow_synthetic,
                require_real_huggingface_data=require_real,
                metadata_save_dir=None
            )

            is_synth = meta.get("is_synthetic", False)
            status_str = "SYNTHETIC_MOCK" if is_synth else "AUTHENTIC_HF"
            print(f"  Source Type       : {status_str}")
            print(f"  Dataset Identity  : {meta.get('dataset_name')} (config: {meta.get('dataset_config')})")
            print(f"  Resolved Commit   : {meta.get('dataset_revision')}")
            print(f"  Preproc Version   : {meta.get('preprocessing_version')}")
            print(f"  Original Sources  : {meta.get('original_source_counts')}")
            print(f"  Excluded Dups     : {meta.get('excluded_duplicates_count')}")
            if meta.get('excluded_duplicates_count', 0) > 0:
                for d in meta.get('excluded_duplicates', []):
                    print(f"    - Excluded train index {d.get('train_index')}: collides with {d.get('matched_split')}[{d.get('matched_index')}] | '{d.get('question')}'")
            print(f"  Final Split Sizes : {meta.get('split_sizes')}")
            print(f"  Content SHA-256 (Train) : {meta['content_fingerprints']['train'][:16]}...")
            print(f"  Content SHA-256 (Calib) : {meta['content_fingerprints']['calibration'][:16]}...")
            print(f"  Content SHA-256 (Val)   : {meta['content_fingerprints']['validation'][:16]}...")
            print(f"  Content SHA-256 (Eval)  : {meta['content_fingerprints']['evaluation'][:16]}...")

            if mode == "final" and is_synth:
                print(f"  [ERROR] Synthetic data detected in FINAL mode for task '{task_name}'!")
                all_passed = False

            audit_res = meta.get("integrity_audit", {})
            if not audit_res.get("integrity_passed", False):
                print(f"  [ERROR] Split leakage detected in task '{task_name}': {audit_res.get('violations')}")
                all_passed = False
            else:
                print("  Split Leakage Check: PASSED (Zero content overlap between partitions)")

            all_manifest[task_name] = meta
            integrity_summary[task_name] = {
                "dataset_name": meta.get("dataset_name"),
                "dataset_config": meta.get("dataset_config"),
                "dataset_revision": meta.get("dataset_revision"),
                "preprocessing_version": meta.get("preprocessing_version"),
                "source": meta["source"],
                "is_synthetic": is_synth,
                "original_source_counts": meta.get("original_source_counts"),
                "excluded_duplicates_count": meta.get("excluded_duplicates_count", 0),
                "excluded_duplicates": meta.get("excluded_duplicates", []),
                "split_sizes": meta["split_sizes"],
                "content_fingerprints": meta["content_fingerprints"],
                "integrity_passed": audit_res.get("integrity_passed", False),
                "violations": audit_res.get("violations", [])
            }

        except Exception as e:
            print(f"  [AUDIT FAILED] Exception for task '{task_name}': {e}")
            all_passed = False
            integrity_summary[task_name] = {
                "error": str(e),
                "integrity_passed": False
            }

    # Save manifest and reports
    manifest_path = prov_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_manifest, f, indent=2)

    report_json_path = prov_dir / "split_integrity_report.json"
    audit_report = {
        "audit_passed": all_passed,
        "mode": mode,
        "allow_synthetic": allow_synthetic,
        "tasks": integrity_summary
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    # Save Markdown report
    report_md_path = prov_dir / "split_integrity_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Authenticity and Benchmark Split Integrity Report\n\n")
        f.write(f"- **Overall Audit Status:** {'PASSED' if all_passed else 'FAILED'}\n")
        f.write(f"- **Execution Mode:** `{mode}`\n")
        f.write(f"- **Allow Synthetic:** `{allow_synthetic}`\n")
        f.write(f"- **Experiment Seed:** `{seed}`\n")
        f.write(f"- **Partition Policy:** Strict Benchmark Preservation (Train-only calibration/validation, zero eval truncation)\n\n")
        
        f.write("## 1. Task Benchmark & Partition Summary\n\n")
        f.write("| Task | Dataset (Config) | Resolved Git Commit SHA | Original Sources | Excluded Dups | Train | Calib | Val | Eval | Integrity |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for t, res in integrity_summary.items():
            if "split_sizes" in res:
                sz = res["split_sizes"]
                src_counts = ", ".join([f"{k}:{v}" for k, v in res.get("original_source_counts", {}).items()])
                cfg_str = f"{res.get('dataset_name')} ({res.get('dataset_config') or 'default'})"
                commit_str = res.get('dataset_revision', 'unknown')[:10]
                status = "PASSED" if res["integrity_passed"] else "FAILED"
                f.write(f"| {t} | {cfg_str} | `{commit_str}` | {src_counts} | {res.get('excluded_duplicates_count', 0)} | {sz['train']} | {sz['calibration']} | {sz['validation']} | {sz['evaluation']} | **{status}** |\n")
            else:
                f.write(f"| {t} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | **FAILED** ({res.get('error', 'unknown')}) |\n")

        f.write("\n## 2. Upstream Cross-Split Duplicate Exclusions\n\n")
        has_dups = any(res.get("excluded_duplicates_count", 0) > 0 for res in integrity_summary.values() if isinstance(res, dict))
        if has_dups:
            f.write("The following upstream duplicates were detected in training sources colliding with official benchmark evaluation/validation splits. In accordance with benchmark preservation protocol, official evaluation examples were kept intact and training duplicates were excluded from training/calibration pools:\n\n")
            for t, res in integrity_summary.items():
                if isinstance(res, dict) and res.get("excluded_duplicates_count", 0) > 0:
                    f.write(f"### Task: `{t}` ({res.get('excluded_duplicates_count')} duplicate(s) excluded)\n\n")
                    for d in res.get("excluded_duplicates", []):
                        f.write(f"- **Train Index {d.get('train_index')}** (`id`: `{d.get('train_id')}`) collides with `{d.get('matched_split')}` index {d.get('matched_index')} (`id`: `{d.get('matched_id')}`):\n")
                        f.write(f"  - Question Excerpt: *\"{d.get('question')}\"*\n")
                        f.write(f"  - Conflicting Target: `{d.get('is_conflicting')}` (Train: `{d.get('train_target')}`, Eval: `{d.get('eval_target')}`)\n\n")
        else:
            f.write("No cross-split duplicates detected across tasks.\n\n")

        f.write("## 3. Cryptographic Partition Content Fingerprints (SHA-256)\n\n")
        for t, res in integrity_summary.items():
            if isinstance(res, dict) and "content_fingerprints" in res:
                f.write(f"### Task: `{t}`\n\n")
                f.write(f"| Split | Number of Examples | Content SHA-256 Fingerprint |\n")
                f.write(f"|---|---|---|\n")
                sz = res.get("split_sizes", {})
                for p, h in res["content_fingerprints"].items():
                    f.write(f"| {p} | {sz.get(p, 'N/A')} | `{h}` |\n")
                f.write("\n")

    print("\n" + "=" * 70)
    print(f"Audit Status: {'ALL CHECKS PASSED' if all_passed else 'AUDIT FAILED'}")
    print(f"Manifest written to : {manifest_path}")
    print(f"JSON Report         : {report_json_path}")
    print(f"Markdown Report     : {report_md_path}")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Authenticity and Integrity Audit")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to experiment.yaml")
    parser.add_argument("--development", "--allow-synthetic", action="store_true", help="Allow synthetic fallback for offline audit")
    args = parser.parse_args()

    success = run_dataset_audit(args.config, allow_synthetic_override=args.development)
    if not success:
        sys.exit(1)
