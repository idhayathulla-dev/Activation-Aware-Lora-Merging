"""
Unit tests for Split Integrity, Disjointness, and Calibration Set Verification.
"""

import pytest
from src.dataset_loader import (
    load_task_dataset,
    verify_split_disjointness,
    partition_dataset_indices,
)


@pytest.mark.parametrize("task_name", ["boolq", "gsm8k", "arc_easy"])
def test_partition_sizes_and_disjointness(task_name):
    """Verify partition sizes (train: 172, calib: 128, val: 100, eval: 200) and zero leakage."""
    splits, meta = load_task_dataset(
        task_name=task_name,
        calib_count=128,
        val_count=100,
        eval_count=200,
        mode="development",
        allow_synthetic=True,
        require_real_huggingface_data=False
    )

    train_set = splits["train"]
    calib_set = splits["calibration"]
    val_set = splits["validation"]
    eval_set = splits["evaluation"]

    # Check partition counts
    assert len(calib_set) == 128
    assert len(val_set) == 100
    assert len(eval_set) == 200
    assert len(train_set) == 172

    # Check content disjointness
    disjoint_res = verify_split_disjointness(splits)
    assert disjoint_res["integrity_passed"] is True, f"Leakage detected: {disjoint_res['violations']}"
    assert len(disjoint_res["violations"]) == 0


def test_split_disjointness_catches_leakage():
    """Verify verify_split_disjointness flags overlapping examples."""
    leaked_example = {"input_text": "Shared question", "label": "answer"}
    splits = {
        "train": [leaked_example, {"input_text": "Q1", "label": "A1"}],
        "validation": [{"input_text": "Q2", "label": "A2"}],
        "calibration": [{"input_text": "Q3", "label": "A3"}],
        "evaluation": [leaked_example, {"input_text": "Q4", "label": "A4"}],
    }

    res = verify_split_disjointness(splits)
    assert res["integrity_passed"] is False
    assert any("train" in v and "evaluation" in v for v in res["violations"])


def test_partition_indices_exhaustive_and_disjoint():
    """Verify index partitioning covers exact ranges without overlaps."""
    total_samples = 600
    indices = partition_dataset_indices(
        total_count=total_samples,
        calib_count=128,
        val_count=100,
        eval_count=200,
        seed=42
    )

    train_idx = set(indices["train"])
    calib_idx = set(indices["calibration"])
    val_idx = set(indices["validation"])
    eval_idx = set(indices["evaluation"])

    # Disjointness
    assert len(train_idx.intersection(calib_idx)) == 0
    assert len(train_idx.intersection(val_idx)) == 0
    assert len(train_idx.intersection(eval_idx)) == 0
    assert len(calib_idx.intersection(val_idx)) == 0
    assert len(calib_idx.intersection(eval_idx)) == 0
    assert len(val_idx.intersection(eval_idx)) == 0

    # Counts
    assert len(calib_idx) == 128
    assert len(val_idx) == 100
    assert len(eval_idx) == 200
    assert len(train_idx) == total_samples - 128 - 100 - 200
