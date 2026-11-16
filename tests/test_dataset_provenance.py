"""
Unit tests for Dataset Provenance, Authenticity Enforcement, and Cryptographic Fingerprints.
"""

import pytest
import hashlib
from src.dataset_loader import (
    load_task_dataset,
    compute_example_content_sha256,
    compute_partition_content_sha256,
    verify_split_disjointness,
    generate_mock_boolq_data,
    generate_mock_gsm8k_data,
    generate_mock_arc_data,
)


def test_content_hash_deterministic():
    """Verify SHA-256 fingerprint is identical for identical content."""
    ex1 = {"input_text": "Is water wet?", "label": "yes", "metadata": {"task": "boolq"}}
    ex2 = {"input_text": "Is water wet?", "label": "yes", "metadata": {"task": "boolq"}}

    h1 = compute_example_content_sha256(ex1)
    h2 = compute_example_content_sha256(ex2)

    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hexadecimal length


def test_content_hash_sensitivity():
    """Verify SHA-256 fingerprint changes when content or label changes."""
    base_ex = {"input_text": "Solve 2 + 2", "label": "4"}
    diff_text = {"input_text": "Solve 2 + 3", "label": "4"}
    diff_label = {"input_text": "Solve 2 + 2", "label": "5"}

    h_base = compute_example_content_sha256(base_ex)
    h_diff_text = compute_example_content_sha256(diff_text)
    h_diff_label = compute_example_content_sha256(diff_label)

    assert h_base != h_diff_text
    assert h_base != h_diff_label
    assert h_diff_text != h_diff_label


def test_partition_content_fingerprint():
    """Verify aggregated partition fingerprint changes if examples change."""
    examples_a = [
        {"input_text": f"Question {i}", "label": str(i)} for i in range(10)
    ]
    examples_b = [
        {"input_text": f"Question {i}", "label": str(i)} for i in range(10)
    ]
    examples_c = [
        {"input_text": f"Question {i}", "label": str(i + 1)} for i in range(10)
    ]

    hash_a = compute_partition_content_sha256(examples_a)
    hash_b = compute_partition_content_sha256(examples_b)
    hash_c = compute_partition_content_sha256(examples_c)

    assert hash_a == hash_b
    assert hash_a != hash_c


def test_final_mode_rejects_synthetic_data():
    """Verify that mode='final' with require_real_huggingface_data strictly raises RuntimeError when synthetic mock data is requested."""
    with pytest.raises(RuntimeError) as exc_info:
        load_task_dataset(
            task_name="boolq",
            mode="final",
            use_mock=True,
            allow_synthetic=False,
            require_real_huggingface_data=True
        )
    err_str = str(exc_info.value)
    assert "FINAL MODE" in err_str or "Synthetic fallback is forbidden" in err_str or "PROVENANCE ERROR" in err_str


def test_development_mode_allows_synthetic():
    """Verify that mode='development' permits synthetic mock data and marks metadata accurately."""
    splits, meta = load_task_dataset(
        task_name="boolq",
        mode="development",
        allow_synthetic=True,
        require_real_huggingface_data=False
    )
    assert meta["is_synthetic"] is True
    assert meta["source"] == "synthetic_mock"
    assert "train" in splits
    assert "validation" in splits
    assert "calibration" in splits
    assert "evaluation" in splits
    assert len(splits["calibration"]) == 128
    assert "content_fingerprints" in meta
