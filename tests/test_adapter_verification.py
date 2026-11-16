"""
Unit tests for Adapter Verification, Divergence Auditing, and Health Checks.
"""

import json
import pytest
import torch
from pathlib import Path
from scripts.verify_adapters import (
    verify_single_adapter,
    run_cross_task_check,
)
from src.training.checkpoint_manager import save_adapter_checkpoint


@pytest.fixture
def mock_adapters_dir(tmp_path):
    """Creates a temporary mock adapters directory with 3 distinct trained adapters."""
    tasks = ["boolq", "gsm8k", "arc_easy"]
    for idx, t in enumerate(tasks):
        t_dir = tmp_path / t
        # Create non-zero weights distinct for each task
        torch.manual_seed(idx * 100 + 7)
        sd = {
            "base_model.model.q_proj.lora_A.weight": torch.randn(8, 64) * 0.1,
            "base_model.model.q_proj.lora_B.weight": torch.randn(64, 8) * 0.1 + (idx + 1),
            "base_model.model.v_proj.lora_A.weight": torch.randn(8, 64) * 0.1,
            "base_model.model.v_proj.lora_B.weight": torch.randn(64, 8) * 0.1 + (idx + 1),
        }
        cfg = {"rank": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]}
        save_adapter_checkpoint(
            output_dir=str(t_dir),
            adapter_state_dict=sd,
            lora_config=cfg,
            base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
            training_metadata={"task_name": t, "eval_loss": 0.5},
            training_log={"steps": []},
            validation_metrics={"best_eval_loss": 0.5}
        )
    return tmp_path


def test_verify_single_adapter_passes(mock_adapters_dir):
    """Verify that a properly constructed adapter directory passes verification."""
    res = verify_single_adapter(mock_adapters_dir / "boolq", "boolq")
    assert res["passed"] is True
    assert res["config_valid"] is True
    assert res["weights_valid"] is True
    assert res["parameter_count"] > 0
    assert len(res["diagnostics"]) == 0


def test_verify_single_adapter_catches_invalid_rank(tmp_path):
    """Verify that rank mismatch is detected and rejected."""
    bad_dir = tmp_path / "bad_rank"
    sd = {
        "base_model.model.q_proj.lora_A.weight": torch.randn(16, 64),
        "base_model.model.q_proj.lora_B.weight": torch.randn(64, 16),
    }
    cfg = {"rank": 16, "alpha": 16, "target_modules": ["q_proj", "v_proj"]}
    save_adapter_checkpoint(
        output_dir=str(bad_dir),
        adapter_state_dict=sd,
        lora_config=cfg,
        base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        training_metadata={"task_name": "bad"},
        training_log={},
        validation_metrics={}
    )

    res = verify_single_adapter(bad_dir, "bad", expected_rank=8)
    assert res["passed"] is False
    assert any("Rank mismatch" in d for d in res["diagnostics"])


def test_cross_task_distinctness_check(mock_adapters_dir):
    """Verify cross-task distinctness check detects differences between tasks."""
    task_res = {}
    for t in ["boolq", "gsm8k", "arc_easy"]:
        task_res[t] = verify_single_adapter(mock_adapters_dir / t, t)

    divergence = run_cross_task_check(task_res)
    assert divergence["all_distinct"] is True
    assert len(divergence["comparisons"]) == 3  # (boolq, gsm8k), (boolq, arc), (gsm8k, arc)


def test_cross_task_catches_duplicated_adapters(tmp_path):
    """Verify cross-task check flags identical/copied adapters."""
    t1_dir = tmp_path / "t1"
    t2_dir = tmp_path / "t2"

    shared_sd = {
        "base_model.model.q_proj.lora_A.weight": torch.randn(8, 64),
        "base_model.model.q_proj.lora_B.weight": torch.randn(64, 8) + 1.0,
    }
    cfg = {"rank": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]}

    save_adapter_checkpoint(str(t1_dir), shared_sd, cfg, "base", {}, {}, {})
    save_adapter_checkpoint(str(t2_dir), shared_sd, cfg, "base", {}, {}, {})

    task_res = {
        "t1": verify_single_adapter(t1_dir, "t1"),
        "t2": verify_single_adapter(t2_dir, "t2")
    }

    divergence = run_cross_task_check(task_res)
    assert divergence["all_distinct"] is False
