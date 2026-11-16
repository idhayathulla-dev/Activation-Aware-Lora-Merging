"""
Unit tests for Adapter Checkpoint Saving and Loading (PEFT format).
"""

import json
import pytest
import torch
from src.training.checkpoint_manager import (
    save_adapter_checkpoint,
    load_adapter_checkpoint,
)


def test_checkpoint_roundtrip_safetensors(tmp_path):
    """Verify saving and loading adapter weights and config with full fidelity."""
    state_dict = {
        "base_model.model.q_proj.lora_A.weight": torch.randn(8, 64),
        "base_model.model.q_proj.lora_B.weight": torch.randn(64, 8),
        "base_model.model.v_proj.lora_A.weight": torch.randn(8, 64),
        "base_model.model.v_proj.lora_B.weight": torch.randn(64, 8),
    }

    lora_cfg = {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM"
    }

    training_meta = {
        "task_name": "gsm8k",
        "epochs": 3,
        "best_eval_loss": 0.42
    }

    save_adapter_checkpoint(
        output_dir=str(tmp_path),
        adapter_state_dict=state_dict,
        lora_config=lora_cfg,
        base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        training_metadata=training_meta,
        training_log={"steps": [1, 2, 3]},
        validation_metrics={"best_eval_loss": 0.42}
    )

    # Check files exist
    assert (tmp_path / "adapter_config.json").exists()
    assert (tmp_path / "adapter_model.safetensors").exists()
    assert (tmp_path / "training_metadata.json").exists()
    assert (tmp_path / "loss_curve.json").exists()

    # Load back
    loaded = load_adapter_checkpoint(str(tmp_path))
    loaded_sd = loaded["state_dict"]
    loaded_cfg = loaded["config"]
    loaded_meta = loaded["metadata"]

    assert loaded_cfg["r"] == 8
    assert loaded_cfg["lora_alpha"] == 16
    assert loaded_cfg["target_modules"] == ["q_proj", "v_proj"]
    assert loaded_meta["task_name"] == "gsm8k"

    for key, tensor in state_dict.items():
        assert key in loaded_sd
        assert torch.allclose(tensor, loaded_sd[key], atol=1e-6)


def test_missing_checkpoint_raises(tmp_path):
    """Verify loading from non-existent directory raises FileNotFoundError."""
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        load_adapter_checkpoint(str(non_existent))
