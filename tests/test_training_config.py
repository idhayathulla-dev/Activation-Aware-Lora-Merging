"""
Unit tests for Training and LoRA Configuration Parity and YAML serialization.
"""

from pathlib import Path
from src.training.training_config import (
    LoRAConfig,
    TrainingHyperparameters,
    ExperimentConfig,
    load_training_config,
    load_experiment_config,
)


def test_load_training_config():
    """Verify loading configs/training.yaml loads proper parameters."""
    cfg_path = "configs/training.yaml"
    training_cfg, lora_cfg = load_training_config(cfg_path)

    # LoRA config checks
    assert lora_cfg.rank == 8
    assert lora_cfg.alpha == 16
    assert lora_cfg.dropout == 0.05
    assert "q_proj" in lora_cfg.target_modules
    assert "v_proj" in lora_cfg.target_modules
    assert lora_cfg.bias == "none"
    assert lora_cfg.task_type == "CAUSAL_LM"

    # Training hyperparameters checks
    assert training_cfg.learning_rate == 2e-4
    assert training_cfg.num_train_epochs == 3
    assert training_cfg.per_device_train_batch_size == 4
    assert training_cfg.gradient_accumulation_steps == 4
    assert training_cfg.seed == 42


def test_load_experiment_config():
    """Verify loading configs/experiment.yaml loads proper tasks and splits."""
    exp_path = "configs/experiment.yaml"
    exp_cfg = load_experiment_config(exp_path)

    assert exp_cfg.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert "boolq" in exp_cfg.tasks
    assert "gsm8k" in exp_cfg.tasks
    assert "arc_easy" in exp_cfg.tasks

    for t_name, t_info in exp_cfg.tasks.items():
        splits = t_info["splits"]
        assert splits["calibration"] == 128
        assert splits["evaluation"] == "official_evaluation"
        assert splits["train"] == "remaining_train"
        if t_name == "arc_easy":
            assert splits["validation"] == "official_validation"
        else:
            assert splits["validation"] == 100


def test_lora_config_dataclass_methods():
    """Verify LoRAConfig dictionary conversion."""
    cfg = LoRAConfig(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])
    d = cfg.to_dict()
    assert d["rank"] == 8
    assert d["alpha"] == 16
    assert d["target_modules"] == ["q_proj", "v_proj"]
