"""
Unit tests for LoRA Adapter Training, StandaloneLoRALinear, and Label Masking.
"""

import pytest
import torch
import torch.nn as nn
from src.training.lora_trainer import (
    StandaloneLoRALinear,
    inject_lora_into_model,
    extract_lora_state_dict,
    SimpleCausalDataset,
    collate_causal_lm,
    LoRATrainer,
)
from src.training.training_config import LoRAConfig, TrainingHyperparameters
from src.data_preprocessing import preprocess_dataset
from src.model_loader import MockTokenizer, MockQwenModel


def test_causal_lm_label_masking():
    """Verify that prompt tokens are masked with -100 and only target tokens have valid labels."""
    tokenizer = MockTokenizer()
    dataset = [
        {"input_text": "Solve 2 + 2", "label": "4"}
    ]
    preprocessed = preprocess_dataset(
        dataset,
        task_name="gsm8k",
        tokenizer=tokenizer,
        max_seq_length=128,
        is_training=True
    )

    labels = preprocessed["labels"][0]
    input_ids = preprocessed["input_ids"][0]

    assert len(labels) == len(input_ids)
    # The prompt portion must be masked with -100
    assert -100 in labels
    assert labels[0] == -100
    # The final target token(s) must have valid non-negative token IDs
    assert any(l != -100 for l in labels)


def test_standalone_lora_linear():
    """Verify StandaloneLoRALinear scaling, parameter freeze, and forward pass."""
    in_features, out_features = 64, 128
    base_linear = nn.Linear(in_features, out_features, bias=False)
    # Save original base weight
    w0 = base_linear.weight.clone().detach()

    lora_linear = StandaloneLoRALinear(
        base_layer=base_linear,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0
    )

    # Base weight must be frozen
    assert not lora_linear.base_layer.weight.requires_grad
    # LoRA matrices must be trainable
    assert lora_linear.lora_A.weight.requires_grad
    assert lora_linear.lora_B.weight.requires_grad

    # LoRA B is initialized to zeros -> initial output must equal base_linear output exactly
    x = torch.randn(2, in_features)
    out_lora = lora_linear(x)
    out_base = torch.matmul(x, w0.t())

    assert torch.allclose(out_lora, out_base, atol=1e-6)

    # Now modify lora_B and verify scaling alpha / r = 16 / 8 = 2.0
    lora_linear.lora_B.weight.data.fill_(1.0)
    lora_linear.lora_A.weight.data.fill_(0.5)

    out_modified = lora_linear(x)
    # Manual delta: (alpha / r) * B(A(x))
    # A(x) = x @ A.T, B(A(x)) = (x @ A.T) @ B.T
    scaling = 16.0 / 8.0
    ax = x @ lora_linear.lora_A.weight.t()
    bax = ax @ lora_linear.lora_B.weight.t()
    expected = out_base + scaling * bax

    assert torch.allclose(out_modified, expected, atol=1e-5)


def test_gradient_updates_only_lora_parameters():
    """Verify that backpropagation updates LoRA weights while leaving base model untouched."""
    model = MockQwenModel()
    lora_cfg = LoRAConfig(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])

    model, lora_modules = inject_lora_into_model(model, lora_cfg)

    # Check that base parameters require_grad is False
    for name, p in model.named_parameters():
        if "lora_" in name:
            assert p.requires_grad, f"LoRA param {name} should be trainable"
        else:
            assert not p.requires_grad, f"Base param {name} should be frozen"

    # Forward pass and backprop
    dummy_input = torch.randint(0, 1000, (2, 16))
    dummy_labels = dummy_input.clone()
    outputs = model(input_ids=dummy_input, labels=dummy_labels)
    loss = outputs.loss
    loss.backward()

    # LoRA parameters must have gradients
    for name, module in lora_modules.items():
        assert module.lora_A.weight.grad is not None
        assert module.lora_B.weight.grad is not None
        assert torch.isfinite(module.lora_A.weight.grad).all()
        assert torch.isfinite(module.lora_B.weight.grad).all()


def test_smoke_lora_trainer_execution(tmp_path):
    """Verify end-to-end execution of LoRATrainer on mock data for 1 epoch."""
    model = MockQwenModel()
    tokenizer = MockTokenizer()

    train_data = [
        {"input_text": f"Question {i}", "label": f"Answer {i}"} for i in range(8)
    ]
    val_data = [
        {"input_text": f"Val Question {i}", "label": f"Val Answer {i}"} for i in range(4)
    ]

    training_cfg = TrainingHyperparameters(
        learning_rate=1e-3,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        seed=42
    )
    lora_cfg = LoRAConfig(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])

    trainer = LoRATrainer(
        model=model,
        tokenizer=tokenizer,
        task_name="test_task",
        train_dataset=train_data,
        val_dataset=val_data,
        training_config=training_cfg,
        lora_config=lora_cfg,
        output_dir=str(tmp_path),
        device="cpu"
    )

    summary = trainer.train()
    assert summary["best_eval_loss"] > 0.0
    assert (tmp_path / "adapter_config.json").exists()
    assert (tmp_path / "adapter_model.safetensors").exists()
    assert (tmp_path / "training_metadata.json").exists()
    assert (tmp_path / "loss_curve.json").exists()
