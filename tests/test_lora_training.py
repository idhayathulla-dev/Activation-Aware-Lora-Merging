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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_standalone_lora_mixed_precision_forward_and_backward(dtype):
    """
    Regression test for RuntimeError: expected mat1 and mat2 to have the same dtype.
    Verifies that StandaloneLoRALinear properly handles inputs in FP32, FP16, and BF16
    when LoRA weights are FP32, producing valid output matching input dtype and finite gradients.
    """
    in_features, out_features = 64, 128
    base_linear = nn.Linear(in_features, out_features, bias=False).to(dtype)
    base_w0 = base_linear.weight.clone().detach()

    lora_linear = StandaloneLoRALinear(
        base_layer=base_linear,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0
    )

    # Base layer must be frozen
    assert not lora_linear.base_layer.weight.requires_grad
    # LoRA weights remain in float32 for optimizer stability
    assert lora_linear.lora_A.weight.dtype == torch.float32
    assert lora_linear.lora_B.weight.dtype == torch.float32

    # Initialize LoRA weights with known non-zero values to test delta
    lora_linear.lora_A.weight.data.fill_(0.25)
    lora_linear.lora_B.weight.data.fill_(0.5)

    x = torch.randn(2, in_features, dtype=dtype, requires_grad=True)
    out = lora_linear(x)

    # Output dtype must strictly match base layer dtype
    assert out.dtype == dtype, f"Expected output dtype {dtype}, got {out.dtype}"

    # Check scaling and mathematical correctness
    scaling = 16.0 / 8.0
    expected_base = torch.matmul(x, base_w0.t())
    lora_a_out = torch.matmul(x.to(torch.float32), lora_linear.lora_A.weight.t())
    lora_b_out = torch.matmul(lora_a_out, lora_linear.lora_B.weight.t()) * scaling
    expected_out = expected_base + lora_b_out.to(dtype)

    assert torch.allclose(out, expected_out, atol=1e-3 if dtype == torch.bfloat16 else 1e-4)

    # Verify backpropagation and gradient flow
    loss = out.float().sum()
    loss.backward()

    assert lora_linear.lora_A.weight.grad is not None
    assert lora_linear.lora_B.weight.grad is not None
    assert torch.isfinite(lora_linear.lora_A.weight.grad).all()
    assert torch.isfinite(lora_linear.lora_B.weight.grad).all()
    assert lora_linear.lora_A.weight.grad.dtype == torch.float32
    assert lora_linear.lora_B.weight.grad.dtype == torch.float32
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_resolve_torch_dtype_hardware_fallback(monkeypatch):
    """Verify that resolve_torch_dtype falls back from BF16 to FP16 on GPUs lacking BF16 support (e.g. Tesla T4)."""
    from src.model_loader import resolve_torch_dtype

    # CPU mode: bfloat16 -> float32
    assert resolve_torch_dtype("bfloat16", device="cpu") == torch.float32

    # CUDA available but bf16 NOT supported (Tesla T4 scenario)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_torch_dtype("bfloat16", device="cuda") == torch.float16

    # CUDA available and bf16 supported (A100/H100 scenario)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_torch_dtype("bfloat16", device="cuda") == torch.bfloat16
