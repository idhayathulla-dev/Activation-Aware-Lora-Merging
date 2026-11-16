"""
Training and Experiment Configuration Data Structures and Loaders.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "bias": self.bias,
            "task_type": self.task_type
        }


@dataclass
class TrainingHyperparameters:
    seed: int = 42
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    max_grad_norm: float = 1.0
    max_seq_length: int = 512
    fp16: bool = False
    bf16: bool = True
    torch_dtype: str = "bfloat16"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_ratio": self.warmup_ratio,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "evaluation_strategy": self.evaluation_strategy,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "max_grad_norm": self.max_grad_norm,
            "max_seq_length": self.max_seq_length,
            "fp16": self.fp16,
            "bf16": self.bf16,
        }


@dataclass
class ExperimentConfig:
    name: str = "activation_aware_lora_merging"
    mode: str = "final"
    allow_synthetic: bool = False
    require_real_huggingface_data: bool = True
    seed: int = 42
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = "main"
    torch_dtype: str = "bfloat16"
    tasks: Dict[str, Any] = field(default_factory=dict)
    artifacts_dir: str = "artifacts"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        return load_experiment_config(path)


def load_training_config(
    config_path: str = "configs/training.yaml"
) -> Tuple[TrainingHyperparameters, LoRAConfig]:
    """Loads training hyperparameters and LoRA configuration from YAML file."""
    p = Path(config_path)
    if not p.exists():
        alt_p = Path("activation-aware-lora-merging") / config_path
        if alt_p.exists():
            p = alt_p
        else:
            raise FileNotFoundError(f"Training config not found at: {config_path}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    lora_data = data.get("lora", {})
    lora_cfg = LoRAConfig(
        rank=lora_data.get("rank", lora_data.get("r", 8)),
        alpha=lora_data.get("alpha", lora_data.get("lora_alpha", 16)),
        dropout=lora_data.get("dropout", lora_data.get("lora_dropout", 0.05)),
        target_modules=lora_data.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_data.get("bias", "none"),
        task_type=lora_data.get("task_type", "CAUSAL_LM")
    )

    tr_data = data.get("training", {})
    tr_cfg = TrainingHyperparameters(
        seed=tr_data.get("seed", 42),
        learning_rate=float(tr_data.get("learning_rate", 2e-4)),
        weight_decay=float(tr_data.get("weight_decay", 0.01)),
        num_train_epochs=int(tr_data.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(tr_data.get("per_device_train_batch_size", 4)),
        gradient_accumulation_steps=int(tr_data.get("gradient_accumulation_steps", 4)),
        warmup_ratio=float(tr_data.get("warmup_ratio", 0.05)),
        logging_steps=int(tr_data.get("logging_steps", 10)),
        save_strategy=tr_data.get("save_strategy", "epoch"),
        evaluation_strategy=tr_data.get("evaluation_strategy", "epoch"),
        load_best_model_at_end=bool(tr_data.get("load_best_model_at_end", True)),
        metric_for_best_model=tr_data.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(tr_data.get("greater_is_better", False)),
        max_grad_norm=float(tr_data.get("max_grad_norm", 1.0)),
        max_seq_length=int(tr_data.get("max_seq_length", 512)),
        fp16=bool(tr_data.get("fp16", False)),
        bf16=bool(tr_data.get("bf16", True)),
        torch_dtype="bfloat16" if tr_data.get("bf16", True) else ("float16" if tr_data.get("fp16", False) else "float32")
    )

    return tr_cfg, lora_cfg


def load_experiment_config(
    config_path: str = "configs/experiment.yaml"
) -> ExperimentConfig:
    """Loads experimental protocol configuration from YAML file."""
    p = Path(config_path)
    if not p.exists():
        alt_p = Path("activation-aware-lora-merging") / config_path
        if alt_p.exists():
            p = alt_p
        else:
            raise FileNotFoundError(f"Experiment config not found at: {config_path}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    exp_info = data.get("experiment", {})
    bm_info = data.get("base_model", {})
    if isinstance(bm_info, str):
        base_model_name = bm_info
        revision = "main"
        torch_dtype = "bfloat16"
    else:
        base_model_name = bm_info.get("name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
        revision = bm_info.get("revision", "main")
        torch_dtype = bm_info.get("torch_dtype", "bfloat16")

    return ExperimentConfig(
        name=exp_info.get("name", "activation_aware_lora_merging"),
        mode=exp_info.get("mode", "final"),
        allow_synthetic=bool(exp_info.get("allow_synthetic", False)),
        require_real_huggingface_data=bool(exp_info.get("require_real_huggingface_data", True)),
        seed=int(exp_info.get("seed", 42)),
        base_model=base_model_name,
        revision=revision,
        torch_dtype=torch_dtype,
        tasks=data.get("tasks", {}),
        artifacts_dir=data.get("artifacts_dir", "artifacts")
    )
