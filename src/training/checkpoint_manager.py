"""
Checkpoint Manager: Saves adapter checkpoints adhering strictly to PEFT standard structure.

Artifacts saved:
- adapter_config.json
- adapter_model.safetensors
- training_metadata.json
- training_log.json
- validation_metrics.json
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import torch


def save_adapter_checkpoint(
    output_dir: str,
    adapter_state_dict: Dict[str, torch.Tensor],
    lora_config: Dict[str, Any],
    base_model_name: str,
    training_metadata: Dict[str, Any],
    training_log: Dict[str, Any],
    validation_metrics: Dict[str, Any]
) -> Path:
    """
    Saves adapter checkpoint in standard PEFT format alongside complete audit metadata.
    """
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Standard PEFT adapter_config.json
    peft_config = {
        "base_model_name_or_path": base_model_name,
        "peft_type": "LORA",
        "task_type": lora_config.get("task_type", "CAUSAL_LM"),
        "r": lora_config.get("rank", lora_config.get("r", 8)),
        "lora_alpha": lora_config.get("alpha", lora_config.get("lora_alpha", 16)),
        "lora_dropout": lora_config.get("dropout", lora_config.get("lora_dropout", 0.05)),
        "target_modules": lora_config.get("target_modules", ["q_proj", "v_proj"]),
        "bias": lora_config.get("bias", "none"),
        "auto_mapping": None
    }
    with open(save_path / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(peft_config, f, indent=2)

    # 2. Adapter model weights (safetensors preferred, fallback to bin)
    saved_weights = False
    try:
        from safetensors.torch import save_file
        # Ensure contiguous cpu tensors
        clean_state_dict = {
            k: v.contiguous().cpu() for k, v in adapter_state_dict.items()
        }
        save_file(clean_state_dict, str(save_path / "adapter_model.safetensors"))
        saved_weights = True
    except ImportError:
        pass

    if not saved_weights:
        torch.save(adapter_state_dict, str(save_path / "adapter_model.bin"))

    # 3. training_metadata.json
    with open(save_path / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(training_metadata, f, indent=2)

    # 4. training_log.json & loss_curve.json
    with open(save_path / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)
    with open(save_path / "loss_curve.json", "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    # 5. validation_metrics.json
    with open(save_path / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(validation_metrics, f, indent=2)

    return save_path


def load_adapter_checkpoint(checkpoint_dir: str) -> Dict[str, Any]:
    """
    Loads and inspects an adapter checkpoint directory.
    """
    path = Path(checkpoint_dir)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    config_path = path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {checkpoint_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load weights
    safetensors_path = path / "adapter_model.safetensors"
    bin_path = path / "adapter_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path))
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter weight file found in {checkpoint_dir}")

    # Load metadata if exists
    meta_path = path / "training_metadata.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return {
        "config": config,
        "state_dict": state_dict,
        "metadata": meta,
        "directory": str(path)
    }
