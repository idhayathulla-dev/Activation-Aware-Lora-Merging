"""
LoRA Trainer Engine: Independent adapter training with validation checkpointing and label masking.

Trains task-specific LoRA adapters on causal language modeling tasks.
Supports both real HuggingFace PEFT models and standalone LoRALinear wrappers for CPU smoke tests.
"""

import copy
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset

from ..data_preprocessing import preprocess_dataset
from .checkpoint_manager import save_adapter_checkpoint
from .training_config import LoRAConfig, TrainingHyperparameters
from .training_metrics import MetricTracker
from .training_utils import SimpleLinearWarmupScheduler, set_seed


class SimpleCausalDataset(Dataset):
    """PyTorch Dataset wrapper for preprocessed causal LM examples."""
    def __init__(self, data_dict: Dict[str, Any]):
        self.input_ids = data_dict["input_ids"]
        self.attention_mask = data_dict["attention_mask"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }


def collate_causal_lm(batch: List[Dict[str, Any]], pad_token_id: int = 151643) -> Dict[str, torch.Tensor]:
    """Dynamically pads input_ids, attention_mask, and labels to maximum batch length."""
    max_len = max(len(item["input_ids"]) for item in batch)
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for item in batch:
        cur_len = len(item["input_ids"])
        diff = max_len - cur_len
        padded_input_ids.append(item["input_ids"] + [pad_token_id] * diff)
        padded_attention_mask.append(item["attention_mask"] + [0] * diff)
        padded_labels.append(item["labels"] + [-100] * diff)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long)
    }


class StandaloneLoRALinear(torch.nn.Module):
    """
    Standard LoRA linear layer wrapper implementing:
        y = W_0 x + (alpha / r) * B(A(dropout(x)))
    Matches official PEFT state dict naming conventions and initialization.
    """
    def __init__(
        self,
        base_layer: torch.nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = float(lora_alpha) / float(r)

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.lora_dropout = torch.nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else torch.nn.Identity()
        self.lora_A = torch.nn.Linear(in_features, r, bias=False)
        self.lora_B = torch.nn.Linear(r, out_features, bias=False)

        # Standard PEFT initialization: Kaiming uniform for A, zeros for B
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out


def inject_lora_into_model(
    model: torch.nn.Module,
    lora_config: LoRAConfig
) -> Tuple[torch.nn.Module, Dict[str, StandaloneLoRALinear]]:
    """
    Injects LoRA adapters into target modules (e.g. q_proj, v_proj).
    Uses PEFT if installed and model is a Hugging Face PreTrainedModel,
    otherwise uses StandaloneLoRALinear.
    """
    # Freeze all base parameters
    for p in model.parameters():
        p.requires_grad = False

    lora_modules = {}

    # Check if PEFT is available and model is a supported transformers model
    try:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
        if hasattr(model, "config") and type(model).__name__ != "MockQwenModel":
            peft_cfg = PeftLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config.rank,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
                target_modules=lora_config.target_modules,
                bias=lora_config.bias
            )
            peft_model = get_peft_model(model, peft_cfg)
            return peft_model, {}
    except ImportError:
        pass

    # Standalone injection for custom/mock models or offline CPU execution
    for name, module in list(model.named_modules()):
        for target in lora_config.target_modules:
            if target in name and isinstance(module, torch.nn.Linear) and not isinstance(module, StandaloneLoRALinear):
                # Replace linear with StandaloneLoRALinear
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                child_name = name.rsplit(".", 1)[1] if "." in name else name
                parent = model.get_submodule(parent_name) if parent_name else model

                wrapped = StandaloneLoRALinear(
                    base_layer=module,
                    r=lora_config.rank,
                    lora_alpha=lora_config.alpha,
                    lora_dropout=lora_config.dropout
                )
                setattr(parent, child_name, wrapped)
                lora_modules[name] = wrapped

    return model, lora_modules


def extract_lora_state_dict(
    model: torch.nn.Module,
    lora_modules: Dict[str, StandaloneLoRALinear]
) -> Dict[str, torch.Tensor]:
    """Extracts only trainable LoRA parameters matching PEFT naming conventions."""
    # If official PEFT model
    if hasattr(model, "peft_config"):
        try:
            from peft import get_peft_model_state_dict
            return get_peft_model_state_dict(model)
        except ImportError:
            pass

    # From StandaloneLoRALinear
    state_dict = {}
    for name, module in lora_modules.items():
        prefix = f"base_model.model.{name}"
        state_dict[f"{prefix}.lora_A.weight"] = module.lora_A.weight.data.clone().cpu()
        state_dict[f"{prefix}.lora_B.weight"] = module.lora_B.weight.data.clone().cpu()

    if not state_dict:
        # Fallback: any parameter requiring grad
        for n, p in model.named_parameters():
            if p.requires_grad:
                state_dict[n] = p.data.clone().cpu()

    return state_dict


class LoRATrainer:
    """Trainer engine for training task-specific LoRA adapters."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        task_name: str,
        train_dataset: List[Dict[str, Any]],
        val_dataset: List[Dict[str, Any]],
        training_config: TrainingHyperparameters,
        lora_config: LoRAConfig,
        output_dir: str,
        device: Optional[str] = None
    ):
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.training_config = training_config
        self.lora_config = lora_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        set_seed(self.training_config.seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Inject LoRA adapters
        self.model, self.lora_modules = inject_lora_into_model(model, self.lora_config)
        self.model.to(self.device)

        # Preprocess train and validation sets
        pad_id = getattr(self.tokenizer, "pad_token_id", 151643)
        train_preprocessed = preprocess_dataset(
            train_dataset,
            task_name=task_name,
            tokenizer=tokenizer,
            max_seq_length=training_config.max_seq_length,
            is_training=True
        )
        val_preprocessed = preprocess_dataset(
            val_dataset,
            task_name=task_name,
            tokenizer=tokenizer,
            max_seq_length=training_config.max_seq_length,
            is_training=True
        )

        train_ds = SimpleCausalDataset(train_preprocessed)
        val_ds = SimpleCausalDataset(val_preprocessed)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=training_config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_causal_lm(b, pad_token_id=pad_id)
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=training_config.per_device_train_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_causal_lm(b, pad_token_id=pad_id)
        )

        # Optimizer: only trainable LoRA parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable LoRA parameters found in model!")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )

        total_steps = len(self.train_loader) * training_config.num_train_epochs
        warmup_steps = int(total_steps * training_config.warmup_ratio)
        self.scheduler = SimpleLinearWarmupScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )

        self.metrics = MetricTracker()

    def evaluate(self) -> float:
        """Evaluates model loss on the validation dataset."""
        self.model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                if loss is not None and torch.isfinite(loss):
                    total_loss += loss.item()
                    batches += 1

        val_loss = total_loss / max(1, batches)
        return val_loss

    def train(self) -> Dict[str, Any]:
        """Executes full training loop across epochs, selecting best model by validation loss."""
        print(f"\n[LoRATrainer] Starting training for task '{self.task_name}' on {self.device}...")
        print(f"  Train examples    : {len(self.train_loader.dataset)}")
        print(f"  Val examples      : {len(self.val_loader.dataset)}")
        print(f"  Epochs            : {self.training_config.num_train_epochs}")
        print(f"  Batch size        : {self.training_config.per_device_train_batch_size}")
        print(f"  Grad Accumulation : {self.training_config.gradient_accumulation_steps}")

        best_val_loss = float("inf")
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        global_step = 0
        start_time = time.time()

        for epoch in range(1, self.training_config.num_train_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            step_in_epoch = 0

            for batch_idx, batch in enumerate(self.train_loader):
                step_start = time.time()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                # Normalize loss for gradient accumulation
                loss = loss / self.training_config.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.training_config.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.training_config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    current_loss = loss.item() * self.training_config.gradient_accumulation_steps
                    epoch_loss += current_loss
                    step_in_epoch += 1

                    step_dur = time.time() - step_start
                    if global_step % self.training_config.logging_steps == 0:
                        cur_lr = self.scheduler.get_last_lr()[0]
                        self.metrics.log_step(
                            step=global_step,
                            epoch=epoch + (batch_idx / len(self.train_loader)),
                            train_loss=current_loss,
                            lr=cur_lr,
                            step_duration=step_dur
                        )

            avg_train_loss = epoch_loss / max(1, step_in_epoch)
            val_loss = self.evaluate()
            is_best = val_loss < best_val_loss

            if is_best:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(extract_lora_state_dict(self.model, self.lora_modules))

            epoch_metric = self.metrics.log_epoch(
                epoch=epoch,
                train_loss=avg_train_loss,
                eval_loss=val_loss,
                best_eval_loss=best_val_loss,
                is_best=is_best
            )

            print(
                f"  Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f} {'(*)' if is_best else ''}"
            )

        total_duration = round(time.time() - start_time, 2)
        print(f"[LoRATrainer] Training complete in {total_duration}s. Best Val Loss: {best_val_loss:.4f}")

        # If no improvement (e.g. 1 epoch), extract current state dict
        if best_state_dict is None:
            best_state_dict = extract_lora_state_dict(self.model, self.lora_modules)

        # Save checkpoint artifacts
        training_meta = {
            "task_name": self.task_name,
            "training_duration_sec": total_duration,
            "total_steps": global_step,
            "best_eval_loss": best_val_loss,
            "final_train_loss": avg_train_loss,
            "seed": self.training_config.seed,
            "device": str(self.device),
        }

        save_adapter_checkpoint(
            output_dir=str(self.output_dir),
            adapter_state_dict=best_state_dict,
            lora_config={
                "rank": self.lora_config.rank,
                "alpha": self.lora_config.alpha,
                "dropout": self.lora_config.dropout,
                "target_modules": self.lora_config.target_modules,
                "bias": self.lora_config.bias,
                "task_type": self.lora_config.task_type
            },
            base_model_name=getattr(self.model, "name_or_path", "Qwen/Qwen2.5-0.5B-Instruct"),
            training_metadata=training_meta,
            training_log=self.metrics.to_dict(),
            validation_metrics={"best_eval_loss": best_val_loss, "epoch_history": self.metrics.epoch_history}
        )

        return {
            "best_eval_loss": best_val_loss,
            "total_duration_sec": total_duration,
            "output_dir": str(self.output_dir),
            "state_dict_keys": list(best_state_dict.keys())
        }
