"""
Training Utilities: Random seeding, hardware detection, and learning rate scheduling.
"""

import math
import os
import random
from typing import Dict, Any, Optional
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def detect_hardware() -> Dict[str, Any]:
    """Detects available hardware and memory diagnostics."""
    cuda_avail = torch.cuda.is_available()
    hw_info = {
        "device": "cuda" if cuda_avail else "cpu",
        "cuda_available": cuda_avail,
        "cuda_version": torch.version.cuda if cuda_avail else None,
        "device_count": torch.cuda.device_count() if cuda_avail else 0,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_avail else "None (CPU only)",
        "bf16_supported": torch.cuda.is_bf16_supported() if cuda_avail else False,
    }

    if cuda_avail:
        try:
            prop = torch.cuda.get_device_properties(0)
            hw_info["gpu_total_vram_gb"] = round(prop.total_memory / (1024 ** 3), 2)
            hw_info["gpu_free_vram_gb"] = round((prop.total_memory - torch.cuda.memory_allocated()) / (1024 ** 3), 2)
        except Exception:
            hw_info["gpu_total_vram_gb"] = 0.0
            hw_info["gpu_free_vram_gb"] = 0.0
    else:
        hw_info["gpu_total_vram_gb"] = 0.0
        hw_info["gpu_free_vram_gb"] = 0.0

    try:
        import psutil
        vm = psutil.virtual_memory()
        hw_info["ram_total_gb"] = round(vm.total / (1024 ** 3), 2)
        hw_info["ram_available_gb"] = round(vm.available / (1024 ** 3), 2)
    except ImportError:
        hw_info["ram_total_gb"] = 0.0
        hw_info["ram_available_gb"] = 0.0

    hw_info["device_type"] = hw_info["device"]
    hw_info["device_name"] = hw_info["gpu_name"]
    hw_info["total_memory_gb"] = hw_info["ram_total_gb"]
    hw_info["free_memory_gb"] = hw_info["ram_available_gb"]

    return hw_info


class SimpleLinearWarmupScheduler:
    """Standard linear warmup and linear decay learning rate scheduler."""
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps, total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            progress = float(self.current_step) / float(self.warmup_steps)
            factor = progress
        else:
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            factor = max(self.min_lr_ratio, 1.0 - progress)

        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]
