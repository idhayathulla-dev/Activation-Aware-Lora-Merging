"""
Training Metrics Tracker and Statistical Logging.
"""

import math
import time
from typing import Any, Dict, List, Optional


def calculate_perplexity(loss: float) -> float:
    """Computes perplexity safely, clamping to prevent overflow."""
    try:
        clamped_loss = min(max(loss, 0.0), 50.0)
        return math.exp(clamped_loss)
    except OverflowError:
        return float("inf")


class MetricTracker:
    """Tracks and formats step-by-step and epoch-level training metrics."""

    def __init__(self):
        self.step_history: List[Dict[str, Any]] = []
        self.epoch_history: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def log_step(
        self,
        step: int,
        epoch: float,
        train_loss: float,
        lr: float,
        step_duration: float
    ) -> Dict[str, Any]:
        entry = {
            "step": step,
            "epoch": round(epoch, 3),
            "train_loss": round(train_loss, 5),
            "train_perplexity": round(calculate_perplexity(train_loss), 3),
            "learning_rate": lr,
            "step_duration_sec": round(step_duration, 4),
            "elapsed_sec": round(time.time() - self.start_time, 2)
        }
        self.step_history.append(entry)
        return entry

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        eval_loss: float,
        best_eval_loss: float,
        is_best: bool
    ) -> Dict[str, Any]:
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_perplexity": round(calculate_perplexity(train_loss), 3),
            "eval_loss": round(eval_loss, 5),
            "eval_perplexity": round(calculate_perplexity(eval_loss), 3),
            "best_eval_loss": round(best_eval_loss, 5),
            "is_best": is_best,
            "elapsed_sec": round(time.time() - self.start_time, 2)
        }
        self.epoch_history.append(entry)
        return entry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_elapsed_sec": round(time.time() - self.start_time, 2),
            "total_steps": len(self.step_history),
            "total_epochs": len(self.epoch_history),
            "step_history": self.step_history,
            "epoch_history": self.epoch_history
        }
