"""
Original KnOTS (Knowledge Orientation Through SVD) Model Merging (Stoica et al., ICLR 2025).

Implements the official KnOTS algorithm:
1. Horizontal concatenation of task updates.
2. Joint SVD to find shared left basis U and right singular vectors V.
3. Task-specific partitioning of right singular vectors into aligned representations.
4. Model merging in aligned coordinate space (Task Arithmetic or TIES).
5. Exact rank-R LoRA parameterization with strict PEFT scaling preservation.
"""

from typing import Dict, List, Optional, Tuple
import torch

from .task_arithmetic import task_arithmetic_merge
from .ties import ties_merge


def knots_align(
    task_updates: List[torch.Tensor],
    rank_budget: int
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Computes joint SVD alignment across task updates.

    Args:
        task_updates: List of K tensors, each of shape (d_out, d_in).
        rank_budget: Target rank R <= min(d_out, K * d_in).

    Returns:
        Tuple of:
            - U_R: Shared left singular basis of shape (d_out, R) in float64.
            - S_R: Singular values of shape (R,) in float64.
            - aligned_task_reps: List of K aligned tensors Z_k = diag(S_R) @ V_{R, k}^T of shape (R, d_in) in float64.
    """
    if not task_updates:
        raise ValueError("task_updates list cannot be empty.")
        
    K = len(task_updates)
    d_out, d_in = task_updates[0].shape
    ref_dtype = task_updates[0].dtype
    ref_device = task_updates[0].device
    
    for i, t in enumerate(task_updates):
        if t.shape != (d_out, d_in):
            raise ValueError(f"Task {i} shape {t.shape} does not match expected ({d_out}, {d_in}).")
            
    # Concatenate horizontally: shape (d_out, K * d_in)
    W_cat = torch.cat([t.to(torch.float64) for t in task_updates], dim=1)
    
    # Compute full/reduced SVD in float64
    U, S, Vh = torch.linalg.svd(W_cat, full_matrices=False)
    
    R = min(rank_budget, len(S), d_out)
    U_R = U[:, :R]           # (d_out, R)
    S_R = S[:R]              # (R,)
    Vh_R = Vh[:R, :]         # (R, K * d_in)
    
    # Partition Vh_R into K task blocks of shape (R, d_in)
    aligned_reps = []
    diag_S = torch.diag(S_R) # (R, R)
    for k in range(K):
        Vh_k = Vh_R[:, k * d_in : (k + 1) * d_in] # (R, d_in)
        Z_k = diag_S @ Vh_k                       # (R, d_in)
        aligned_reps.append(Z_k)
        
    return U_R, S_R, aligned_reps


def knots_merge(
    task_updates: List[torch.Tensor],
    rank_budget: int,
    merge_operator: str = "task_arithmetic",
    merge_kwargs: Optional[Dict] = None,
    lora_scaling: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Executes full KnOTS merging and reconstructs both the dense update and rank-R LoRA factors.

    The reconstructed parameters satisfy:
        DeltaW_dense = lora_scaling * (lora_B @ lora_A)

    Args:
        task_updates: List of K tensors of shape (d_out, d_in).
        rank_budget: Target rank R.
        merge_operator: "task_arithmetic" or "ties".
        merge_kwargs: Optional dict of arguments passed to the merge operator.
        lora_scaling: Effective PEFT scaling factor (lora_alpha / r).

    Returns:
        Tuple of:
            - dense_update: Merged update tensor of shape (d_out, d_in).
            - lora_B: LoRA B matrix of shape (d_out, R).
            - lora_A: LoRA A matrix of shape (R, d_in).
    """
    if lora_scaling <= 0.0:
        raise ValueError(f"lora_scaling must be positive, got {lora_scaling}")
        
    merge_kwargs = merge_kwargs or {}
    ref_dtype = task_updates[0].dtype
    ref_device = task_updates[0].device
    
    # 1. Align task updates via joint SVD
    U_R, S_R, aligned_reps = knots_align(task_updates, rank_budget=rank_budget)
    
    # 2. Merge in aligned coordinate space
    if merge_operator == "task_arithmetic":
        Z_merged = task_arithmetic_merge(aligned_reps, **merge_kwargs)
    elif merge_operator == "ties":
        Z_merged = ties_merge(aligned_reps, **merge_kwargs)
    else:
        raise ValueError(f"Unsupported merge_operator: {merge_operator}")
        
    # 3. Dense reconstruction in float64
    DeltaW_dense = U_R @ Z_merged
    
    # 4. LoRA factor parameterization:
    # DeltaW = lora_scaling * (lora_B @ lora_A)
    # With lora_B = U_R and lora_A = (1 / lora_scaling) * Z_merged
    lora_B = U_R.to(dtype=ref_dtype, device=ref_device)
    lora_A = (Z_merged / lora_scaling).to(dtype=ref_dtype, device=ref_device)
    dense_update = DeltaW_dense.to(dtype=ref_dtype, device=ref_device)
    
    return dense_update, lora_B, lora_A
