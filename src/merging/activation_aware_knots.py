"""
Activation-Distribution-Aware KnOTS Model Merging (Algorithm B).

Transforms task-specific LoRA updates into activation-weighted coordinates via regularized
covariance square roots prior to joint SVD alignment and merging, then reconstructs exact
rank-R LoRA adapters in the original parameter space.
"""

from typing import Dict, List, Optional, Tuple
import torch

from ..matrix_utils import compute_matrix_square_roots
from .task_arithmetic import task_arithmetic_merge
from .ties import ties_merge


def activation_aware_knots_merge(
    task_updates: List[torch.Tensor],
    covariance: torch.Tensor,
    rank_budget: int,
    merge_operator: str = "task_arithmetic",
    merge_kwargs: Optional[Dict] = None,
    alpha_reg: float = 1e-3,
    lora_scaling: float = 1.0,
    precomputed_square_roots: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Executes Activation-Aware KnOTS multi-adapter merging (Algorithm B).

    The reconstructed parameters satisfy:
        DeltaW_dense = lora_scaling * (lora_B @ lora_A)

    Args:
        task_updates: List of K tensors of shape (d_out, d_in).
        covariance: Empirical activation second-moment matrix of shape (d_in, d_in).
        rank_budget: Target rank budget R <= min(d_out, K * d_in).
        merge_operator: "task_arithmetic" or "ties".
        merge_kwargs: Optional dict of arguments passed to the merge operator.
        alpha_reg: Tikhonov regularization factor for covariance.
        lora_scaling: Effective PEFT scaling factor (lora_alpha / r).
        precomputed_square_roots: Optional tuple (C_sqrt, C_inv_sqrt) if already cached.

    Returns:
        Tuple of:
            - dense_update: Reconstructed merged update of shape (d_out, d_in).
            - lora_B: Merged LoRA B matrix of shape (d_out, R).
            - lora_A: Merged LoRA A matrix of shape (R, d_in).
            - diagnostics: Dictionary of numerical conditioning and transformation metrics.
    """
    if lora_scaling <= 0.0:
        raise ValueError(f"lora_scaling must be positive, got {lora_scaling}")
    if not task_updates:
        raise ValueError("task_updates list cannot be empty.")
        
    K = len(task_updates)
    d_out, d_in = task_updates[0].shape
    ref_dtype = task_updates[0].dtype
    ref_device = task_updates[0].device
    merge_kwargs = merge_kwargs or {}
    
    # 1. Obtain C_lambda^{1/2} and C_lambda^{-1/2} in float64
    if precomputed_square_roots is not None:
        C_sqrt, C_inv_sqrt = precomputed_square_roots
        diagnostics = {"cached_square_roots": 1.0}
    else:
        C_sqrt, C_inv_sqrt, diagnostics = compute_matrix_square_roots(
            covariance, alpha_reg=alpha_reg
        )
        
    C_sqrt_64 = C_sqrt.to(dtype=torch.float64, device=ref_device)
    C_inv_sqrt_64 = C_inv_sqrt.to(dtype=torch.float64, device=ref_device)
    
    # 2. Transform each task update into canonical activation space:
    # W_tilde_k = DeltaW_k @ C_lambda^{1/2}
    transformed_updates = []
    for k, dW in enumerate(task_updates):
        if dW.shape != (d_out, d_in):
            raise ValueError(f"Task {k} shape {dW.shape} does not match expected ({d_out}, {d_in}).")
        dW_64 = dW.to(dtype=torch.float64, device=ref_device)
        W_tilde = dW_64 @ C_sqrt_64
        transformed_updates.append(W_tilde)
        
    # 3. Concatenate transformed updates horizontally: (d_out, K * d_in)
    W_cat_tilde = torch.cat(transformed_updates, dim=1)
    
    # 4. Joint SVD decomposition in transformed coordinates
    U, S, Vh = torch.linalg.svd(W_cat_tilde, full_matrices=False)
    
    R = min(rank_budget, len(S), d_out)
    U_R = U[:, :R]           # (d_out, R)
    S_R = S[:R]              # (R,)
    Vh_R = Vh[:R, :]         # (R, K * d_in)
    
    # 5. Partition Vh_R into K task blocks of shape (R, d_in)
    diag_S = torch.diag(S_R) # (R, R)
    aligned_reps_tilde = []
    for k in range(K):
        Vh_k = Vh_R[:, k * d_in : (k + 1) * d_in] # (R, d_in)
        Z_k_tilde = diag_S @ Vh_k                 # (R, d_in)
        aligned_reps_tilde.append(Z_k_tilde)
        
    # 6. Model merging in aligned transformed coordinate space
    if merge_operator == "task_arithmetic":
        Z_merged_tilde = task_arithmetic_merge(aligned_reps_tilde, **merge_kwargs)
    elif merge_operator == "ties":
        Z_merged_tilde = ties_merge(aligned_reps_tilde, **merge_kwargs)
    else:
        raise ValueError(f"Unsupported merge_operator: {merge_operator}")
        
    # 7. Dense reconstruction and inverse coordinate transformation:
    # DeltaW_merged = (U_R @ Z_merged_tilde) @ C_lambda^{-1/2}
    #              = U_R @ (Z_merged_tilde @ C_lambda^{-1/2})
    Z_recovered = Z_merged_tilde @ C_inv_sqrt_64  # (R, d_in)
    DeltaW_dense = U_R @ Z_recovered              # (d_out, d_in)
    
    # 8. Factorize into exact rank-R LoRA parameters:
    # DeltaW = lora_scaling * (lora_B @ lora_A)
    # lora_B = U_R
    # lora_A = (1 / lora_scaling) * Z_recovered
    lora_B = U_R.to(dtype=ref_dtype, device=ref_device)
    lora_A = (Z_recovered / lora_scaling).to(dtype=ref_dtype, device=ref_device)
    dense_update = DeltaW_dense.to(dtype=ref_dtype, device=ref_device)
    
    # Record transformation diagnostics
    diagnostics["rank_budget"] = R
    diagnostics["total_singular_values"] = len(S)
    diagnostics["top_singular_value"] = S[0].item()
    diagnostics["trunc_singular_value"] = S_R[-1].item()
    
    return dense_update, lora_B, lora_A, diagnostics
