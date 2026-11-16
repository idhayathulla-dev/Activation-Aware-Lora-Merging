"""
Task Arithmetic Merging Operator (Ilharco et al., ICLR 2023).

Performs linear combination of task vectors or aligned coordinate representations.
"""

from typing import List, Optional
import torch


def task_arithmetic_merge(
    tensors: List[torch.Tensor],
    weights: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Computes a weighted linear combination of input tensors:
        T_merged = sum_{k=1}^K w_k * T_k

    Args:
        tensors: List of K tensors, all sharing identical shapes and dtypes.
        weights: Optional list of K scalar weights. If None, defaults to uniform 1 / K.

    Returns:
        Merged tensor of the same shape and dtype.
    """
    if not tensors:
        raise ValueError("Cannot merge an empty list of tensors.")
    
    K = len(tensors)
    ref_shape = tensors[0].shape
    ref_dtype = tensors[0].dtype
    ref_device = tensors[0].device
    
    for i, t in enumerate(tensors):
        if t.shape != ref_shape:
            raise ValueError(f"Tensor {i} shape {t.shape} does not match reference shape {ref_shape}.")
        if not torch.all(torch.isfinite(t)):
            raise ValueError(f"Tensor {i} contains NaN or infinite values.")
            
    if weights is None:
        w_list = [1.0 / float(K)] * K
    else:
        if len(weights) != K:
            raise ValueError(f"Expected {K} weights, got {len(weights)}.")
        w_list = [float(w) for w in weights]
        
    merged = torch.zeros(ref_shape, dtype=ref_dtype, device=ref_device)
    for w, t in zip(w_list, tensors):
        merged = merged + (w * t)
        
    return merged
