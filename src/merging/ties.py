"""
TIES-Merging Operator (Yadav et al., NeurIPS 2023).

Implements Trim, Elect Sign, and Disjoint Merge to resolve interference across tasks.
"""

from typing import List, Optional
import torch


def trim(tensors: List[torch.Tensor], trim_fraction: float = 0.2) -> List[torch.Tensor]:
    """
    Trims the smallest coordinates by absolute magnitude for each tensor.
    Retains the top (1 - trim_fraction) fraction of entries; resets the bottom entries to zero.

    Args:
        tensors: List of K tensors sharing the same shape.
        trim_fraction: Fraction p in [0.0, 1.0) of entries to zero out.

    Returns:
        List of K trimmed tensors.
    """
    if not (0.0 <= trim_fraction < 1.0):
        raise ValueError(f"trim_fraction must be in [0.0, 1.0), got {trim_fraction}")
        
    if trim_fraction == 0.0:
        return [t.clone() for t in tensors]
        
    trimmed = []
    for t in tensors:
        flat = t.abs().view(-1)
        k_count = int(trim_fraction * flat.numel())
        if k_count > 0:
            threshold = torch.kthvalue(flat, k_count).values
            mask = t.abs() > threshold
            trimmed.append(torch.where(mask, t, torch.zeros_like(t)))
        else:
            trimmed.append(t.clone())
            
    return trimmed


def elect_sign(trimmed_tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Elects the dominant sign for each coordinate across all tasks:
        S = sgn(sum_{k=1}^K T_k^{trim})

    Args:
        trimmed_tensors: List of K trimmed tensors.

    Returns:
        Tensor of coordinate-wise elected signs in {-1.0, 0.0, +1.0}.
    """
    stacked = torch.stack(trimmed_tensors, dim=0)
    summed = stacked.sum(dim=0)
    return torch.sign(summed)


def disjoint_merge(
    trimmed_tensors: List[torch.Tensor],
    elected_sign: torch.Tensor,
    weights: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Disjointly averages entries that agree with the elected sign.

    Args:
        trimmed_tensors: List of K trimmed tensors.
        elected_sign: Tensor of elected signs.
        weights: Optional list of K scalar weights. If None, computes unweighted disjoint mean.

    Returns:
        Merged tensor.
    """
    K = len(trimmed_tensors)
    stacked = torch.stack(trimmed_tensors, dim=0) # (K, ...)
    
    # Agreement mask: sign matches elected sign AND value is non-zero
    signs = torch.sign(stacked)
    agreement_mask = (signs == elected_sign.unsqueeze(0)) & (signs != 0.0)
    
    if weights is None:
        valid_counts = agreement_mask.sum(dim=0).clamp(min=1)
        summed_agreed = (stacked * agreement_mask).sum(dim=0)
        has_any_match = agreement_mask.any(dim=0)
        merged = torch.where(has_any_match, summed_agreed / valid_counts.to(stacked.dtype), torch.zeros_like(elected_sign))
    else:
        w_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device).view(K, *([1] * (stacked.dim() - 1)))
        weighted_agreed = (stacked * agreement_mask) * w_tensor
        summed_agreed = weighted_agreed.sum(dim=0)
        weight_sum = (agreement_mask * w_tensor).sum(dim=0).clamp(min=1e-8)
        has_any_match = agreement_mask.any(dim=0)
        merged = torch.where(has_any_match, summed_agreed / weight_sum, torch.zeros_like(elected_sign))
        
    return merged


def ties_merge(
    tensors: List[torch.Tensor],
    trim_fraction: float = 0.2,
    weights: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Executes full TIES-merging pipeline: Trim -> Elect Sign -> Disjoint Merge.

    Args:
        tensors: List of K tensors of matching shape.
        trim_fraction: Fraction of parameters to trim per task.
        weights: Optional task weights.

    Returns:
        Merged tensor.
    """
    trimmed = trim(tensors, trim_fraction=trim_fraction)
    elected_sign = elect_sign(trimmed)
    merged = disjoint_merge(trimmed, elected_sign, weights=weights)
    return merged
