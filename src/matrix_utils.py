"""
Matrix Utilities: Covariance estimation, regularized square roots, and objective evaluation.

This module provides numerically stable routines for computing uncentered input second moments,
task-balanced covariance estimation, symmetric eigendecomposition, matrix square roots, and
verifying the regularized activation-weighted objective.
"""

from typing import Dict, List, Optional, Tuple
import torch


def compute_uncentered_covariance(X: torch.Tensor) -> torch.Tensor:
    """
    Computes the empirical uncentered second moment matrix C_x = (1 / N) * X^T @ X.

    Args:
        X: Input activation tensor of shape (N, d_in), where N is token count.

    Returns:
        Uncentered second moment matrix of shape (d_in, d_in) in float64.
    """
    if X.dim() != 2:
        raise ValueError(f"Expected 2D tensor (N, d_in), got shape {X.shape}")
    N, d_in = X.shape
    if N == 0:
        raise ValueError("Cannot compute covariance from an empty activation tensor (N=0).")
    
    X_64 = X.to(torch.float64)
    C = (X_64.T @ X_64) / float(N)
    return C


def compute_task_balanced_covariance(task_activations: List[torch.Tensor]) -> torch.Tensor:
    """
    Computes the task-balanced uncentered second moment matrix:
        C_x_balanced = (1 / K) * sum_{k=1}^K (1 / N_k) * (X_k^T @ X_k)

    This ensures equal geometric weighting across tasks regardless of token sample counts.

    Args:
        task_activations: List of K tensors, each of shape (N_k, d_in).

    Returns:
        Task-balanced covariance matrix of shape (d_in, d_in) in float64.
    """
    if not task_activations:
        raise ValueError("task_activations list cannot be empty.")
    
    K = len(task_activations)
    d_in = task_activations[0].shape[1]
    
    C_total = torch.zeros((d_in, d_in), dtype=torch.float64, device=task_activations[0].device)
    for k, X_k in enumerate(task_activations):
        if X_k.dim() != 2 or X_k.shape[1] != d_in:
            raise ValueError(
                f"Task {k} activation shape {X_k.shape} does not match expected (N_k, {d_in})."
            )
        C_k = compute_uncentered_covariance(X_k)
        C_total += C_k
        
    return C_total / float(K)


def regularize_covariance(
    C: torch.Tensor,
    alpha_reg: float = 1e-3,
    eps: float = 1e-12
) -> Tuple[torch.Tensor, float]:
    """
    Applies symmetric Tikhonov regularization to covariance matrix:
        C_lambda = C_sym + lambda * I
        where lambda = alpha_reg * mean(diag(C_sym))

    Args:
        C: Covariance tensor of shape (d_in, d_in).
        alpha_reg: Regularization strength relative to mean diagonal energy.
        eps: Minimum allowable value for lambda.

    Returns:
        Tuple of (C_lambda in float64, scalar lambda).
    """
    if C.dim() != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Expected square covariance matrix, got shape {C.shape}")
        
    C_64 = C.to(torch.float64)
    C_sym = 0.5 * (C_64 + C_64.T)
    d_in = C_sym.shape[0]
    
    mean_diag = torch.diag(C_sym).mean().item()
    if not torch.isfinite(torch.tensor(mean_diag)):
        raise ValueError(f"Non-finite mean diagonal encountered in covariance: {mean_diag}")
        
    lam = max(alpha_reg * max(mean_diag, 0.0), eps)
    eye = torch.eye(d_in, dtype=torch.float64, device=C.device)
    C_lambda = C_sym + lam * eye
    return C_lambda, lam


def compute_matrix_square_roots(
    C: torch.Tensor,
    alpha_reg: float = 1e-3,
    eps: float = 1e-12,
    return_diagnostics: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Computes regularized C_lambda^{1/2} and C_lambda^{-1/2} via symmetric eigendecomposition.

    C_lambda = Q @ diag(mu) @ Q^T
    C_lambda^{1/2} = Q @ diag(sqrt(mu)) @ Q^T
    C_lambda^{-1/2} = Q @ diag(1 / sqrt(mu)) @ Q^T

    Args:
        C: Covariance matrix of shape (d_in, d_in).
        alpha_reg: Tikhonov regularization coefficient.
        eps: Tolerance floor for clamping negative or degenerate eigenvalues.
        return_diagnostics: Whether to return numerical conditioning metrics.

    Returns:
        Tuple of (C_sqrt, C_inv_sqrt, diagnostics_dict).
        Tensors are returned in float64 for mathematical fidelity.
    """
    C_lambda, lam = regularize_covariance(C, alpha_reg=alpha_reg, eps=eps)
    d_in = C_lambda.shape[0]
    
    # Compute symmetric eigendecomposition
    eigenvalues, Q = torch.linalg.eigh(C_lambda)
    
    max_eig = eigenvalues.max().item()
    min_eig = eigenvalues.min().item()
    
    # Numerical safeguard: clamp any numerically non-positive or tiny eigenvalues
    min_allowable = max(eps * max_eig, eps)
    clamped_eigs = torch.clamp(eigenvalues, min=min_allowable)
    
    sqrt_eigs = torch.sqrt(clamped_eigs)
    inv_sqrt_eigs = 1.0 / sqrt_eigs
    
    C_sqrt = (Q * sqrt_eigs.unsqueeze(0)) @ Q.T
    C_inv_sqrt = (Q * inv_sqrt_eigs.unsqueeze(0)) @ Q.T
    
    # Enforce exact symmetry
    C_sqrt = 0.5 * (C_sqrt + C_sqrt.T)
    C_inv_sqrt = 0.5 * (C_inv_sqrt + C_inv_sqrt.T)
    
    # Diagnostic checks
    inv_error = torch.norm(C_sqrt @ C_inv_sqrt - torch.eye(d_in, dtype=torch.float64, device=C.device)).item()
    cond_num = max_eig / max(clamped_eigs.min().item(), 1e-15)
    
    diagnostics = {
        "lambda": lam,
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "condition_number": cond_num,
        "inversion_error_frobenius": inv_error,
    }
    
    return C_sqrt, C_inv_sqrt, diagnostics


def compute_regularized_objective(E: torch.Tensor, C_lambda: torch.Tensor) -> float:
    """
    Computes the regularized activation-weighted objective:
        ||E @ C_lambda^{1/2}||_F^2 = Tr(E @ C_lambda @ E^T)

    Args:
        E: Parameter error matrix (DeltaW - DeltaW_hat) of shape (d_out, d_in).
        C_lambda: Regularized covariance of shape (d_in, d_in).

    Returns:
        Objective scalar value in float64.
    """
    E_64 = E.to(torch.float64)
    C_64 = C_lambda.to(torch.float64)
    # Tr(E @ C @ E^T) = sum((E @ C) * E)
    val = torch.sum((E_64 @ C_64) * E_64).item()
    return val


def compute_empirical_task_balanced_error(
    E: torch.Tensor,
    task_activations: List[torch.Tensor],
    lam: float
) -> float:
    """
    Computes the exact empirical task-balanced output reconstruction error plus Tikhonov penalty:
        (1 / K) * sum_{k=1}^K (1 / N_k) * sum_{n=1}^{N_k} ||E @ x_{n, k}||_2^2 + lambda * ||E||_F^2

    Args:
        E: Parameter error matrix of shape (d_out, d_in).
        task_activations: List of K tensors, each of shape (N_k, d_in).
        lam: Regularization parameter lambda.

    Returns:
        Empirical error scalar value in float64.
    """
    E_64 = E.to(torch.float64)
    K = len(task_activations)
    total_task_error = 0.0
    
    for X_k in task_activations:
        X_k_64 = X_k.to(torch.float64)
        N_k = X_k.shape[0]
        # Layer output error for all tokens in task k: (N_k, d_out)
        out_error = X_k_64 @ E_64.T
        task_error = torch.sum(out_error ** 2).item() / float(N_k)
        total_task_error += task_error
        
    avg_output_error = total_task_error / float(K)
    frob_penalty = lam * (torch.norm(E_64) ** 2).item()
    
    return avg_output_error + frob_penalty
