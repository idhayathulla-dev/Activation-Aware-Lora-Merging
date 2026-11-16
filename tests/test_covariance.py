"""
Unit tests for empirical covariance estimation, rank deficiency, and numerical stability.
"""

import pytest
import torch
import numpy as np

from src.matrix_utils import (
    compute_uncentered_covariance,
    compute_task_balanced_covariance,
    regularize_covariance,
    compute_matrix_square_roots,
)


@pytest.fixture
def rng_seed():
    torch.manual_seed(42)
    np.random.seed(42)


def test_rank_deficient_covariance(rng_seed):
    """
    Test behavior when token sample count N is much smaller than hidden dimension d_in (N < d_in).
    Unregularized covariance has rank at most N.
    Regularization must guarantee strict positive definiteness and stable inversion.
    """
    d_in = 64
    N = 12  # Severely rank-deficient: rank <= 12 < 64
    X = torch.randn(N, d_in, dtype=torch.float64)
    
    C = compute_uncentered_covariance(X)
    eigs_unreg = torch.linalg.eigvalsh(C)
    # Check that unregularized covariance has near-zero eigenvalues
    num_near_zero = (eigs_unreg.abs() < 1e-10).sum().item()
    assert num_near_zero >= (d_in - N), f"Expected at least {d_in - N} near-zero eigs, got {num_near_zero}"
    
    # Regularize
    alpha_reg = 1e-3
    C_sqrt, C_inv_sqrt, diag = compute_matrix_square_roots(C, alpha_reg=alpha_reg)
    
    eye = torch.eye(d_in, dtype=torch.float64)
    inv_err = torch.norm(C_sqrt @ C_inv_sqrt - eye).item()
    
    print(f"\n[test_rank_deficient_covariance]")
    print(f"  Unregularized near-zero eigenvalues : {num_near_zero} / {d_in}")
    print(f"  Condition number after reg          : {diag['condition_number']:.2e}")
    print(f"  Inversion error ||C^(1/2) C^(-1/2) - I||_F : {inv_err:.2e}")
    
    assert torch.all(torch.isfinite(C_sqrt))
    assert torch.all(torch.isfinite(C_inv_sqrt))
    assert inv_err < 1e-10


def test_regularization_parameter_sweep(rng_seed):
    """
    Test numerical stability across a wide sweep of regularization coefficients:
    alpha_reg in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1].
    Evaluates condition numbers and inversion errors.
    """
    d_in = 48
    N = 64
    X = torch.randn(N, d_in, dtype=torch.float64)
    C = compute_uncentered_covariance(X)
    
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    results = {}
    
    print(f"\n[test_regularization_parameter_sweep]")
    for alpha in alphas:
        C_sqrt, C_inv_sqrt, diag = compute_matrix_square_roots(C, alpha_reg=alpha)
        eye = torch.eye(d_in, dtype=torch.float64)
        inv_err = torch.norm(C_sqrt @ C_inv_sqrt - eye).item()
        results[alpha] = (diag["condition_number"], inv_err)
        print(f"  alpha = {alpha:<8.1e} | Cond: {diag['condition_number']:<10.2e} | Inv Err: {inv_err:<10.2e}")
        
        assert torch.all(torch.isfinite(C_sqrt))
        assert torch.all(torch.isfinite(C_inv_sqrt))
        assert inv_err < 1e-8


def test_task_balanced_vs_pooled_difference(rng_seed):
    """
    Verifies that task-balanced covariance strictly differs from naive pooled covariance
    when sample counts are unequal across tasks.
    """
    d_in = 32
    # Tasks with highly imbalanced sample counts
    task_activations = [
        torch.randn(10, d_in, dtype=torch.float64),
        torch.randn(200, d_in, dtype=torch.float64),
    ]
    
    # Task-balanced: equal 50% weight per task
    C_balanced = compute_task_balanced_covariance(task_activations)
    
    # Naive pooled: (X_cat^T @ X_cat) / (10 + 200) -> heavily biased toward task 2 (200/210 = 95%)
    X_pooled = torch.cat(task_activations, dim=0)
    C_pooled = compute_uncentered_covariance(X_pooled)
    
    diff = torch.norm(C_balanced - C_pooled).item()
    print(f"\n[test_task_balanced_vs_pooled_difference]")
    print(f"  ||C_balanced - C_pooled||_F : {diff:.4f}")
    
    assert diff > 0.1, "Task-balanced and pooled covariances should differ significantly under sample imbalance."
