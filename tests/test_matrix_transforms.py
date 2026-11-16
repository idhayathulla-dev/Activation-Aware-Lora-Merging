"""
Unit tests for matrix transformations, regularized objective decomposition, and square root invariants.
"""

import pytest
import torch
import numpy as np

from src.matrix_utils import (
    compute_uncentered_covariance,
    compute_task_balanced_covariance,
    regularize_covariance,
    compute_matrix_square_roots,
    compute_regularized_objective,
    compute_empirical_task_balanced_error,
)


@pytest.fixture
def rng_seed():
    torch.manual_seed(42)
    np.random.seed(42)


def test_regularized_objective_decomposition(rng_seed):
    """
    Requirement 1: Verify that ||E @ C_lambda^{1/2}||_F^2 exactly equals the task-balanced
    empirical reconstruction error plus Tikhonov weight-space penalty:
        (1/K) * sum_{k=1}^K (1/N_k) * sum_{n=1}^{N_k} ||E @ x_{n, k}||_2^2 + lambda * ||E||_F^2
    specifically under UNEQUAL task sample counts (N_1 != N_2 != N_3).
    """
    d_out = 32
    d_in = 64
    K = 3
    # Unequal task sample counts
    sample_counts = [37, 83, 51]
    
    task_activations = [torch.randn(n, d_in, dtype=torch.float64) for n in sample_counts]
    E = torch.randn(d_out, d_in, dtype=torch.float64)
    
    # Compute task-balanced covariance
    C_balanced = compute_task_balanced_covariance(task_activations)
    alpha_reg = 1e-3
    C_lambda, lam = regularize_covariance(C_balanced, alpha_reg=alpha_reg)
    
    # Objective via matrix square root / trace: Tr(E @ C_lambda @ E^T)
    C_sqrt, _, _ = compute_matrix_square_roots(C_balanced, alpha_reg=alpha_reg)
    obj_via_sqrt = (torch.norm(E @ C_sqrt) ** 2).item()
    obj_via_trace = compute_regularized_objective(E, C_lambda)
    
    # Objective via explicit token-by-token task-balanced forward error
    empirical_err = compute_empirical_task_balanced_error(E, task_activations, lam)
    
    rel_diff_sqrt_trace = abs(obj_via_sqrt - obj_via_trace) / obj_via_trace
    rel_diff_trace_emp = abs(obj_via_trace - empirical_err) / empirical_err
    
    print(f"\n[test_regularized_objective_decomposition]")
    print(f"  obj_via_sqrt   : {obj_via_sqrt:.10f}")
    print(f"  obj_via_trace  : {obj_via_trace:.10f}")
    print(f"  empirical_err  : {empirical_err:.10f}")
    print(f"  rel_diff (sqrt vs trace) : {rel_diff_sqrt_trace:.2e}")
    print(f"  rel_diff (trace vs emp)  : {rel_diff_trace_emp:.2e}")
    
    assert rel_diff_sqrt_trace < 1e-9, f"Sqrt vs Trace discrepancy: {rel_diff_sqrt_trace}"
    assert rel_diff_trace_emp < 1e-9, f"Trace vs Empirical discrepancy: {rel_diff_trace_emp}"


def test_matrix_square_root_invariants(rng_seed):
    """
    Requirement 2 & 4: Test algebraic invariants of symmetric square roots in float64:
      1. Symmetry: C_sqrt == C_sqrt.T
      2. Invertibility: ||C_sqrt @ C_inv_sqrt - I||_F < 1e-10
      3. Square reconstruction: ||C_sqrt @ C_sqrt - C_lambda||_F / ||C_lambda||_F < 1e-10
    """
    d_in = 64
    N = 128
    X = torch.randn(N, d_in, dtype=torch.float64)
    C = compute_uncentered_covariance(X)
    
    C_lambda, lam = regularize_covariance(C, alpha_reg=1e-3)
    C_sqrt, C_inv_sqrt, diag = compute_matrix_square_roots(C, alpha_reg=1e-3)
    
    # 1. Symmetry
    sym_err_sqrt = torch.norm(C_sqrt - C_sqrt.T).item()
    sym_err_inv = torch.norm(C_inv_sqrt - C_inv_sqrt.T).item()
    
    # 2. Invertibility
    eye = torch.eye(d_in, dtype=torch.float64)
    inv_err = torch.norm(C_sqrt @ C_inv_sqrt - eye).item()
    
    # 3. Square reconstruction
    recon_C = C_sqrt @ C_sqrt
    square_err = (torch.norm(recon_C - C_lambda) / torch.norm(C_lambda)).item()
    
    print(f"\n[test_matrix_square_root_invariants]")
    print(f"  Condition Number        : {diag['condition_number']:.2e}")
    print(f"  Symmetry error (sqrt)   : {sym_err_sqrt:.2e}")
    print(f"  Symmetry error (inv)    : {sym_err_inv:.2e}")
    print(f"  Invertibility error     : {inv_err:.2e}")
    print(f"  Square recon rel error  : {square_err:.2e}")
    
    assert sym_err_sqrt < 1e-12
    assert sym_err_inv < 1e-12
    assert inv_err < 1e-10
    assert square_err < 1e-10


def test_inverse_transformation_recovery(rng_seed):
    """
    Requirement 4: Test that transforming DeltaW by C_sqrt and immediately inverting with C_inv_sqrt
    reconstructs the original DeltaW matrix up to numerical tolerance.
    """
    d_out, d_in = 48, 64
    DeltaW = torch.randn(d_out, d_in, dtype=torch.float64)
    
    X = torch.randn(100, d_in, dtype=torch.float64)
    C = compute_uncentered_covariance(X)
    C_sqrt, C_inv_sqrt, _ = compute_matrix_square_roots(C, alpha_reg=1e-3)
    
    transformed = DeltaW @ C_sqrt
    recovered = transformed @ C_inv_sqrt
    
    recon_err = (torch.norm(recovered - DeltaW) / torch.norm(DeltaW)).item()
    print(f"\n[test_inverse_transformation_recovery]")
    print(f"  Reconstruction rel error : {recon_err:.2e}")
    
    assert recon_err < 1e-11
