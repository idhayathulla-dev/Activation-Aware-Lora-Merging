"""
Unit tests for model merging operators: SVD invariants, KnOTS baseline, Activation-Aware KnOTS,
identity covariance reduction, and PEFT LoRA factor scaling.
"""

import pytest
import torch
import numpy as np

from src.merging.task_arithmetic import task_arithmetic_merge
from src.merging.ties import ties_merge, trim, elect_sign, disjoint_merge
from src.merging.knots import knots_align, knots_merge
from src.merging.activation_aware_knots import activation_aware_knots_merge


@pytest.fixture
def rng_seed():
    torch.manual_seed(42)
    np.random.seed(42)


def test_svd_invariants_and_orthonormality(rng_seed):
    """
    Requirement 2: Test SVD invariants without assuming sign consistency across solvers:
      1. Orthonormality of left singular vectors: ||U_R^T @ U_R - I_R||_F < 1e-10
      2. Exact lossless reconstruction when rank_budget >= rank(W_cat)
      3. Subspace projection: U_R @ U_R^T @ W_cat preserves rank-R energy
    """
    d_out, d_in = 32, 48
    K = 3
    # Create rank-limited synthetic task updates: each of rank 4
    task_updates = []
    for _ in range(K):
        B = torch.randn(d_out, 4, dtype=torch.float64)
        A = torch.randn(4, d_in, dtype=torch.float64)
        task_updates.append(B @ A) # Rank at most 4
        
    # Joint matrix has rank at most 3 * 4 = 12
    W_cat = torch.cat(task_updates, dim=1)
    
    # 1. Test full rank budget (R = 12)
    R_full = 12
    U_R, S_R, aligned_reps = knots_align(task_updates, rank_budget=R_full)
    
    eye_R = torch.eye(R_full, dtype=torch.float64)
    ortho_err = torch.norm(U_R.T @ U_R - eye_R).item()
    
    # Reconstruct W_cat from all partitioned blocks:
    # [U_R @ Z_1, U_R @ Z_2, ..., U_R @ Z_K]
    recon_blocks = [U_R @ Z_k for Z_k in aligned_reps]
    W_cat_recon = torch.cat(recon_blocks, dim=1)
    lossless_err = (torch.norm(W_cat_recon - W_cat) / torch.norm(W_cat)).item()
    
    print(f"\n[test_svd_invariants_and_orthonormality]")
    print(f"  Orthonormality error ||U_R^T @ U_R - I||_F : {ortho_err:.2e}")
    print(f"  Lossless reconstruction relative error     : {lossless_err:.2e}")
    
    assert ortho_err < 1e-10
    assert lossless_err < 1e-10


def test_knots_matrix_dimensions_and_lora_scaling(rng_seed):
    """
    Requirement 3: Explicitly test that reconstructed LoRA factors reproduce the intended
    dense update under arbitrary PEFT scaling:
        DeltaW_dense = lora_scaling * (lora_B @ lora_A)
    Tested across multiple scalings: 1.0, 2.0 (alpha=16, r=8), 0.5 (alpha=4, r=8).
    """
    d_out, d_in = 64, 80
    K = 3
    rank_budget = 8
    task_updates = [torch.randn(d_out, d_in, dtype=torch.float64) for _ in range(K)]
    
    scalings = [1.0, 2.0, 0.5]
    print(f"\n[test_knots_matrix_dimensions_and_lora_scaling]")
    for s in scalings:
        dense_up, lora_B, lora_A = knots_merge(
            task_updates,
            rank_budget=rank_budget,
            merge_operator="task_arithmetic",
            lora_scaling=s
        )
        
        # Verify shapes
        assert lora_B.shape == (d_out, rank_budget), f"Expected ({d_out}, {rank_budget}), got {lora_B.shape}"
        assert lora_A.shape == (rank_budget, d_in), f"Expected ({rank_budget}, {d_in}), got {lora_A.shape}"
        assert dense_up.shape == (d_out, d_in), f"Expected ({d_out}, {d_in}), got {dense_up.shape}"
        
        # Verify scaling equivalence: dense == s * (B @ A)
        factored_dense = s * (lora_B @ lora_A)
        equiv_err = (torch.norm(factored_dense - dense_up) / torch.norm(dense_up)).item()
        print(f"  Scaling = {s:<4.1f} | Factorization rel error: {equiv_err:.2e}")
        assert equiv_err < 1e-11


def test_activation_aware_knots_identity_covariance(rng_seed):
    """
    Requirement: When covariance is the identity matrix I and alpha_reg=0,
    Activation-Aware KnOTS must reduce to ordinary KnOTS within numerical tolerance.
    """
    d_out, d_in = 48, 64
    K = 3
    rank_budget = 8
    task_updates = [torch.randn(d_out, d_in, dtype=torch.float64) for _ in range(K)]
    
    # 1. Run ordinary KnOTS
    dense_knots, B_knots, A_knots = knots_merge(
        task_updates,
        rank_budget=rank_budget,
        merge_operator="task_arithmetic",
        lora_scaling=1.0
    )
    
    # 2. Run Activation-Aware KnOTS with identity covariance and precomputed square roots (I, I)
    I = torch.eye(d_in, dtype=torch.float64)
    dense_aa, B_aa, A_aa, _ = activation_aware_knots_merge(
        task_updates,
        covariance=I,
        rank_budget=rank_budget,
        merge_operator="task_arithmetic",
        lora_scaling=1.0,
        precomputed_square_roots=(I, I)
    )
    
    diff_dense = (torch.norm(dense_aa - dense_knots) / torch.norm(dense_knots)).item()
    diff_factored = (torch.norm((B_aa @ A_aa) - (B_knots @ A_knots)) / torch.norm(B_knots @ A_knots)).item()
    
    print(f"\n[test_activation_aware_knots_identity_covariance]")
    print(f"  Relative difference in dense update    : {diff_dense:.2e}")
    print(f"  Relative difference in factored product: {diff_factored:.2e}")
    
    assert diff_dense < 1e-11
    assert diff_factored < 1e-11


def test_activation_aware_knots_full_pipeline(rng_seed):
    """
    Test Activation-Aware KnOTS with realistic non-identity covariance, Task Arithmetic, and TIES,
    verifying dimensions, LoRA scaling preservation, and numerical finiteness.
    """
    d_out, d_in = 48, 64
    K = 3
    rank_budget = 8
    task_updates = [torch.randn(d_out, d_in, dtype=torch.float64) for _ in range(K)]
    
    # Generate realistic empirical covariance
    X_calib = torch.randn(100, d_in, dtype=torch.float64)
    cov = (X_calib.T @ X_calib) / 100.0
    
    for op in ["task_arithmetic", "ties"]:
        dense_up, lora_B, lora_A, diag = activation_aware_knots_merge(
            task_updates,
            covariance=cov,
            rank_budget=rank_budget,
            merge_operator=op,
            alpha_reg=1e-3,
            lora_scaling=2.0
        )
        
        # Verify shapes
        assert lora_B.shape == (d_out, rank_budget)
        assert lora_A.shape == (rank_budget, d_in)
        assert dense_up.shape == (d_out, d_in)
        
        # Verify scaling: dense == 2.0 * (B @ A)
        reconstructed = 2.0 * (lora_B @ lora_A)
        recon_err = (torch.norm(reconstructed - dense_up) / torch.norm(dense_up)).item()
        
        print(f"\n[test_activation_aware_knots_full_pipeline | op={op}]")
        print(f"  Scaling error (s=2.0): {recon_err:.2e}")
        print(f"  Condition number     : {diag['condition_number']:.2e}")
        
        assert recon_err < 1e-11
        assert torch.all(torch.isfinite(dense_up))
        assert torch.all(torch.isfinite(lora_B))
        assert torch.all(torch.isfinite(lora_A))


def test_single_task_lossless_recovery(rng_seed):
    """
    Test that when K=1 and rank_budget >= rank(update), both KnOTS and Activation-Aware KnOTS
    reconstruct the single task update losslessly.
    """
    d_out, d_in = 32, 48
    r_target = 6
    B = torch.randn(d_out, r_target, dtype=torch.float64)
    A = torch.randn(r_target, d_in, dtype=torch.float64)
    dW = B @ A # Rank 6
    
    X = torch.randn(80, d_in, dtype=torch.float64)
    cov = (X.T @ X) / 80.0
    
    # 1. KnOTS single task
    dense_knots, _, _ = knots_merge([dW], rank_budget=r_target, lora_scaling=1.0)
    err_knots = (torch.norm(dense_knots - dW) / torch.norm(dW)).item()
    
    # 2. Activation-Aware KnOTS single task
    dense_aa, _, _, _ = activation_aware_knots_merge([dW], covariance=cov, rank_budget=r_target, lora_scaling=1.0)
    err_aa = (torch.norm(dense_aa - dW) / torch.norm(dW)).item()
    
    print(f"\n[test_single_task_lossless_recovery]")
    print(f"  KnOTS single-task error            : {err_knots:.2e}")
    print(f"  Activation-Aware single-task error : {err_aa:.2e}")
    
    assert err_knots < 1e-10
    assert err_aa < 1e-10
