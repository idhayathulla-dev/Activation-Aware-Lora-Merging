# Comprehensive Mathematical Unit-Test Plan

This document outlines the strict mathematical tests required to validate all matrix operations, coordinate transformations, and merging steps on synthetic data before executing any real model training.

---

## 1. Test Suite Invariants

### 1.1 Test 1: Regularized Objective Decomposition
- **Target:** Verify that for an arbitrary error matrix $E = \Delta W - \widehat{\Delta W} \in \mathbb{R}^{d_{out} \times d_{in}}$ and uncentered second moment $C_x = \frac{1}{N} X^T X$:
  $$\|E C_\lambda^{1/2}\|_F^2 \equiv \frac{1}{N} \sum_{n=1}^N \|E x_n\|_2^2 + \lambda \|E\|_F^2$$
- **Verification Condition:** Relative absolute error $< 10^{-7}$ in float64.
- **Scientific Significance:** Confirms that the regularized matrix square root transformation exactly encodes the empirical output error plus Tikhonov weight-space penalty.

### 1.2 Test 2: Matrix Square Root and Inverse Reconstruction
- **Target:** Verify symmetric eigendecomposition $C_\lambda = Q \Lambda Q^T$.
- **Verification Conditions:**
  - Symmetry: $\|C_\lambda - C_\lambda^T\|_F = 0$.
  - Invertibility: $\|C_\lambda^{1/2} C_\lambda^{-1/2} - I\|_F < 10^{-10}$.
  - Square property: $\|C_\lambda^{1/2} C_\lambda^{1/2} - C_\lambda\|_F / \|C_\lambda\|_F < 10^{-10}$.

### 1.3 Test 3: Matrix Dimensions at Every Stage
- **Target:** Confirm that throughout the KnOTS and Activation-Aware KnOTS pipelines, every intermediate tensor conforms exactly to the specification table:
  - Input: $B_k \in \mathbb{R}^{d_{out} \times r}, A_k \in \mathbb{R}^{r \times d_{in}}$
  - Dense update: $\Delta W_k \in \mathbb{R}^{d_{out} \times d_{in}}$
  - Transformed: $\widetilde{\Delta W}_k \in \mathbb{R}^{d_{out} \times d_{in}}$
  - Stacked: $\widetilde{W}_{\text{cat}} \in \mathbb{R}^{d_{out} \times (K \cdot d_{in})}$
  - Shared bases: $U_R \in \mathbb{R}^{d_{out} \times R}, \Sigma_R \in \mathbb{R}^{R \times R}, V_R^T \in \mathbb{R}^{R \times (K \cdot d_{in})}$
  - Slices: $V_{R, k}^T \in \mathbb{R}^{R \times d_{in}}$
  - Merged coordinate: $\widetilde{Z}_{\text{merged}} \in \mathbb{R}^{R \times d_{in}}$
  - Final LoRA parameters: $B_{\text{merged}} \in \mathbb{R}^{d_{out} \times R}, A_{\text{merged}} \in \mathbb{R}^{R \times d_{in}}$

### 1.4 Test 4: Shared-Basis Reconstruction Fidelity
- **Target:** Verify that when rank budget $R \ge \text{rank}(\widetilde{W}_{\text{cat}})$, the truncated SVD achieves exact lossless reconstruction:
  $$\|U_R \Sigma_R V_R^T - \widetilde{W}_{\text{cat}}\|_F < 10^{-8}$$

### 1.5 Test 5: Correct Task-Specific Partitioning
- **Target:** Verify that for each task $k \in \{1, \dots, K\}$, slicing $V_R^T[:, (k-1)d_{in} : k d_{in}]$ exactly recovers task $k$'s rank-$R$ transformed update:
  $$\|U_R \Sigma_R V_{R, k}^T - \widetilde{\Delta W}_k^{\text{rank-}R}\|_F < 10^{-8}$$

### 1.6 Test 6: Invertibility & Inverse Transformation
- **Target:** Verify that transforming $\widetilde{\Delta W} \to \widetilde{\Delta W} C_\lambda^{-1/2}$ exactly undoes $\Delta W \to \Delta W C_\lambda^{1/2}$:
  $$\|(\Delta W C_\lambda^{1/2}) C_\lambda^{-1/2} - \Delta W\|_F / \|\Delta W\|_F < 10^{-10}$$

### 1.7 Test 7: Equivalence of Dense Update and Reconstructed LoRA Product
- **Target:** Verify that the product of the output LoRA factors $B_{\text{merged}} A_{\text{merged}}$ equals the dense merged update $\Delta W_{\text{merged}}$:
  $$\|B_{\text{merged}} A_{\text{merged}} - \Delta W_{\text{merged}}\|_F < 10^{-8}$$
  with effective rank $\le R$.

### 1.8 Test 8: Numerical Stability Across Regularization Strengths
- **Target:** Evaluate condition numbers and inversion errors across $\alpha_{\text{reg}} \in \{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ on both full-rank and rank-deficient covariance matrices.
- **Verification Condition:** No NaN or infinite values; all condition numbers remain finite ($\kappa < 10^8$ for all tested $\alpha_{\text{reg}} \ge 10^{-4}$).

---

## 2. Test Execution Mapping

| Test Function | Target File | Verification Level |
|---|---|---|
| `test_regularized_objective_decomposition` | `tests/test_matrix_transforms.py` | Mathematical Proof Verification |
| `test_matrix_square_root_invariants` | `tests/test_matrix_transforms.py` | Matrix Algebra Invariant |
| `test_pipeline_matrix_dimensions` | `tests/test_merging.py` | Structural Conformance |
| `test_knots_shared_basis_and_partitioning` | `tests/test_merging.py` | Algorithm Verification |
| `test_inverse_transformation_recovery` | `tests/test_matrix_transforms.py` | Coordinate Mapping Integrity |
| `test_dense_vs_lora_factorization_equivalence` | `tests/test_lora_updates.py` | PEFT Compatibility |
| `test_covariance_numerical_stability` | `tests/test_covariance.py` | Numerical Conditioning |
