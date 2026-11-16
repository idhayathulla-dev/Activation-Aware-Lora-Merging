# Algorithmic Comparison & Complexity Analysis: OSSC vs. Baselines

## 1. Algorithmic Pseudocode Comparison

### 1.1 Baseline 1: Task Arithmetic (Ilharco et al., ICLR 2023)
```python
def task_arithmetic(task_updates: List[Tensor], weights: List[float]) -> Tensor:
    # Linear combination in raw parameter space
    return sum(w * dW for w, dW in zip(weights, task_updates))
```

### 1.2 Baseline 2: TIES-Merging (Yadav et al., NeurIPS 2023)
```python
def ties_merging(task_updates: List[Tensor], trim_fraction: float) -> Tensor:
    # 1. Trim bottom-p% coordinates by absolute magnitude
    trimmed = [trim(dW, trim_fraction) for dW in task_updates]
    # 2. Elect dominant sign
    elected_sign = torch.sign(sum(trimmed))
    # 3. Disjoint average over coordinates agreeing with elected sign
    return disjoint_average(trimmed, elected_sign)
```

### 1.3 Baseline 3: EMR-Merging (Huang et al., NeurIPS 2024 Spotlight)
```python
def emr_merging(task_updates: List[Tensor]) -> Tensor:
    # 1. Elect unified backbone via max absolute coordinate
    backbone = elect_backbone(task_updates)
    # 2. Mask opposing directional updates
    masks = [torch.sign(dW) == torch.sign(backbone) for dW in task_updates]
    # 3. Rescale task updates to match energy
    return rescale_and_aggregate(task_updates, masks)
```

### 1.4 Baseline 4: KnOTS + Task Arithmetic (Stoica et al., ICLR 2025)
```python
def knots_task_arithmetic(task_updates: List[Tensor], rank_budget: int) -> Tuple[Tensor, Tensor]:
    # 1. Horizontal concatenation
    W_cat = torch.cat(task_updates, dim=1) # (d_out, K * d_in)
    # 2. Truncated SVD
    U_R, S_R, Vh_R = truncated_svd(W_cat, rank_budget)
    # 3. Partition right singular vectors into aligned coordinates
    Z = [torch.diag(S_R) @ Vh_R[:, i*d_in : (i+1)*d_in] for i in range(K)]
    # 4. Aligned Task Arithmetic
    Z_merged = sum(Z) / K
    # 5. Factorize into rank-R LoRA
    lora_B = U_R
    lora_A = Z_merged
    return lora_B, lora_A
```

### 1.5 Baseline 5: KnOTS + TIES (Stoica et al., ICLR 2025)
```python
def knots_ties(task_updates: List[Tensor], rank_budget: int, trim_fraction: float) -> Tuple[Tensor, Tensor]:
    # Steps 1-3 identical to KnOTS
    U_R, S_R, Vh_R = truncated_svd(torch.cat(task_updates, dim=1), rank_budget)
    Z = [torch.diag(S_R) @ Vh_R[:, i*d_in : (i+1)*d_in] for i in range(K)]
    # 4. Aligned TIES
    Z_merged = ties_merging(Z, trim_fraction)
    return U_R, Z_merged
```

### 1.6 OSSC Formulation 1: KnOTS + EMR
```python
def ossc_formulation_1(task_updates: List[Tensor], rank_budget: int) -> Tuple[Tensor, Tensor]:
    # 1. Horizontal concatenation (identical to KnOTS)
    W_cat = torch.cat(task_updates, dim=1)
    # 2. Truncated SVD (identical to KnOTS)
    U_R, S_R, Vh_R = truncated_svd(W_cat, rank_budget)
    # 3. Partition right singular vectors into aligned coordinates (identical to KnOTS)
    Z = [torch.diag(S_R) @ Vh_R[:, i*d_in : (i+1)*d_in] for i in range(K)]
    # 4. Apply EMR sign consensus in aligned coordinate space
    Z_merged = emr_merging(Z)
    # 5. Factorize into rank-R LoRA
    return U_R, Z_merged
```

### 1.7 OSSC Formulation 2: Two-Sided Orthogonal Projection + EMR
```python
def ossc_formulation_2(task_updates: List[Tensor], rank_budget: int) -> Tuple[Tensor, Tensor]:
    # 1. Extract shared left and right orthonormal bases
    U_R, _, V_R = compute_biorthogonal_bases(task_updates, rank_budget) # U_R: (d_out, R), V_R: (d_in, R)
    # 2. Project into R x R core tensors
    C = [U_R.T @ dW @ V_R for dW in task_updates] # K tensors of shape (R, R)
    # 3. Core-space EMR
    C_merged = emr_merging(C) # (R, R)
    # 4. SVD of core tensor to factorize into rank-R LoRA
    P, S_core, Qh = torch.linalg.svd(C_merged)
    lora_B = U_R @ P @ torch.diag(torch.sqrt(S_core))
    lora_A = torch.diag(torch.sqrt(S_core)) @ Qh @ V_R.T
    return lora_B, lora_A
```

### 1.8 Activation-Aware KnOTS (Proposed Research Method)
```python
def activation_aware_knots(task_updates: List[Tensor], C_lambda: Tensor, rank_budget: int, op: str) -> Tuple[Tensor, Tensor]:
    # 1. Compute regularized matrix square roots
    C_sqrt, C_inv_sqrt = compute_matrix_square_roots(C_lambda)
    # 2. Transform updates into canonical activation coordinates
    transformed_updates = [dW @ C_sqrt for dW in task_updates]
    # 3. KnOTS alignment in transformed coordinates
    W_cat_tilde = torch.cat(transformed_updates, dim=1)
    U_R, S_R, Vh_R = truncated_svd(W_cat_tilde, rank_budget)
    Z_tilde = [torch.diag(S_R) @ Vh_R[:, i*d_in : (i+1)*d_in] for i in range(K)]
    # 4. Merge in aligned space (TA or TIES)
    Z_merged_tilde = merge_operator(Z_tilde, op)
    # 5. Invert transformation back to parameter space and factorize
    lora_B = U_R
    lora_A = Z_merged_tilde @ C_inv_sqrt
    return lora_B, lora_A
```

---

## 2. Theoretical Complexity & Resource Comparison

| Method | SVD Flops | Merging Flops | Peak Memory (Intermediate) | Output Rank | Factorized Output? | Requires Data? |
|---|---|---|---|---|---|---|
| **Task Arithmetic** | $0$ | $\mathcal{O}(K \cdot d_{out} d_{in})$ | $\mathcal{O}(d_{out} d_{in})$ | $\min(d_{out}, \sum r_i)$ | No (Dense $\Delta W$) | No |
| **TIES-Merging** | $0$ | $\mathcal{O}(K \cdot d_{out} d_{in} \log(d))$ | $\mathcal{O}(K \cdot d_{out} d_{in})$ | $\min(d_{out}, \sum r_i)$ | No (Dense $\Delta W$) | No |
| **EMR-Merging** | $0$ | $\mathcal{O}(K \cdot d_{out} d_{in})$ | $\mathcal{O}(K \cdot d_{out} d_{in})$ | $\min(d_{out}, \sum r_i)$ | No (Dense $\Delta W$) | No |
| **KnOTS + TA** | $\mathcal{O}(d_{out}^2 \cdot K d_{in})$ | $\mathcal{O}(K \cdot R d_{in})$ | $\mathcal{O}(K \cdot d_{out} d_{in})$ | $R$ | **Yes** ($B \in \mathbb{R}^{d \times R}, A \in \mathbb{R}^{R \times d}$) | No |
| **KnOTS + TIES** | $\mathcal{O}(d_{out}^2 \cdot K d_{in})$ | $\mathcal{O}(K \cdot R d_{in} \log(R d_{in}))$ | $\mathcal{O}(K \cdot d_{out} d_{in})$ | $R$ | **Yes** | No |
| **OSSC Form. 1 (KnOTS+EMR)** | $\mathcal{O}(d_{out}^2 \cdot K d_{in})$ | $\mathcal{O}(K \cdot R d_{in})$ | $\mathcal{O}(K \cdot d_{out} d_{in})$ | $R$ | **Yes** | No |
| **OSSC Form. 2 (Two-Sided+EMR)**| $\mathcal{O}(d_{out}^2 d_{in} + d_{in}^2 d_{out})$ | $\mathcal{O}(K \cdot R^2)$ | $\mathcal{O}(d_{out} R + d_{in} R)$ | $R$ | **Yes** | No |
| **Activation-Aware KnOTS** | $\mathcal{O}(d_{in}^3 + d_{out}^2 \cdot K d_{in})$| $\mathcal{O}(K \cdot R d_{in})$ | $\mathcal{O}(d_{in}^2 + K \cdot d_{out} d_{in})$| $R$ | **Yes** | Yes (Calibration activations) |

### Key Algorithmic Insights:
1. **Computational Identity:** OSSC Formulation 1 and KnOTS+TIES share the identical SVD computational bottleneck ($\mathcal{O}(d_{out}^2 \cdot K d_{in})$). EMR replaces TIES's sorting-based trim with an element-wise maximum absolute value selection ($\mathcal{O}(K \cdot R d_{in})$ vs $\mathcal{O}(K \cdot R d_{in} \log(R d_{in}))$), yielding a negligible arithmetic speedup in the merge step that is dwarfed by the SVD decomposition time.
2. **Formulation 2 Advantage:** Formulation 2 avoids the horizontal expansion to $K \cdot d_{in}$, executing sign consensus on a tiny $R \times R$ core tensor. However, it requires computing both left and right projection bases, and severely constrains capacity by imposing an $R \times R$ bottleneck ($R^2$ parameters vs $R \cdot d_{in}$ parameters).
