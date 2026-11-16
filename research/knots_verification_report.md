# KnOTS Algorithm Verification & Matrix Mechanics Report

## 1. Overview and Source Documents

This report documents the verification of the **KnOTS** (*Knowledge Orientation Through SVD*) algorithm based on:
1. **Original Publication:** *Model Merging with SVD to tie the KnOTS*, George Stoica, Pratik Ramesh, Boglarka Ecsedi, Leshem Choshen, Judy Hoffman. Published at the International Conference on Learning Representations (ICLR), 2025 ([Paper Link](https://proceedings.iclr.cc/paper_files/paper/2025/file/0d4f8a5109c5083b5307fcd0bddae7a7-Paper-Conference.pdf)).
2. **Official Codebase:** GitHub repository [`gstoica27/KnOTS`](https://github.com/gstoica27/KnOTS), specifically `task_merger.py`.

---

## 2. Mathematical Derivation of KnOTS Steps & Matrix Shapes

Let $K$ be the number of task models being merged.
For a target linear layer with input dimension $d_{in}$ and output dimension $d_{out}$:
Each task adapter $k \in \{1, \dots, K\}$ provides low-rank factor matrices:
- $B_k \in \mathbb{R}^{d_{out} \times r_k}$
- $A_k \in \mathbb{R}^{r_k \times d_{in}}$
with scaling factor $\gamma_k = \frac{\alpha_k}{r_k}$.
The effective task update is:
$$\Delta W_k = \gamma_k B_k A_k \in \mathbb{R}^{d_{out} \times d_{in}}$$

### Step 1: Concatenation Direction
In KnOTS, the $K$ task-specific update matrices are concatenated horizontally (along the column / input dimension):
$$W_{\text{cat}} = \left[ \Delta W_1, \; \Delta W_2, \; \dots, \; \Delta W_K \right] \in \mathbb{R}^{d_{out} \times (K \cdot d_{in})}$$

*(Note: In the official repository `task_merger.py`, `concat_across_output` determines whether concatenation occurs across output or input dimensions. For standard causal LM layers where $d_{in} = d_{out}$ or when aligning input projections, horizontal concatenation yields a joint column space with shared output basis $U$.)*

### Step 2: Joint SVD Decomposition
Singular Value Decomposition is computed on the joint matrix:
$$\text{SVD}(W_{\text{cat}}) = U \Sigma V^T$$
where:
- $U \in \mathbb{R}^{d_{out} \times d_{out}}$: Orthonormal columns spanning the shared output subspace across all $K$ tasks.
- $\Sigma \in \mathbb{R}^{d_{out} \times (K \cdot d_{in})}$: Rectangular diagonal matrix containing singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.
- $V \in \mathbb{R}^{(K \cdot d_{in}) \times (K \cdot d_{in})}$: Orthonormal columns of right singular vectors ($V^T \in \mathbb{R}^{(K \cdot d_{in}) \times (K \cdot d_{in})}$).

### Step 3: Rank Truncation to Budget $R$
Given target rank budget $R \le \min(d_{out}, K \cdot d_{in})$:
- Truncated left singular vectors: $U_R = U[:, :R] \in \mathbb{R}^{d_{out} \times R}$
- Truncated singular values: $\Sigma_R = \text{diag}(\sigma_1, \dots, \sigma_R) \in \mathbb{R}^{R \times R}$
- Truncated right singular vectors: $V_R^T = V^T[:R, :] \in \mathbb{R}^{R \times (K \cdot d_{in})}$

### Step 4: Partitioning Task-Specific Components
The matrix $V_R^T$ is partitioned horizontally into $K$ contiguous blocks of width $d_{in}$:
$$V_R^T = \left[ V_{R, 1}^T, \; V_{R, 2}^T, \; \dots, \; V_{R, K}^T \right]$$
where each $V_{R, k}^T \in \mathbb{R}^{R \times d_{in}}$ contains task $k$'s coordinates in the shared basis.

The aligned task-specific representation in the shared coordinate system is:
$$Z_k = \Sigma_R V_{R, k}^T \in \mathbb{R}^{R \times d_{in}}$$
Notice that the rank-$R$ approximation of task $k$'s original update is exactly:
$$\widehat{\Delta W}_k = U_R Z_k = U_R \Sigma_R V_{R, k}^T \in \mathbb{R}^{d_{out} \times d_{in}}$$

### Step 5: Merging in Aligned Space
Because all $K$ tasks now share the identical orthonormal basis $U_R$, merging operations are applied directly to $Z_k$ (or $V_{R, k}^T$):

#### A. Task Arithmetic:
$$Z_{\text{merged}} = \sum_{k=1}^K w_k Z_k = \Sigma_R \left( \sum_{k=1}^K w_k V_{R, k}^T \right) \in \mathbb{R}^{R \times d_{in}}$$
where $\sum_k w_k = 1$ (typically $w_k = 1/K$).

#### B. TIES-Merging:
1. **Trim:** Keep top $(1 - p)$ fraction of entries by magnitude for each task:
   $$Z_k^{\text{trim}} = \text{Trim}(Z_k, p)$$
2. **Elect Sign:** Coordinate-wise majority sign direction:
   $$S_{i, j} = \text{sgn}\left(\sum_{k=1}^K Z_{k, i, j}^{\text{trim}}\right)$$
3. **Disjoint Average:** Average entries agreeing with the elected sign:
   $$Z_{\text{merged}, i, j} = \frac{1}{|K_{(i, j)}^+|} \sum_{k \in K_{(i, j)}^+} Z_{k, i, j}^{\text{trim}} \quad \text{where} \quad K_{(i, j)}^+ = \{k : \text{sgn}(Z_{k, i, j}^{\text{trim}}) = S_{i, j}\}$$

### Step 6: Reconstruction into LoRA Format
The final dense merged update is:
$$\Delta W_{\text{merged}} = U_R Z_{\text{merged}} \in \mathbb{R}^{d_{out} \times d_{in}}$$

To deploy this as a standard PEFT LoRA module without dense expansion:
$$\Delta W_{\text{merged}} = \frac{\alpha_{\text{merged}}}{R} B_{\text{merged}} A_{\text{merged}}$$
By choosing $\alpha_{\text{merged}} = R$ (effective scale = 1.0):
- **$B_{\text{merged}} = U_R \in \mathbb{R}^{d_{out} \times R}$**
- **$A_{\text{merged}} = Z_{\text{merged}} \in \mathbb{R}^{R \times d_{in}}$**

This produces an exact rank-$R$ LoRA adapter compatible with standard forward inference.

---

## 3. Comprehensive Matrix Dimension Table

| Stage | Operation / Variable | Mathematical Shape | Example ($d_{in}=896, d_{out}=896, K=3, R=8$) |
|---|---|---|---|
| Input LoRA | Task $k$ Down-projection $B_k$ | $(d_{out}, r_k)$ | $(896, 8)$ |
| Input LoRA | Task $k$ Up-projection $A_k$ | $(r_k, d_{in})$ | $(8, 896)$ |
| Dense Update | $\Delta W_k = \frac{\alpha}{r} B_k A_k$ | $(d_{out}, d_{in})$ | $(896, 896)$ |
| KnOTS Stacking | $W_{\text{cat}} = [\Delta W_1, \dots, \Delta W_K]$ | $(d_{out}, K \cdot d_{in})$ | $(896, 2688)$ |
| SVD Left Basis | $U_R$ | $(d_{out}, R)$ | $(896, 8)$ |
| SVD Singular Values | $\Sigma_R$ | $(R, R)$ | $(8, 8)$ |
| SVD Right Vectors | $V_R^T$ | $(R, K \cdot d_{in})$ | $(8, 2688)$ |
| Task Slices | $V_{R, k}^T$ | $(R, d_{in})$ | $(8, 896)$ |
| Aligned Task Update | $Z_k = \Sigma_R V_{R, k}^T$ | $(R, d_{in})$ | $(8, 896)$ |
| Merged Aligned Rep | $Z_{\text{merged}} = \text{Merge}(Z_1, \dots, Z_K)$ | $(R, d_{in})$ | $(8, 896)$ |
| Merged Dense Update | $\Delta W_{\text{merged}} = U_R Z_{\text{merged}}$ | $(d_{out}, d_{in})$ | $(896, 896)$ |
| Final LoRA B | $B_{\text{merged}} = U_R$ | $(d_{out}, R)$ | $(896, 8)$ |
| Final LoRA A | $A_{\text{merged}} = Z_{\text{merged}}$ | $(R, d_{in})$ | $(8, 896)$ |

---

## 4. Key Verification Findings

1. **Exact Equivalence:** The product $B_{\text{merged}} A_{\text{merged}} = U_R Z_{\text{merged}} = \Delta W_{\text{merged}}$ is mathematically identical to the dense reconstruction.
2. **Rank Invariant:** Rank of $\Delta W_{\text{merged}}$ is strictly bounded by $R$, preserving the exact parameter budget of a rank-$R$ LoRA adapter.
3. **No SVD on Individual LoRAs First:** KnOTS does not compress adapters individually before concatenation; it concatenates the full task updates to find a *jointly shared* left basis $U_R$.
4. **Distinction from Activation-Aware KnOTS:**
   In Activation-Aware KnOTS, updates are multiplied by $C_\lambda^{1/2}$ *before* concatenation, and the final $A_{\text{merged}}$ is multiplied by $C_\lambda^{-1/2}$ *after* merging. The inner SVD alignment and partitioning remain structurally identical to original KnOTS, guaranteeing full baseline compatibility.
