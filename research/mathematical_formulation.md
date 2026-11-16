# Mathematical Formulation: Activation-Distribution-Aware LoRA Merging

## 1. Linear Layer and Adapter Dynamics

Let a target pre-trained linear module have frozen base weights $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ and input activations $x \in \mathbb{R}^{d_{in}}$.
Under standard LoRA parameterization for task $k \in \{1, \dots, K\}$:
$$\Delta W_k = \frac{\alpha_k}{r_k} B_k A_k$$
where $B_k \in \mathbb{R}^{d_{out} \times r_k}$, $A_k \in \mathbb{R}^{r_k \times d_{in}}$, $r_k \ll \min(d_{in}, d_{out})$, and $\alpha_k$ is the LoRA scaling constant. In PEFT, $B_k$ is initialized to zero and $A_k$ to Gaussian noise.

The forward output of task model $k$ on input $x$ is:
$$y_k = (W_0 + \Delta W_k) x = W_0 x + \Delta W_k x$$

---

## 2. Derivation of the Regularized Activation-Weighted Objective

Let $E = \Delta W - \widehat{\Delta W} \in \mathbb{R}^{d_{out} \times d_{in}}$ denote the parameter error between the target task update $\Delta W$ and an approximation or merged candidate $\widehat{\Delta W}$.

### 2.1 Unregularized Expected Output Distortion
The expected squared $L_2$ error between the target update output $\Delta W x$ and the candidate update output $\widehat{\Delta W} x$ over the input distribution $\mathcal{D}$ is:
$$\mathcal{L}_{\text{output}}(\widehat{\Delta W}) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \|\Delta W x - \widehat{\Delta W} x\|_2^2 \right] = \mathbb{E}_{x \sim \mathcal{D}} \left[ \|E x\|_2^2 \right]$$

Expanding using the trace operator:
$$\begin{aligned}
\mathbb{E}_{x \sim \mathcal{D}} \left[ \|E x\|_2^2 \right] &= \mathbb{E}_{x \sim \mathcal{D}} \left[ \text{Tr}\left( E x x^T E^T \right) \right] \\
&= \text{Tr}\left( E \, \mathbb{E}_{x \sim \mathcal{D}}[x x^T] \, E^T \right) \\
&= \text{Tr}\left( E \, C_x \, E^T \right)
\end{aligned}$$
where $C_x = \mathbb{E}_{x \sim \mathcal{D}}[x x^T] \in \mathbb{R}^{d_{in} \times d_{in}}$ is the uncentered second moment of the layer input activations.

### 2.2 Tikhonov Regularization and the Regularized Objective
To guarantee positive definiteness and invertibility of the empirical covariance, we add Tikhonov regularization:
$$C_\lambda = C_x + \lambda I$$
where $\lambda > 0$ is a scalar regularization parameter.

Now consider transforming the parameter error $E$ by the regularized matrix square root $C_\lambda^{1/2}$:
$$\|E C_\lambda^{1/2}\|_F^2 = \text{Tr}\left( (E C_\lambda^{1/2})(E C_\lambda^{1/2})^T \right) = \text{Tr}\left( E C_\lambda E^T \right)$$

Substituting $C_\lambda = C_x + \lambda I$:
$$\begin{aligned}
\|E C_\lambda^{1/2}\|_F^2 &= \text{Tr}\left( E (C_x + \lambda I) E^T \right) \\
&= \text{Tr}\left( E C_x E^T \right) + \lambda \, \text{Tr}\left( E E^T \right) \\
&= \mathbb{E}_{x \sim \mathcal{D}}\left[ \|E x\|_2^2 \right] + \lambda \, \|E\|_F^2
\end{aligned}$$

> **Important Scientific Clarification:**
> The quantity $\|E C_\lambda^{1/2}\|_F^2$ is **not strictly the unregularized expected output reconstruction error**. Rather, it is a **regularized activation-weighted objective**:
> $$\|E C_\lambda^{1/2}\|_F^2 = \underbrace{\mathbb{E}_{x \sim \mathcal{D}}[\|(\Delta W - \widehat{\Delta W})x\|_2^2]}_{\text{Empirical Output Distortion}} + \underbrace{\lambda \|\Delta W - \widehat{\Delta W}\|_F^2}_{\text{Weight-Space Regularization Penalty}}$$
> The parameter $\lambda$ explicitly trades off functional output fidelity against weight-space Frobenius proximity. In the limit $\lambda \to 0$, it approaches pure output reconstruction; as $\lambda \to \infty$, it degenerates into standard weight-space Frobenius minimization.

---

## 3. Two Distinct Algorithms: Individual Approximation vs. Multi-Adapter Merging

It is critical to distinguish between two mathematically distinct operations:

### Algorithm A: Individual Activation-Weighted Low-Rank Approximation (Single-Adapter Compression)
For a single adapter update $\Delta W_i$:
1. Transform: $A_i = \Delta W_i C_\lambda^{1/2} \in \mathbb{R}^{d_{out} \times d_{in}}$.
2. Compute rank-$r$ truncated SVD on the individual matrix:
   $$\text{SVD}(A_i) = U_i \Sigma_i V_i^T \implies A_{i, r} = U_{i, :r} \Sigma_{i, :r} V_{i, :r}^T$$
3. Invert transformation: $\widehat{\Delta W}_i = A_{i, r} C_\lambda^{-1/2}$.
- **Mathematical Property:** $\widehat{\Delta W}_i$ is the exact minimizer of $\min_{\text{rank}(W) \le r} \|(\Delta W_i - W) C_\lambda^{1/2}\|_F^2$ by the Eckart-Young-Mirsky theorem.
- **Scope:** Single-adapter compression/pruning. It does **not** perform cross-task alignment or multi-task merging.

### Algorithm B: Activation-Aware Shared-Basis Multi-Adapter Merging (Our Proposed Method)
For $K$ independently trained task adapters $\{\Delta W_1, \dots, \Delta W_K\}$:
1. Estimate task-balanced regularized second moment $C_\lambda = C_{x, \text{balanced}} + \lambda I$.
2. Transform each adapter update: $\widetilde{\Delta W}_k = \Delta W_k C_\lambda^{1/2}$ for $k \in \{1, \dots, K\}$.
3. Concatenate horizontally: $\widetilde{W}_{\text{cat}} = [\widetilde{\Delta W}_1, \dots, \widetilde{\Delta W}_K] \in \mathbb{R}^{d_{out} \times (K \cdot d_{in})}$.
4. Compute joint SVD: $\widetilde{W}_{\text{cat}} = U \Sigma V^T$.
5. Truncate to rank budget $R \le \min(d_{out}, K \cdot d_{in})$: $U_R \in \mathbb{R}^{d_{out} \times R}$, $\Sigma_R \in \mathbb{R}^{R \times R}$, $V_R^T \in \mathbb{R}^{R \times (K \cdot d_{in})}$.
6. Partition $V_R^T$ into $K$ task-specific blocks: $V_R^T = [V_{R, 1}^T, \dots, V_{R, K}^T]$, where $V_{R, k}^T \in \mathbb{R}^{R \times d_{in}}$.
7. Merge in aligned space: $\widetilde{Z}_{\text{merged}} = \text{Merge}\left(\Sigma_R V_{R, 1}^T, \dots, \Sigma_R V_{R, K}^T\right)$ (via Task Arithmetic or TIES).
8. Invert transformation back to original coordinates:
   $$\Delta W_{\text{merged}} = U_R \widetilde{Z}_{\text{merged}} C_\lambda^{-1/2}$$
9. Factorize as exact rank-$R$ LoRA adapter:
   $$B_{\text{merged}} = U_R \in \mathbb{R}^{d_{out} \times R}, \quad A_{\text{merged}} = \widetilde{Z}_{\text{merged}} C_\lambda^{-1/2} \in \mathbb{R}^{R \times d_{in}}$$
   with scaling factor $\frac{\alpha}{R} = 1.0$.

> **The actual research investigation implements and evaluates Algorithm B.**

---

## 4. Matrix Dimensions at Every Stage

| Variable | Mathematical Expression | Shape / Dimensions | Description |
|---|---|---|---|
| $W_0$ | Base frozen weight | $(d_{out}, d_{in})$ | Pre-trained layer weight |
| $x$ | Input activation vector | $(d_{in},)$ | Forward pass hidden states |
| $B_k$ | LoRA down-projection | $(d_{out}, r_k)$ | Task $k$ LoRA B matrix |
| $A_k$ | LoRA up-projection | $(r_k, d_{in})$ | Task $k$ LoRA A matrix |
| $\Delta W_k$ | Effective task update $\frac{\alpha_k}{r_k} B_k A_k$ | $(d_{out}, d_{in})$ | Dense task update |
| $C_x$ | Uncentered second moment $\mathbb{E}[x x^T]$ | $(d_{in}, d_{in})$ | Activation second moment |
| $C_\lambda$ | Regularized covariance $C_x + \lambda I$ | $(d_{in}, d_{in})$ | Regularized covariance (SPD) |
| $Q, \Lambda$ | Eigendecomposition $C_\lambda = Q \Lambda Q^T$ | $Q: (d_{in}, d_{in}), \Lambda: (d_{in}, d_{in})$ | Orthogonal eigenvectors and eigenvalues |
| $C_\lambda^{1/2}$ | Matrix square root $Q \Lambda^{1/2} Q^T$ | $(d_{in}, d_{in})$ | Whitening / scaling transform |
| $C_\lambda^{-1/2}$ | Matrix inverse square root $Q \Lambda^{-1/2} Q^T$ | $(d_{in}, d_{in})$ | Unwhitening / recovery transform |
| $\widetilde{\Delta W}_k$ | Transformed update $\Delta W_k C_\lambda^{1/2}$ | $(d_{out}, d_{in})$ | Transformed coordinate update |
| $\widetilde{W}_{\text{cat}}$ | Concatenated transformed updates | $(d_{out}, K \cdot d_{in})$ | Horizontally stacked updates |
| $U_R$ | Truncated left singular vectors | $(d_{out}, R)$ | Shared orthonormal output basis |
| $\Sigma_R$ | Truncated singular values | $(R, R)$ | Shared singular values |
| $V_R^T$ | Truncated right singular vectors | $(R, K \cdot d_{in})$ | Joint right singular vectors |
| $V_{R, k}^T$ | Block $k$ of $V_R^T$ | $(R, d_{in})$ | Task $k$ aligned right coordinates |
| $\widetilde{Z}_k$ | Task $k$ aligned update $\Sigma_R V_{R, k}^T$ | $(R, d_{in})$ | Aligned coordinates for merging |
| $\widetilde{Z}_{\text{merged}}$ | Merged aligned representation | $(R, d_{in})$ | Result of Task Arithmetic or TIES |
| $\widetilde{\Delta W}_{\text{merged}}$ | Transformed merged update $U_R \widetilde{Z}_{\text{merged}}$ | $(d_{out}, d_{in})$ | Merged update in transformed space |
| $\Delta W_{\text{merged}}$ | Reconstructed update $\widetilde{\Delta W}_{\text{merged}} C_\lambda^{-1/2}$ | $(d_{out}, d_{in})$ | Final dense merged update |
| $B_{\text{merged}}$ | Final LoRA B matrix ($U_R$) | $(d_{out}, R)$ | Merged LoRA B parameter |
| $A_{\text{merged}}$ | Final LoRA A matrix ($\widetilde{Z}_{\text{merged}} C_\lambda^{-1/2}$) | $(R, d_{in})$ | Merged LoRA A parameter |

---

## 5. Merging Operators in Aligned Coordinates

Once task updates are mapped to $\widetilde{Z}_k = \Sigma_R V_{R, k}^T \in \mathbb{R}^{R \times d_{in}}$:

### 5.1 Task Arithmetic in Aligned Coordinates
$$\widetilde{Z}_{\text{merged}} = \sum_{k=1}^K w_k \widetilde{Z}_k$$
where $w_k = 1/K$ by default.

### 5.2 TIES Merging in Aligned Coordinates
1. **Trim:** Keep top $(1 - p)$ fraction of coordinates by absolute value per task:
   $$\widetilde{Z}_k^{\text{trim}} = \text{Trim}(\widetilde{Z}_k, p)$$
2. **Elect Sign:** Determine coordinate-wise majority sign:
   $$S_{i, j} = \text{sgn}\left(\sum_{k=1}^K \widetilde{Z}_{k, i, j}^{\text{trim}}\right)$$
3. **Disjoint Average:** Average entries that match the elected sign:
   $$\widetilde{Z}_{\text{merged}, i, j} = \frac{1}{|K_{(i, j)}^+|} \sum_{k \in K_{(i, j)}^+} \widetilde{Z}_{k, i, j}^{\text{trim}} \quad \text{where} \quad K_{(i, j)}^+ = \{k : \text{sgn}(\widetilde{Z}_{k, i, j}^{\text{trim}}) = S_{i, j}\}$$

---

## 6. Numerical Stability and Validation Guarantees

1. **Float64 Precision:** Eigendecomposition and SVD are executed in `torch.float64` (`np.float64`).
2. **Eigenvalue Clamping:** Small numerical non-positive eigenvalues are clamped to $\epsilon = 10^{-12} \cdot \max(\mu)$.
3. **Reconstruction Invariants:**
   - $\|C_\lambda^{1/2} C_\lambda^{-1/2} - I\|_F < 10^{-10}$
   - Invertibility: $\Delta W_{\text{merged}} = B_{\text{merged}} A_{\text{merged}}$ holds identically (with effective rank $R$).
