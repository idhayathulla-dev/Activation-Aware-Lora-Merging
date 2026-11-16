# Mathematical Audit of Orthogonal Subspace Sign Consensus (OSSC-Merging)

## 1. Executive Summary & Formulation Definitions

Orthogonal Subspace Sign Consensus (OSSC-Merging) is proposed as a method to mitigate parameter interference in multi-task LoRA merging by projecting task updates into a shared orthogonal subspace before applying sign-consensus merging (such as EMR or TIES).

This audit rigorously evaluates two formulations of OSSC:
- **Formulation 1 (One-Sided Concatenation SVD):** Concatenating task updates horizontally, extracting a shared left output basis via SVD, projecting each task's right coordinates, applying sign consensus (EMR) in coordinate space, and reconstructing.
- **Formulation 2 (Two-Sided Orthogonal Projection):** Extracting both shared left ($U_R$) and right ($V_R$) orthogonal bases, projecting updates into a core tensor $C_i = U_R^T \Delta W_i V_R$, applying sign consensus in core tensor space, and reconstructing via two-sided expansion.

---

## 2. Independent Mathematical Derivations

Let $K$ be the number of task adapters to be merged.
For a target linear layer with output dimension $d_{out}$ and input dimension $d_{in}$:
Each task adapter $i \in \{1, \dots, K\}$ provides low-rank factor matrices:
$$\Delta W_i = \gamma_i B_i A_i \in \mathbb{R}^{d_{out} \times d_{in}}$$
where $B_i \in \mathbb{R}^{d_{out} \times r_i}$, $A_i \in \mathbb{R}^{r_i \times d_{in}}$, and $\gamma_i = \frac{\alpha_i}{r_i}$ is the PEFT scaling constant.

### 2.1 Formulation 1: One-Sided Horizontal Concatenation SVD

#### Step 1: Horizontal Matrix Stacking
Construct the horizontally concatenated matrix of all $K$ task updates:
$$W_{\text{cat}} = \left[ \Delta W_1, \; \Delta W_2, \; \dots, \; \Delta W_K \right] \in \mathbb{R}^{d_{out} \times (K \cdot d_{in})}$$

#### Step 2: Joint Truncated SVD
Compute the thin Singular Value Decomposition of $W_{\text{cat}}$:
$$W_{\text{cat}} = U \Sigma V^T$$
where:
- $U \in \mathbb{R}^{d_{out} \times d_{\min}}$ contains orthonormal left singular vectors ($U^T U = I$).
- $\Sigma = \text{diag}(\sigma_1, \dots, \sigma_{d_{\min}})$ contains ordered singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$, with $d_{\min} = \min(d_{out}, K \cdot d_{in})$.
- $V \in \mathbb{R}^{(K \cdot d_{in}) \times d_{\min}}$ contains orthonormal right singular vectors ($V^T V = I$).

Truncate to target rank budget $R \le d_{\min}$:
$$U_R = U[:, :R] \in \mathbb{R}^{d_{out} \times R}$$
$$\Sigma_R = \text{diag}(\sigma_1, \dots, \sigma_R) \in \mathbb{R}^{R \times R}$$
$$V_R^T = V^T[:R, :] \in \mathbb{R}^{R \times (K \cdot d_{in})}$$

#### Step 3: Right Singular Vector Partitioning
Partition $V_R^T$ horizontally into $K$ task-specific contiguous blocks:
$$V_R^T = \left[ V_{R, 1}^T, \; V_{R, 2}^T, \; \dots, \; V_{R, K}^T \right]$$
where each block $V_{R, i}^T \in \mathbb{R}^{R \times d_{in}}$ represents task $i$'s right singular slice.

#### Step 4: Task-Specific Aligned Coordinate Representation
Define the aligned coordinate matrix for task $i$:
$$Z_i = \Sigma_R V_{R, i}^T \in \mathbb{R}^{R \times d_{in}}$$
Notice that the rank-$R$ projection of task $i$'s individual update in this shared basis is:
$$\widehat{\Delta W}_i = U_R Z_i = U_R \Sigma_R V_{R, i}^T \in \mathbb{R}^{d_{out} \times d_{in}}$$

#### Step 5: EMR Sign Consensus in Aligned Space
Apply the Elect, Mask & Rescale (EMR) operator across the $K$ aligned representations $\{Z_1, \dots, Z_K\}$:
$$Z_{\text{merged}} = \text{EMR}(Z_1, \dots, Z_K) \in \mathbb{R}^{R \times d_{in}}$$
where EMR performs:
1. **Elect:** Selects backbone parameters based on maximum coordinate magnitude:
   $$Z_{\text{backbone}, r, c} = \arg\max_{Z_{i, r, c}} |Z_{i, r, c}|$$
2. **Mask:** Computes coordinate-wise consensus direction:
   $$S_{r, c} = \text{sgn}\left(\sum_{i=1}^K Z_{i, r, c}\right)$$
   and masks out task coordinates where $\text{sgn}(Z_{i, r, c}) \ne S_{r, c}$.
3. **Rescale:** Multiplies by task-specific scaling factors $\beta$ to align coordinate energy.

#### Step 6: Reconstruction into Original Parameter Space & LoRA Factorization
Reconstruct the merged dense update:
$$\Delta W_{\text{merged}} = U_R Z_{\text{merged}} \in \mathbb{R}^{d_{out} \times d_{in}}$$
Because $\text{rank}(\Delta W_{\text{merged}}) \le R$, this factorizes directly into a valid PEFT LoRA adapter:
$$B_{\text{merged}} = U_R \in \mathbb{R}^{d_{out} \times R}$$
$$A_{\text{merged}} = \frac{1}{\gamma_{\text{merged}}} Z_{\text{merged}} \in \mathbb{R}^{R \times d_{in}}$$
such that $\Delta W_{\text{merged}} \equiv \gamma_{\text{merged}} B_{\text{merged}} A_{\text{merged}}$ identically.

---

### 2.2 Formulation 2: Two-Sided Orthogonal Projection

In Formulation 2, both the input and output dimensions are independently projected onto orthogonal bases.

#### Step 1: Extraction of Left and Right Shared Subspaces
Let $U_R \in \mathbb{R}^{d_{out} \times R}$ and $V_R \in \mathbb{R}^{d_{in} \times R}$ be orthonormal bases ($U_R^T U_R = I_R$, $V_R^T V_R = I_R$).
These are obtained either by:
- SVD of the average task update $\frac{1}{K}\sum_i \Delta W_i \approx U_R \Sigma V_R^T$, or
- Concatenating all updates horizontally for $U_R$ and vertically for $V_R$.

#### Step 2: Core Tensor Coordinate Projection
Project each task update $\Delta W_i$ onto the bi-orthogonal basis:
$$C_i = U_R^T \Delta W_i V_R \in \mathbb{R}^{R \times R}$$
Notice $C_i$ is a compact $R \times R$ core matrix representing task $i$'s bilinear coordinates.

#### Step 3: Core-Space EMR Merging
Apply EMR sign consensus directly on the $R \times R$ core matrices:
$$C_{\text{merged}} = \text{EMR}(C_1, \dots, C_K) \in \mathbb{R}^{R \times R}$$

#### Step 4: Two-Sided Reconstruction
Reconstruct the dense parameter update:
$$\Delta W_{\text{merged}} = U_R C_{\text{merged}} V_R^T \in \mathbb{R}^{d_{out} \times d_{in}}$$

#### Step 5: Rank-$R$ LoRA Factorization
To convert the two-sided product $U_R C_{\text{merged}} V_R^T$ into standard LoRA format $B A$:
Compute the SVD of the small $R \times R$ core matrix:
$$C_{\text{merged}} = P \Lambda Q^T \quad \text{where } P, Q \in \mathbb{R}^{R \times R}, \Lambda \in \mathbb{R}^{R \times R}$$
Then define:
$$B_{\text{merged}} = U_R P \Lambda^{1/2} \in \mathbb{R}^{d_{out} \times R}$$
$$A_{\text{merged}} = \frac{1}{\gamma_{\text{merged}}} \Lambda^{1/2} Q^T V_R^T \in \mathbb{R}^{R \times d_{in}}$$
Then:
$$\gamma_{\text{merged}} B_{\text{merged}} A_{\text{merged}} = U_R P \Lambda Q^T V_R^T = U_R C_{\text{merged}} V_R^T = \Delta W_{\text{merged}}$$

---

## 3. Investigation of the 10 Critical Mathematical Questions

### Q1: Is the shared basis genuinely orthogonal across all tasks?
- **In Formulation 1:**
  - $U_R \in \mathbb{R}^{d_{out} \times R}$ is orthonormal: $U_R^T U_R = I_R$. The output basis is strictly orthogonal across all tasks.
  - **However, the task-specific right singular blocks $V_{R, i} \in \mathbb{R}^{d_{in} \times R}$ are NOT orthogonal!**
    The global matrix $V_R$ satisfies $V_R^T V_R = \sum_{i=1}^K V_{R, i}^T V_{R, i} = I_R$.
    Therefore, for any individual task $i$:
    $$V_{R, i}^T V_{R, i} \ne I_R \quad \text{and} \quad V_{R, i} V_{R, i}^T \ne I_{d_{in}}$$
    The individual right coordinates $V_{R, i}$ have non-orthogonal columns and norms strictly less than 1.
- **In Formulation 2:**
  - Both $U_R$ and $V_R$ are globally orthonormal ($U_R^T U_R = I_R, V_R^T V_R = I_R$). However, the projection $P_U = U_R U_R^T$ and $P_V = V_R V_R^T$ are lossy projections that discard all orthogonal components.

### Q2: Does the projection preserve task-specific information?
- **Finding: No.**
- The truncated SVD minimizes joint Frobenius reconstruction error:
  $$\min_{\text{rank}(M) \le R} \|W_{\text{cat}} - M\|_F^2 = \sum_{j=R+1}^{\min(d_{out}, K d_{in})} \sigma_j^2(W_{\text{cat}})$$
- SVD inherently favors dominant singular directions that have high variance across multiple tasks or high individual magnitude.
- If task $k$ contains specialized, unique knowledge represented by low-energy singular directions, those directions are truncated to zero during the rank-$R$ reduction, **destroying task-specific information before sign consensus is even evaluated.**

### Q3: What is the reconstruction residual for each individual adapter?
For task $i$, the residual is:
$$\mathcal{R}_i = \|\Delta W_i - \widehat{\Delta W}_i\|_F^2 = \|\Delta W_i - U_R \Sigma_R V_{R, i}^T\|_F^2$$
Summing across all tasks yields the tail singular sum:
$$\sum_{i=1}^K \mathcal{R}_i = \sum_{j=R+1}^{\min(d_{out}, K d_{in})} \sigma_j^2(W_{\text{cat}})$$
- **Disproportionate Degradation:** This residual is **not** distributed equally across tasks. Tasks whose weight updates are aligned with the principal singular vectors have $\mathcal{R}_i \approx 0$, while outlier tasks with unique geometry suffer high relative residual $\frac{\mathcal{R}_i}{\|\Delta W_i\|_F^2} \gg 0$.

### Q4: Is sign consensus invariant to SVD sign ambiguity?
- **Finding: Partially for global flips, but FATALLY NON-INVARIANT under subspace rotation.**
- **Case 1 (Sign Flip):** For any singular triplet $(u_j, \sigma_j, v_j)$, an SVD solver can arbitrarily choose $(-u_j, \sigma_j, -v_j)$.
  Because $v_j$ is flipped across all tasks simultaneously ($Z_{i, j, :} \to -Z_{i, j, :}$), the elected sign flips ($S_j \to -S_j$), and the agreement mask is preserved. When reconstructed, $(-u_j) (-Z_{\text{merged}, j}) = u_j Z_{\text{merged}, j}$. Thus, scalar sign flips cancel out.
- **Case 2 (Eigenspace Rotation - Fatal Flaw):**
  If two singular values are degenerate or nearly equal ($\sigma_j \approx \sigma_{j+1}$), the singular vectors are not uniquely defined; any orthogonal rotation $Q \in \text{SO}(2)$ generates an equally valid SVD basis:
  $$\widetilde{U} = U Q, \quad \widetilde{V} = V Q$$
  Under rotation $Q$, the aligned coordinates rotate: $\widetilde{Z}_i = Q^T Z_i$.
  **EMR, TIES, and sign consensus operate ELEMENT-WISE (coordinate-by-coordinate).**
  Element-wise operations (such as $\arg\max |Z_{i, r, c}|$ or coordinate-wise sign election) are **NOT invariant under orthogonal coordinate rotations!**
  Rotating the coordinate frame completely alters which coordinates are elected, which are trimmed, and which are masked.

### Q5: Do repeated or nearly repeated singular values create unstable bases?
- **Finding: Yes. Highly vulnerable to spectral instability.**
- By the Davis-Kahan $\sin \Theta$ theorem, the perturbation angle $\Theta$ between the empirical subspace and true subspace is bounded by:
  $$\|\sin \Theta\| \le \frac{\|E\|}{\delta}$$
  where $\delta = \min_{j \ne k} |\sigma_j - \sigma_k|$ is the spectral gap.
- In neural networks, singular value spectra of weight updates often exhibit flat, clustered tails ($\sigma_j \approx \sigma_{j+1}$).
- When $\delta \to 0$, tiny numerical perturbations, differences in float precision, or fine-tuning noise cause massive rotations of the basis vectors, leading to chaotic sign election in EMR.

### Q6: Do EMR's coordinate-wise operations remain meaningful after basis transformation?
- **Finding: No physical grounding.**
- In the original weight space, parameter $W_{j, k}$ represents a direct connection between input feature $k$ and output neuron $j$. Sign disagreements have physical meaning: Task A inhibits while Task B excites.
- In SVD space, coordinate $Z_{i, r, c}$ is a linear combination across all output dimensions weighted by singular vector $u_r$. Singular coordinates have no natural coordinate alignment, no localized semantic identity, and no reason to prefer coordinate axes over rotated axes.
- Applying coordinate-wise sign consensus on arbitrary orthogonal mixtures is an unprincipled heuristic.

### Q7: Does the method introduce additional approximation error?
- **Finding: Yes. Errors are strictly additive.**
- For each task $i$:
  $$\|\Delta W_i - \Delta W_{\text{merged}}\|_F^2 = \underbrace{\|\Delta W_i - U_R Z_i\|_F^2}_{\text{Subspace Truncation Error}} + \underbrace{\|U_R (Z_i - Z_{\text{merged}})\|_F^2}_{\text{Sign Consensus / Merging Error}}$$
- Because $(I - U_R U_R^T) \Delta W_i$ is orthogonal to the range of $U_R$, the cross-term vanishes identically:
  $$\|\Delta W_i - \Delta W_{\text{merged}}\|_F^2 = \|\Delta W_i - \widehat{\Delta W}_i\|_F^2 + \|Z_i - Z_{\text{merged}}\|_F^2$$
- Truncating to rank $R$ adds unavoidable reconstruction loss that compounds on top of the cross-task interference error.

### Q8: Does the method have a valid low-rank LoRA factorization?
- **Finding: Yes, for both formulations.**
- In Formulation 1: $B_{\text{merged}} = U_R \in \mathbb{R}^{d_{out} \times R}$, $A_{\text{merged}} = \frac{1}{\gamma} Z_{\text{merged}} \in \mathbb{R}^{R \times d_{in}}$.
- In Formulation 2: SVD of the core tensor $C_{\text{merged}} = P \Lambda Q^T$ yields $B_{\text{merged}} = U_R P \Lambda^{1/2} \in \mathbb{R}^{d_{out} \times R}$ and $A_{\text{merged}} = \frac{1}{\gamma} \Lambda^{1/2} Q^T V_R^T \in \mathbb{R}^{R \times d_{in}}$.
- Both produce standard PEFT-compatible rank-$R$ LoRA modules.

### Q9: Does the proposed method degenerate into an existing algorithm under special cases?
- **CRITICAL AUDIT FINDING: OSSC Formulation 1 is literally KnOTS + EMR.**
- In the official KnOTS framework (Stoica et al., ICLR 2025):
  1. Concatenate $W_{\text{cat}} = [\Delta W_1, \dots, \Delta W_K]$.
  2. Compute SVD: $W_{\text{cat}} \approx U_R \Sigma_R V_R^T$.
  3. Partition $V_R^T$ into $V_{R, i}^T$ and form aligned representations $Z_i = \Sigma_R V_{R, i}^T$.
  4. Apply a model-merging function $M(Z_1, \dots, Z_K)$ (KnOTS explicitly parameterizes this with Task Arithmetic, TIES, DARE, etc.).
  5. Reconstruct $\Delta W = U_R M(Z_1, \dots, Z_K)$.
- OSSC Formulation 1 executes steps 1–5 identically, simply specifying $M = \text{EMR}$.
- Therefore, OSSC Formulation 1 is **not a novel algorithm**; it is an instance of the existing KnOTS framework with an EMR merging operator.

### Q10: Can the method be computed directly from LoRA factors without constructing large dense matrices?
- **Finding: Yes.**
- Notice that:
  $$W_{\text{cat}} W_{\text{cat}}^T = \sum_{i=1}^K \Delta W_i \Delta W_i^T = \sum_{i=1}^K \gamma_i^2 B_i (A_i A_i^T) B_i^T \in \mathbb{R}^{d_{out} \times d_{out}}$$
- For rank-$r$ LoRA, $A_i A_i^T \in \mathbb{R}^{r \times r}$ is a tiny $r \times r$ matrix.
- The Gram matrix $M = W_{\text{cat}} W_{\text{cat}}^T$ can be formed in $\mathcal{O}(d_{out} \cdot K r^2)$ operations without ever materializing dense matrices of size $(d_{out}, K d_{in})$.
- Eigendecomposition of $M$ directly yields $U_R$ and $\Sigma_R^2$.
- The aligned coordinates can be computed via:
  $$Z_i = \Sigma_R V_{R, i}^T = U_R^T \Delta W_i = \gamma_i (U_R^T B_i) A_i \in \mathbb{R}^{R \times d_{in}}$$
  which only requires multiplying $(R \times r)$ by $(r \times d_{in})$.
- This reduces peak memory from $\mathcal{O}(K \cdot d_{out} d_{in})$ to $\mathcal{O}(d_{out}^2 + R d_{in})$.

---

## 4. Matrix Dimensions Table

| Symbol | Definition | Formulation 1 Dimensions | Formulation 2 Dimensions | Example ($d_{in}=896, d_{out}=896, K=3, R=8$) |
|---|---|---|---|---|
| $B_i$ | Task $i$ LoRA down-projection | $(d_{out}, r_i)$ | $(d_{out}, r_i)$ | $(896, 8)$ |
| $A_i$ | Task $i$ LoRA up-projection | $(r_i, d_{in})$ | $(r_i, d_{in})$ | $(8, 896)$ |
| $\Delta W_i$ | Dense update $\gamma_i B_i A_i$ | $(d_{out}, d_{in})$ | $(d_{out}, d_{in})$ | $(896, 896)$ |
| $W_{\text{cat}}$ | Concatenated update matrix | $(d_{out}, K \cdot d_{in})$ | N/A | $(896, 2688)$ |
| $U_R$ | Shared left singular vectors | $(d_{out}, R)$ | $(d_{out}, R)$ | $(896, 8)$ |
| $\Sigma_R$ | Truncated singular values | $(R, R)$ | N/A | $(8, 8)$ |
| $V_R^T$ | Truncated right singular vectors | $(R, K \cdot d_{in})$ | $(R, d_{in})$ | $(8, 2688)$ or $(8, 896)$ |
| $V_{R, i}^T$ | Task $i$ right slice | $(R, d_{in})$ | N/A | $(8, 896)$ |
| $Z_i$ or $C_i$ | Aligned coordinate matrix | $(R, d_{in})$ | $(R, R)$ | $(8, 896)$ or $(8, 8)$ |
| $Z_{\text{merged}}$ / $C_{\text{merged}}$ | Post-EMR merged coordinate | $(R, d_{in})$ | $(R, R)$ | $(8, 896)$ or $(8, 8)$ |
| $\Delta W_{\text{merged}}$ | Reconstructed dense update | $(d_{out}, d_{in})$ | $(d_{out}, d_{in})$ | $(896, 896)$ |
| $B_{\text{merged}}$ | Final LoRA B matrix | $(d_{out}, R)$ | $(d_{out}, R)$ | $(896, 8)$ |
| $A_{\text{merged}}$ | Final LoRA A matrix | $(R, d_{in})$ | $(R, d_{in})$ | $(8, 896)$ |
