# Computational Feasibility Analysis: OSSC-Merging

## 1. Hardware Architecture & Environment Constraints

The development testbed exhibits the following physical specifications:
- **Processor:** AMD Ryzen 5 7000-series (AMD64 Family 25 Model 68).
- **Installed System RAM:** 16 GB (with ~1.1 GB currently free unallocated RAM).
- **GPU:** AMD Radeon 660M Integrated Graphics (no NVIDIA CUDA runtime support).
- **Secondary Environment:** Google Colab Cloud GPU runtime (NVIDIA T4 / V100 with 15–16 GB VRAM).

---

## 2. Memory Bottleneck Analysis of SVD on Concatenated LoRA Updates

### 2.1 The Dense Stacking Trap (Naive Implementation)
Consider merging $K$ task adapters for `Qwen/Qwen2.5-0.5B-Instruct` linear layers:
- Layer dimensions: $d_{in} = 896$, $d_{out} = 896$.
- Number of layers: 24 (with `q_proj` and `v_proj`, total 48 target modules).
- For $K = 3$ tasks:
  $$W_{\text{cat}} \in \mathbb{R}^{896 \times (3 \cdot 896)} = \mathbb{R}^{896 \times 2688}$$
  In float64 (required for SVD numerical stability):
  $$\text{Memory per layer} = 896 \times 2688 \times 8 \text{ bytes} \approx 19.27 \text{ MB}$$
  For 48 modules, holding all $W_{\text{cat}}$ dense matrices simultaneously requires $\approx 925 \text{ MB}$. On a machine with only ~1.1 GB free RAM, materializing all dense concatenated matrices simultaneously poses an immediate Out-Of-Memory (OOM) threat.

For larger models (e.g. LLaMA-3-8B with $d = 4096$ and $K = 10$):
$$W_{\text{cat}} \in \mathbb{R}^{4096 \times 40960} \implies 4096 \times 40960 \times 8 \text{ bytes} \approx 1.34 \text{ GB per layer!}$$
Naive dense SVD on CPU would instantly crash.

---

## 3. Low-Rank Gram Matrix Acceleration (Memory-Safe CPU Execution)

Because each adapter is parameterized as a low-rank product $\Delta W_i = \gamma_i B_i A_i$ with rank $r \ll \min(d_{in}, d_{out})$ (e.g., $r = 8$ while $d = 896$):

### 3.1 Mathematical Optimization
We can compute the thin left singular vectors $U_R$ and singular values $\Sigma_R$ without ever instantiating $W_{\text{cat}}$:
$$M = W_{\text{cat}} W_{\text{cat}}^T = \sum_{i=1}^K \Delta W_i \Delta W_i^T = \sum_{i=1}^K \gamma_i^2 B_i (A_i A_i^T) B_i^T \in \mathbb{R}^{d_{out} \times d_{out}}$$

1. **Inner Core Matrix:** For each task $i$, compute $G_i = A_i A_i^T \in \mathbb{R}^{r \times r}$.
   - Size: $8 \times 8 = 64$ entries ($512$ bytes).
2. **Intermediate Product:** $H_i = B_i G_i \in \mathbb{R}^{d_{out} \times r}$.
   - Size: $896 \times 8 = 7168$ entries ($57.3$ KB).
3. **Gram Accumulation:** $M = \sum_{i=1}^K \gamma_i^2 H_i B_i^T \in \mathbb{R}^{896 \times 896}$.
   - Size: $896 \times 896 \times 8 \text{ bytes} = 6.42 \text{ MB}$.
4. **Symmetric Eigendecomposition on $M$:**
   $$M = U \Lambda U^T \implies U_R = U[:, :R], \quad \Sigma_R = \Lambda[:R, :R]^{1/2}$$
5. **Right Singular Block Projection:**
   $$Z_i = \Sigma_R V_{R, i}^T = U_R^T \Delta W_i = \gamma_i (U_R^T B_i) A_i \in \mathbb{R}^{R \times d_{in}}$$
   - Computed as $(R \times r) \times (r \times d_{in})$, requiring negligible memory ($8 \times 896 \times 8 \text{ bytes} \approx 57.3 \text{ KB}$).

### 3.2 Computational Complexity Comparison

| Step | Naive Dense SVD | Low-Rank Gram Acceleration | Reduction Factor |
|---|---|---|---|
| **Peak Working RAM** | $\mathcal{O}(K \cdot d_{out} d_{in})$ | $\mathcal{O}(d_{out}^2 + R d_{in})$ | **$3\times$ to $10\times$ lower** |
| **SVD / Eigh Flops** | $\mathcal{O}(K \cdot d_{out}^2 d_{in})$ | $\mathcal{O}(d_{out}^3 + K \cdot d_{out} d_{in} r)$ | **$2\times$ faster on CPU** |
| **LoRA Factorization** | Dense matrix expansion | Direct low-rank matrix multiply | **Zero dense materialization** |

---

## 4. Execution Feasibility Decision

1. **Local CPU Feasibility (Windows 11):**
   - **FEASIBLE:** Mathematical unit tests, synthetic matrix validation, condition number checks, and single-module SVD benchmarking can easily execute locally within 200 MB of RAM using the low-rank Gram formulation.
   - **INFEASIBLE:** Fine-tuning all 3 task adapters (`Qwen/Qwen2.5-0.5B-Instruct` on BoolQ, GSM8K, ARC-Easy) locally on CPU with 1.1 GB free RAM would cause extreme paging, memory thrashing, and unacceptable training times (estimated >30 hours on CPU).
2. **Google Colab GPU Feasibility:**
   - **HIGHLY FEASIBLE:** Full model training, activation calibration, and inference evaluation fit comfortably within a standard Colab 16 GB T4/V100 GPU instance (model weights: ~1 GB in bfloat16, peak training VRAM with LoRA: ~4.2 GB).
