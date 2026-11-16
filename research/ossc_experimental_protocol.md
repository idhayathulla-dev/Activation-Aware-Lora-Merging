# Experimental Protocol: Rigorous Benchmarking & Ablation Suite for OSSC

## 1. Objective and Hypothesis Testing

This protocol defines the experimental framework designed to rigorously evaluate the central claim of Orthogonal Subspace Sign Consensus (OSSC-Merging):
> *"Does applying sign consensus in a shared low-rank orthogonal representation reduce cross-task interference and outperform both raw-space sign consensus (EMR/TIES) and standard subspace alignment (KnOTS) under strictly equal parameter and rank budgets?"*

---

## 2. Experimental Setup

### 2.1 Base Model and Tasks
- **Base Model:** `Qwen/Qwen2.5-0.5B-Instruct` (Revision: `main`, 24 layers, $d_{in} = 896$, intermediate size $4864$).
- **Target Modules:** `["q_proj", "v_proj"]`.
- **Tasks & Datasets:**
  1. **Task A:** BoolQ (`google/boolq`) — Reading comprehension, Boolean QA.
  2. **Task B:** GSM8K (`openai/gsm8k`, `main`) — Multi-step mathematical reasoning.
  3. **Task C:** ARC-Easy (`allenai/ai2_arc`, `ARC-Easy`) — Grade-school scientific reasoning.

### 2.2 Strict Data Split Partitioning (Zero Leakage)
- **Training Split:** Used exclusively to train individual task LoRA adapters ($r=8, \alpha=16$).
- **Calibration Split:** 128 held-out examples per task.
- **Validation Split:** 100 held-out examples per task.
- **Final Evaluation Split:** 200 held-out examples per task. Under no circumstances may evaluation examples be used for hyperparameter selection or basis estimation.

---

## 3. Mandatory Baselines & Contribution Isolation

To scientifically isolate the source of any observed performance difference, OSSC must be evaluated against the following 8 systems:

| Baseline Index | System Name | Representation Space | Merging Operator | Purpose of Comparison |
|---|---|---|---|---|
| **B1** | Individual LoRA Experts | Task-specific ($r=8$) | None | Upper bound reference on single-task capability |
| **B2** | Task Arithmetic | Raw weight space | Linear addition | Measures unaligned baseline degradation |
| **B3** | TIES-Merging | Raw weight space | Trim + Elect Sign + Disjoint Merge | Measures raw parameter sign consensus |
| **B4** | EMR-Merging | Raw weight space | Elect + Mask + Rescale | Evaluates raw EMR without orthogonal projection |
| **B5** | KnOTS + Task Arithmetic | SVD space ($U_R$) | Linear addition | Isolates effect of SVD alignment alone |
| **B6** | KnOTS + EMR (OSSC Form. 1) | SVD space ($U_R$) | Elect + Mask + Rescale | Evaluates whether EMR improves over TA in SVD space |
| **B7** | Two-Sided SVD + EMR (OSSC Form. 2) | Core tensor ($U_R^T \Delta W V_R$) | Elect + Mask + Rescale | Evaluates two-sided bilinear core bottleneck |
| **B8** | Random Orthogonal Basis + EMR | Random Haar orthogonal ($U_{\text{rand}}$) | Elect + Mask + Rescale | **Negative Control:** Tests whether gains stem from SVD geometry or arbitrary rotation |

### Key Experimental Isolations:
- **Comparison B4 vs. B6 (Raw EMR vs. OSSC Form. 1):** Isolates the exact effect of projecting into an SVD basis before applying EMR.
- **Comparison B5 vs. B6 (KnOTS+TA vs. KnOTS+EMR):** Isolates whether sign consensus provides any benefit over linear averaging once updates are aligned by SVD.
- **Comparison B6 vs. B8 (SVD Basis vs. Random Basis):** Disproves whether any observed effect is an artifact of random coordinate mixing.

---

## 4. Mandatory Ablation Studies

### Ablation A: Shared Basis vs. Independent Task Bases
- Compare SVD of $W_{\text{cat}}$ against independent per-task SVDs: $\Delta W_i \approx U_i \Sigma_i V_i^T$.
- Hypothesizes whether a globally shared basis is necessary or if per-task orthogonal bases suffice.

### Ablation B: SVD Basis vs. Random Orthogonal Basis
- Replace $U_R$ with a random orthogonal matrix $Q \in \mathbb{R}^{d_{out} \times R}$ sampled from the Haar measure (via QR decomposition of Gaussian noise).
- Tests whether the spectral properties of the weights matter, or if coordinate spreading alone impacts EMR.

### Ablation C: Ordinary EMR vs. EMR in Shared Coordinates
- Direct comparison between raw-parameter EMR (B4) and SVD-coordinate EMR (B6) across increasing numbers of tasks ($K \in \{2, 3\}$).

### Ablation D: One-Sided vs. Two-Sided Projection
- Directly compare Formulation 1 ($U_R \Sigma_R V_R^T$, dimension $R \times d_{in}$) against Formulation 2 ($U_R C_i V_R^T$, dimension $R \times R$).
- Evaluates capacity loss in the $R \times R$ core tensor.

### Ablation E: Shared-Subspace Rank Sweep
- Evaluate across rank budgets: $R \in \{4, 8, 16, 32\}$.
- Evaluates whether OSSC performance collapses at low ranks due to tail singular value truncation.

### Ablation F: Projection Residual vs. Downstream Accuracy
- Measure the correlation between the per-task subspace residual $\mathcal{R}_i = \|\Delta W_i - U_R U_R^T \Delta W_i\|_F^2$ and the downstream task accuracy drop $\Delta Acc_i$.
- Tests the hypothesis that SVD truncation disproportionately harms outlier tasks.

### Ablation G: Sensitivity to Singular-Value Degeneracy and Basis Perturbations
- Inject small numerical perturbations ($\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ with $\sigma \in \{10^{-5}, 10^{-4}, 10^{-3}\}$) into $W_{\text{cat}}$ before SVD.
- Measure the stability of the elected signs ($S_{\text{elected}}$) and downstream accuracy.
- Specifically tests the Davis-Kahan spectral gap instability.

### Ablation H: Sensitivity to Adapter Scaling and Task Weights
- Sweep PEFT scaling $\gamma_i \in \{0.5, 1.0, 2.0\}$ and task weighting $w_i \in \{0.2, 0.5, 0.8\}$.
- Evaluates whether imbalanced adapter norms cause one task to dominate SVD singular vectors.

### Ablation I: KnOTS+EMR vs. KnOTS+TIES Under Exact Parameter Parity
- Direct head-to-head comparison of EMR and TIES inside KnOTS at identical rank $R=8$.
- Tests whether EMR offers any empirical advantage over the verified KnOTS+TIES baseline.

---

## 5. Evaluation Metrics

1. **Task Accuracy:**
   - BoolQ: Accuracy ($\%$)
   - GSM8K: Exact Match (EM $\%$) on normalized final numerical answer
   - ARC-Easy: Accuracy ($\%$)
2. **Aggregated Multi-Task Metrics:**
   - Macro Average Accuracy: $\bar{A} = \frac{1}{3}(A_{\text{BoolQ}} + A_{\text{GSM8K}} + A_{\text{ARC}})$
   - Worst-Task Performance: $A_{\min} = \min(A_{\text{BoolQ}} + A_{\text{GSM8K}} + A_{\text{ARC}})$
   - Task Degradation / Forgetting: $\Delta_i = A_i(\text{Merged}) - A_i(\text{Individual Expert } i)$
3. **Spectral and Geometric Metrics:**
   - Subspace Reconstruction Residual: $\mathcal{R}_{\text{total}} = \frac{\|W_{\text{cat}} - U_R \Sigma_R V_R^T\|_F^2}{\|W_{\text{cat}}\|_F^2}$
   - Retained Singular Energy: $\mathcal{E}_R = \frac{\sum_{j=1}^R \sigma_j^2}{\sum_{j=1}^{d_{\min}} \sigma_j^2}$
   - Sign Consensus Stability Rate: Fraction of elected signs invariant under $\pm 10^{-4}$ perturbation of weights.
4. **Computational Metrics:**
   - Wall-clock runtime for alignment and merging (ms).
   - Peak RAM / VRAM allocation during SVD and reconstruction (MB).
