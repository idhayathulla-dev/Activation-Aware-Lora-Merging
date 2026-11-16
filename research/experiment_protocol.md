# Experiment Protocol: Activation-Distribution-Aware LoRA Merging

## 1. Experimental Design & Objectives

This protocol specifies the empirical framework to test the central research hypothesis:
> *"Whether task-balanced activation-weighted shared-subspace alignment improves multi-adapter LoRA merging compared with conventional weight-space merging under controlled rank and parameter budgets."*

---

## 2. Regularized Objective and Algorithm Clarification

### 2.1 The Regularized Objective
For error matrix $E = \Delta W - \widehat{\Delta W}$ and regularized covariance $C_\lambda = C_x + \lambda I$:
$$\|E C_\lambda^{1/2}\|_F^2 = \mathbb{E}_{x \sim \mathcal{D}}[\|E x\|_2^2] + \lambda \|E\|_F^2$$
This objective explicitly trades off empirical layer output reconstruction against weight-space Frobenius norm regularization. It is **not** identical to unregularized expected output error.

### 2.2 Investigated Algorithm
Our experimental pipeline implements and tests **Algorithm B (Activation-Aware Shared-Basis Multi-Adapter Merging)**:
1. Transforming all $K$ task updates by $C_\lambda^{1/2}$.
2. Concatenating horizontally into $\widetilde{W}_{\text{cat}}$.
3. Joint SVD alignment $\widetilde{W}_{\text{cat}} = U \Sigma V^T$.
4. Truncating to rank $R$ and partitioning $V_R^T$ into $K$ task components.
5. Performing merging (Task Arithmetic or TIES) in aligned coordinates.
6. Recovering exact rank-$R$ LoRA factors via inverse transformation $C_\lambda^{-1/2}$.

*(Note: Algorithm A—individual adapter activation-weighted SVD compression—is treated purely as a single-adapter reconstruction baseline and is not confused with multi-adapter merging).*

---

## 3. Models and Hardware Architecture

### 3.1 Base Model Specifications
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Architecture:** Qwen2ForCausalLM (hidden size $d_{in} = 896$, intermediate size $4864$, num layers $24$, num heads $14$, num key-value heads $2$).
- **Target Modules:** `["q_proj", "v_proj"]`
- **Default Precision:** `torch.bfloat16` on GPU, `torch.float32` on CPU.

### 3.2 Execution Environments
- **Local Dev / Validation:** Windows 11 CPU smoke testing, matrix transformation verification, unit tests, and pipeline integrity checks.
- **Compute Cluster / Google Colab:** Free/pro GPU runtime (NVIDIA T4 / V100 / A100) for training LoRA adapters, activation calibration, and full multi-task evaluation.

---

## 4. Benchmark Tasks & Datasets

| Adapter | Dataset Name | HF Path | Subsplit / Config | Task Description | Metric |
|---|---|---|---|---|---|
| **A** | BoolQ | `google/boolq` | default | Yes/No Reading Comprehension | Accuracy |
| **B** | GSM8K | `openai/gsm8k` | `main` | Multi-step Grade School Math | Exact Match (Normalized Number) |
| **C** | ARC-Easy | `allenai/ai2_arc` | `ARC-Easy` | Grade-school Science QA | Accuracy |

### Dataset Partitioning Rules (No Data Leakage):
1. **Training Split:** Used exclusively to train the single-task LoRA adapter.
2. **Calibration Split:** A held-out slice (128 examples per task) from validation/train data used *only* to collect layer input activations for the uncentered second-moment covariance $C_x$.
3. **Evaluation Split:** Dedicated test/validation examples used solely for post-merging benchmarking. Under no circumstances may evaluation examples be used for calibration or training.

---

## 5. Hyperparameter Specifications

### 5.1 Adapter Training:
- **Base LoRA Rank ($r$):** 8
- **LoRA Alpha ($\alpha$):** 16 (scaling factor = 2.0)
- **Dropout:** 0.05
- **Target Modules:** `q_proj`, `v_proj`
- **Max Sequence Length:** 512
- **Batch Size:** 2 (gradient accumulation = 8, effective batch size = 16)
- **Learning Rate:** $2 \times 10^{-4}$ with linear warmup and cosine decay
- **Epochs:** 1 to 3 depending on dataset size (pilot uses 1 epoch)
- **Random Seed:** 42 (seeds 43, 44 reserved for multi-seed validation)

### 5.2 Merging Configurations:
- **Rank Budgets ($R$):** 4, 8, 16
- **Regularization ($\alpha_{\text{reg}}$):** $10^{-3}$ (ablation over $10^{-4}, 10^{-3}, 10^{-2}$)
- **TIES Trim Fraction ($p$):** 0.2 (keep top 80%)

---

## 6. Evaluation Protocol and Baselines

### 6.1 Evaluated Systems:
1. **Base Model (Zero-Shot)** (no adapters)
2. **Individual Adapters** (Adapter A on all 3 tasks, B on all 3 tasks, C on all 3 tasks)
3. **Task Arithmetic** (uniform weighting $\frac{1}{3}$)
4. **TIES-Merging** (weight-space TIES)
5. **KnOTS + Task Arithmetic** (weight-space SVD alignment + linear average)
6. **KnOTS + TIES** (weight-space SVD alignment + TIES merge)
7. **Activation-Aware KnOTS + Task Arithmetic** (Proposed Method A)
8. **Activation-Aware KnOTS + TIES** (Proposed Method B)

### 6.2 Metrics:
- $Acc_{\text{BoolQ}}$, $EM_{\text{GSM8K}}$, $Acc_{\text{ARC}}$
- **Macro Average Performance:** $\bar{M} = \frac{1}{3} (Acc_{\text{BoolQ}} + EM_{\text{GSM8K}} + Acc_{\text{ARC}})$
- **Worst-Task Performance:** $\min(Acc_{\text{BoolQ}}, EM_{\text{GSM8K}}, Acc_{\text{ARC}})$
- **Task Degradation / Forgetting:** $\Delta_k = M_k(\text{Merged}) - M_k(\text{Individual Adapter } k)$
- **Layer Output Reconstruction Error:** $\| \Delta W_k x - \Delta W_{\text{merged}} x \|_2^2$ over calibration tokens
- **Effective Parameter Count:** Total non-frozen parameters deployed at test time

---

## 7. Comprehensive Matrix Dimension Table

| Step | Expression | Dimensions | Example ($d_{in}=896, d_{out}=896, K=3, R=8$) |
|---|---|---|---|
| Raw Adapters | $B_k \in \mathbb{R}^{d_{out} \times r}, A_k \in \mathbb{R}^{r \times d_{in}}$ | $B: (d_{out}, r), A: (r, d_{in})$ | $B: (896, 8), A: (8, 896)$ |
| Dense Update | $\Delta W_k = \frac{\alpha}{r} B_k A_k$ | $(d_{out}, d_{in})$ | $(896, 896)$ |
| Covariance | $C_\lambda = C_{x, \text{balanced}} + \lambda I$ | $(d_{in}, d_{in})$ | $(896, 896)$ |
| Matrix Sqrt | $C_\lambda^{1/2}, C_\lambda^{-1/2}$ | $(d_{in}, d_{in})$ | $(896, 896)$ |
| Transformed Update | $\widetilde{\Delta W}_k = \Delta W_k C_\lambda^{1/2}$ | $(d_{out}, d_{in})$ | $(896, 896)$ |
| Stacked Matrix | $\widetilde{W}_{\text{cat}} = [\widetilde{\Delta W}_1, \dots, \widetilde{\Delta W}_K]$ | $(d_{out}, K \cdot d_{in})$ | $(896, 2688)$ |
| Left Basis | $U_R$ | $(d_{out}, R)$ | $(896, 8)$ |
| Singular Values | $\Sigma_R$ | $(R, R)$ | $(8, 8)$ |
| Right Basis | $V_R^T$ | $(R, K \cdot d_{in})$ | $(8, 2688)$ |
| Task Partition | $V_{R, k}^T$ | $(R, d_{in})$ | $(8, 896)$ |
| Aligned Rep | $\widetilde{Z}_k = \Sigma_R V_{R, k}^T$ | $(R, d_{in})$ | $(8, 896)$ |
| Aligned Merge | $\widetilde{Z}_{\text{merged}}$ | $(R, d_{in})$ | $(8, 896)$ |
| Merged Dense | $\Delta W_{\text{merged}} = U_R \widetilde{Z}_{\text{merged}} C_\lambda^{-1/2}$ | $(d_{out}, d_{in})$ | $(896, 896)$ |
| Merged LoRA B | $B_{\text{merged}} = U_R$ | $(d_{out}, R)$ | $(896, 8)$ |
| Merged LoRA A | $A_{\text{merged}} = \widetilde{Z}_{\text{merged}} C_\lambda^{-1/2}$ | $(R, d_{in})$ | $(8, 896)$ |
