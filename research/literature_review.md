# Literature Review and Research Gap Assessment

## 1. Executive Summary

Model merging has emerged as a compelling paradigm for synthesizing multiple task-specialized neural networks into a single multi-task system without retraining from scratch. While early techniques—such as linear weight averaging, Task Arithmetic (Ilharco et al., 2023), and TIES-Merging (Yadav et al., 2023)—were primarily conceived for fully fine-tuned (FFT) dense checkpoints, modern parameter-efficient fine-tuning (PEFT), particularly Low-Rank Adaptation (LoRA; Hu et al., 2022), introduces unique geometric challenges.

Unlike full parameter fine-tuning, where shared pre-trained initialization can induce high subspace alignment across tasks, independently trained LoRA updates $\Delta W_k = B_k A_k$ frequently occupy misaligned subspaces across tasks. Recent efforts, such as KnOTS (Stoica et al., ICLR 2025), CtM (He et al., ICML 2026), and representation geometry studies (Findings of ACL 2026), seek to explicitly analyze and align LoRA subspace representations before merging. However, existing LoRA merging techniques operate strictly in **weight space**, assigning uniform geometric importance to all input dimensions. In contrast, neural network activations exhibit highly anisotropic, non-uniform distributions across hidden dimensions—a foundational observation that propelled Activation-aware Weight Quantization (AWQ; Lin et al., 2024), ASVD (Yuan et al., 2023), and SVD-LLM (Wang et al., 2025).

This research project investigates **Activation-Distribution-Aware LoRA Merging**: transforming task-specific parameter updates $\Delta W_k$ by an empirical activation covariance square root prior to subspace alignment and merging, investigating whether this preserves task-specific behavior better than conventional weight-space merging under equal rank and parameter constraints.

---

## 2. In-Depth Review of Related Works

### 2.1 KnOTS: Knowledge Orientation Through SVD (ICLR 2025)
- **Full Title:** *Model Merging with SVD to tie the KnOTS*
- **Authors:** George Stoica, Pratik Ramesh, Boglarka Ecsedi, Leshem Choshen, Judy Hoffman
- **Venue & Year:** International Conference on Learning Representations (ICLR), 2025 (`arXiv:2410.19735`, [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/0d4f8a5109c5083b5307fcd0bddae7a7-Paper-Conference.pdf))
- **Official Implementation:** [https://github.com/gstoica27/KnOTS](https://github.com/gstoica27/KnOTS)
- **Core Mechanism:** KnOTS identifies that LoRA adapters exhibit severe parameter misalignment compared to full fine-tuning. For each layer $j$, KnOTS concatenates the task updates horizontally $[\Delta W_1, \dots, \Delta W_K] \in \mathbb{R}^{d_{out} \times (K \cdot d_{in})}$ and computes an SVD:
  $$\text{SVD}([\Delta W_1, \dots, \Delta W_K]) = U \Sigma V^T$$
  The left singular vectors $U \in \mathbb{R}^{d_{out} \times R}$ capture a shared orthonormal basis. Task-specific components in the aligned space are given by blocks of $\Sigma_R V_R^T$ (or partitioned right singular vectors $V_{R, k}^T$). Standard merging operators (such as Task Arithmetic or TIES) are applied to the aligned task components, and the final update is reconstructed as $\Delta W_{merged} = U_R \Sigma_R V_{merged}^T$.
- **Assumptions & Purpose:** Designed for LoRA adapter alignment prior to merging. Data-free and gradient-free.
- **Limitations:** Operates entirely in raw parameter space. It assumes an isotropic Euclidean geometry on the input space ($L_2$ error on weights), ignoring that different input directions have vastly differing activation variances during actual inference.

### 2.2 Representation Geometry of LoRA Model Merging (Findings of ACL 2026)
- **Full Title:** *On the Representation Geometry of LoRA Model Merging*
- **Authors:** ACL 2026 Authors
- **Venue & Year:** Findings of the Association for Computational Linguistics (ACL), 2026 ([ACL Anthology](https://aclanthology.org/2026.findings-acl.261.pdf))
- **Core Mechanism & Insights:**
  - Investigates the representation geometry of LoRA models and demonstrates a fundamental trade-off between SVD-based alignment and CUR-based matrix decomposition:
    - **SVD-based alignment** (like KnOTS) excels at isolating **globally shared components and principal directions** common across all tasks, but tends to dilute or wash out localized, task-specific features.
    - **CUR decomposition** (which reconstructs updates using actual column and row subsets of the original parameter matrices) preserves **localized, task-specific updates** with high fidelity, but lacks joint multi-task coordinate alignment.
  - Proposes a hybrid approach balancing global consensus (via SVD) with localized individuality (via CUR).
- **Relation to Activation-Aware Merging:**
  - The ACL 2026 Findings paper demonstrates that matrix decomposition choice fundamentally dictates what information is preserved during merging.
  - However, both SVD and CUR in that work are computed exclusively under standard parameter-space metrics. Transforming the coordinate space by activation covariance $C_\lambda^{1/2}$ alters the geometry of the input features themselves, modifying the principal directions captured by SVD or the selection probabilities in CUR.
- **Reassessment of Novelty Risk:**
  - Activation weighting is an established concept in compression (ASVD, SVD-LLM, AWQ). Applying it to LoRA merging is a natural empirical hypothesis, not an unprecedented theoretical breakthrough.
  - Furthermore, given the known geometric trade-off between consensus (SVD) and individuality (CUR), activation-weighted SVD could potentially amplify or compress certain task-specific features unevenly. Therefore, we treat activation-aware LoRA merging strictly as an empirical hypothesis to be critically tested, not as a guaranteed improvement.

### 2.3 CtM: Compress-then-Merge (ICML 2026)
- **Full Title:** *Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter*
- **Authors:** Zhengbao He, Ruiqi Ding, Zhehao Huang, Ruikai Yang, Tao Li, Xiaolin Huang
- **Venue & Year:** International Conference on Machine Learning (ICML), 2026 ([PMLR Proceedings v306](https://proceedings.mlr.press/v306/he26h.html), `arXiv:2606.03723`)
- **Official Implementation:** [https://github.com/sjtu-mSingle/CtM](https://github.com/sjtu-mSingle/CtM)
- **Core Mechanism:** Traditional "Merge-then-Compress" (MtC) combines full adapters and subsequently compresses via truncated SVD, which disrupts low-rank structure. CtM reverses this: it first finds shared $r$-dimensional subspaces across adapters using only LoRA factor weights, projects adapters into shared $r \times r$ coordinate spaces, and performs merging in this low-dimensional bottleneck.
- **Assumptions & Purpose:** Designed for rank-constrained multi-LoRA compression and merging.
- **Limitations:** Relies strictly on weight Frobenius norm; does not incorporate input activation statistics.

### 2.4 TIES-Merging: TrIm, Elect Sign & Merge (NeurIPS 2023)
- **Full Title:** *Resolving Interference When Merging Models*
- **Authors:** Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, Mohit Bansal
- **Venue & Year:** Advances in Neural Information Processing Systems (NeurIPS), 2023 (`arXiv:2306.01708`)
- **Official Implementation:** [https://github.com/prateeky2806/ties-merging](https://github.com/prateeky2806/ties-merging)
- **Core Mechanism:** Addresses two destructive interference issues: (1) parameter redundancy, and (2) sign disagreements across tasks. Steps:
  1. **Trim:** Retains only top-$k\%$ highest magnitude parameter changes per task.
  2. **Elect Sign:** Computes aggregate majority sign direction ($\text{sgn}(\sum_i \tau_i)$).
  3. **Disjoint Merge:** Retains and averages only parameter values that agree with the elected sign direction.
- **Limitations:** Coordinate-wise magnitude is used as a proxy for parameter importance, which can fail if small-magnitude weights align with high-variance activation channels.

### 2.5 Task Arithmetic (ICLR 2023)
- **Full Title:** *Editing Models with Task Arithmetic*
- **Authors:** Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi
- **Venue & Year:** International Conference on Learning Representations (ICLR), 2023 (`arXiv:2212.04089`)
- **Official Implementation:** [https://github.com/mlfoundations/task_vectors](https://github.com/mlfoundations/task_vectors)
- **Core Mechanism:** Computes task vectors $\tau_k = \theta_k - \theta_0$ and performs linear combination $\theta_{merged} = \theta_0 + \sum_k \lambda_k \tau_k$.
- **Limitations:** Subject to severe cross-task interference, cancellation, and subspace drift.

### 2.6 ASVD: Activation-aware Singular Value Decomposition (2023)
- **Full Title:** *ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models*
- **Authors:** Zhihang Yuan, Yuzhang Shang, Yue Song, Qiang Wu, Yan Yan, Guangyu Sun
- **Venue & Year:** arXiv preprint, December 2023 (`arXiv:2312.05825`)
- **Core Mechanism:** Scales weight matrix by an activation-derived diagonal scaling matrix $S$ ($S_{jj} = \|X_{:, j}\|_2$). Computes SVD on $W S \approx U \Sigma V^T$, and recovers $W \approx U (\Sigma V^T S^{-1})$.
- **Limitations:** Designed for single-model post-training compression, not multi-task LoRA alignment or merging. Diagonal scaling only (ignores off-diagonal correlations).

### 2.7 SVD-LLM: Truncation-aware SVD for LLM Compression (ICLR 2025)
- **Full Title:** *SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression*
- **Authors:** Xin Wang, Yu Zheng, Zhongwei Wan, Mi Zhang
- **Venue & Year:** International Conference on Learning Representations (ICLR), 2025 (`arXiv:2403.07378`)
- **Core Mechanism:** Incorporates input activation covariance $H = X^T X$ into weight low-rank approximation by solving $\min_{U, V} \|(W - U V^T) X\|_F^2$. Uses Cholesky decomposition of $H = L L^T$ to transform weights $W L$, performs SVD, and applies data whitening.
- **Limitations:** Focuses on single-model full-weight post-training pruning/compression; does not address multi-adapter alignment, cross-task interference, or multi-LoRA merging.

### 2.8 AWQ: Activation-aware Weight Quantization (MLSys 2024)
- **Full Title:** *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration*
- **Authors:** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Chen, Wei-Chen Wang, Wei-Ming Chen, Jiwoo Choi, Song Han
- **Venue & Year:** Annual Conference on Machine Learning and Systems (MLSys), 2024 (`arXiv:2306.00978`)
- **Core Mechanism:** Identifies salient weight channels via average activation magnitude $s_X = \mathbb{E}[|X|]$ and protects them during quantization via per-channel scaling.
- **Limitations:** Quantization method, not a subspace alignment or multi-model merging algorithm.

### 2.9 Scale-QLoRA (2025)
- **Full Title:** *Scale-QLoRA: Code-Invariant Adapter Merging for Native 4-bit Microscaling LLMs*
- **Authors:** arXiv preprint, February 2025 (`arXiv:2502.16489`)
- **Core Mechanism:** Focuses on merging LoRA adapters when the base model is quantized with microscaling (MX) or block-scaled formats, ensuring scale factors are harmonized before adapter addition.

---

## 3. Comparative Taxonomy Table

| Method | Main Purpose | Uses Activations? | Uses SVD? | Multi-Adapter Merging? | Primary Limitations / Assumptions |
|---|---|---|---|---|---|
| **Task Arithmetic** (Ilharco et al., 2023) | Multi-task model editing | No | No | Yes | Naive linear addition; severe interference and parameter cancellation. |
| **TIES-Merging** (Yadav et al., 2023) | Interference-resolved merging | No | No | Yes | Trimming/sign election based strictly on weight magnitude; ignores activation energy. |
| **KnOTS** (Stoica et al., ICLR 2025) | LoRA subspace alignment & merging | No | Yes (joint on $[\Delta W_k]$) | Yes | Operates purely in weight space; assumes Euclidean isotropic metric on input features. |
| **ACL 2026 Findings** | LoRA geometry analysis & merge | No | Yes (SVD + CUR) | Yes | Analyzes SVD vs. CUR representation trade-off; strictly weight-space. |
| **CtM** (He et al., ICML 2026) | Rank-constrained LoRA merging | No | Yes (LoRA factor projection) | Yes | Subspace projection based solely on parameter weights without activation weighting. |
| **ASVD** (Yuan et al., 2023) | LLM weight compression | Yes (diagonal channel norms) | Yes (on $W \cdot \text{diag}(s)$) | No (Single model) | Only scales diagonal channels; does not address multi-task merging or full covariance. |
| **SVD-LLM** (Wang et al., ICLR 2025) | LLM weight compression | Yes (full activation second moment) | Yes (on $W L$) | No (Single model) | Designed for single-model full parameter compression, not LoRA adapter alignment or merging. |
| **AWQ** (Lin et al., MLSys 2024) | LLM weight quantization | Yes (activation magnitude) | No | No (Single model) | Quantization framework; does not perform low-rank subspace alignment or multi-task merging. |
| **Proposed: Activation-Aware KnOTS** | Multi-LoRA alignment & merging | **Yes (task-balanced second moment)** | **Yes (on transformed updates)** | **Yes** | Requires activation calibration data; matrix square roots must be numerically well-conditioned. |

---

## 4. Updated Research Gap Assessment & Scientific Hypothesis

### 4.1 Research Hypothesis
We formulate the central research question not as an assertion of novelty or superiority, but as an empirical hypothesis:

> **Research Hypothesis:**
> *"Whether task-balanced activation-weighted shared-subspace alignment improves multi-adapter LoRA merging compared with conventional weight-space merging under controlled rank and parameter budgets."*

### 4.2 Existing Methods That Must Be Compared
To rigorously evaluate this hypothesis, the proposed method must be compared directly against:
1. **Base Model (Zero-Shot)**: Lower-bound baseline without adaptation.
2. **Individual Adapters (Oracle / Reference)**: Upper-bound single-task performance for each adapter independently.
3. **Task Arithmetic (Ilharco et al., 2023)**: Weight-space uniform linear combination.
4. **TIES-Merging (Yadav et al., 2023)**: Weight-space parameter trimming, sign election, and disjoint merging.
5. **KnOTS + Task Arithmetic (Stoica et al., ICLR 2025)**: Weight-space joint SVD alignment with linear averaging.
6. **KnOTS + TIES (Stoica et al., ICLR 2025)**: Weight-space joint SVD alignment with TIES merging.
7. **CtM (He et al., ICML 2026)**: If compute and code feasibility permit within our environment budget.

### 4.3 Precise Unresolved Questions & Evidence Needed
To establish a genuine empirical contribution, the research must answer:
1. **Functional Fidelity vs. Weight Proximity:** Does minimizing the regularized activation-weighted objective $\|(\Delta W - \widehat{\Delta W}) C_\lambda^{1/2}\|_F^2$ translate into higher downstream task accuracy on BoolQ, GSM8K, and ARC-Easy compared to minimizing weight Frobenius norm $\|\Delta W - \widehat{\Delta W}\|_F^2$?
2. **Equal-Budget Parameter Parity:** When both KnOTS and Activation-Aware KnOTS are constrained to identical LoRA ranks ($R \in \{4, 8, 16\}$), does activation transformation yield statistically significant performance retention, or does it introduce distortion due to calibration sample shift?
3. **Cross-Task Interference vs. Activation Weighting:** How does activation weighting interact with sign conflicts (TIES)? Does whitening the coordinates resolve or exacerbate directional conflicts across heterogeneous tasks?
4. **Ablation Evidence:** Evidence must be provided across multiple regularization levels ($\alpha_{\text{reg}} \in \{10^{-4}, 10^{-3}, 10^{-2}\}$) and calibration sample sizes (e.g., 32, 64, 128 samples) to determine whether any observed benefit is robust or fragile.
