# Literature Review: Orthogonal Subspace & Sign-Consensus LoRA Merging

## 1. Overview

This document presents a comprehensive, verified survey of the peer-reviewed literature and recent preprints related to **Orthogonal Subspace Sign Consensus (OSSC-Merging)**, spanning model merging, parameter-efficient fine-tuning (PEFT/LoRA), spectral alignment, and sign-consensus aggregation.

---

## 2. In-Depth Analysis of Related Publications

### 2.1 EMR-Merging: Elect, Mask & Rescale-Merging
- **Full Title:** *EMR-Merging: Tuning-Free High-Performance Model Merging*
- **Authors:** Chenyu Huang, Peng Ye, Tao Chen, Tong He, Xiangyu Yue, Wanli Ouyang
- **Venue & Year:** Advances in Neural Information Processing Systems (**NeurIPS 2024**, Spotlight) (`arXiv:2405.17461`, [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024))
- **Mathematical Contribution:** Proposes a three-stage tuning-free parameter merging algorithm in raw parameter space:
  1. *Elect:* Elects parameter coordinates with maximum absolute values across task models to serve as the unified model backbone.
  2. *Mask:* Computes coordinate-wise task directional masks based on consensus direction to filter opposing gradient updates.
  3. *Rescale:* Rescales task-specific magnitude to recover individual task calibration.
- **Shared Orthogonal Basis:** No. Operates strictly in the original parameter coordinate system.
- **Sign Consensus:** Yes. Computes directional agreement across tasks in raw weight space.
- **Representation:** Dense parameters and flattened weight vectors.
- **Exact Difference from OSSC:** EMR-Merging was designed for and applied to full-parameter weight spaces without orthogonal basis decomposition. OSSC attempts to apply EMR's three steps to SVD-transformed coordinate vectors $Z_i$.

---

### 2.2 KnOTS: Model Merging with SVD to tie the KnOTS
- **Full Title:** *Model Merging with SVD to tie the KnOTS*
- **Authors:** George Stoica, Pratik Ramesh, Boglarka Ecsedi, Leshem Choshen, Judy Hoffman
- **Venue & Year:** International Conference on Learning Representations (**ICLR 2025**) (`arXiv:2410.19735`, [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/0d4f8a5109c5083b5307fcd0bddae7a7-Paper-Conference.pdf))
- **Mathematical Contribution:** Identifies that LoRA updates suffer from severe subspace misalignment across tasks. Concatenates task updates $W_{\text{cat}} = [\Delta W_1, \dots, \Delta W_K]$, computes truncated SVD $W_{\text{cat}} \approx U_R \Sigma_R V_R^T$, partitions $V_R^T$ into task blocks $V_{R, i}^T$, forms aligned task representations $Z_i = \Sigma_R V_{R, i}^T$, applies a merging function (e.g. Task Arithmetic, TIES), and reconstructs $\Delta W_{\text{merged}} = U_R Z_{\text{merged}}$.
- **Shared Orthogonal Basis:** Yes. $U_R$ is a shared orthonormal output basis.
- **Sign Consensus:** Yes, when parameterized with TIES (KnOTS+TIES).
- **Representation:** LoRA updates expanded to dense or low-rank coordinates.
- **Exact Difference from OSSC:** **OSSC Formulation 1 is structurally identical to KnOTS.** The only distinction is substituting the EMR operator in place of TIES as the inner merging function. KnOTS explicitly describes this modular decoupling.

---

### 2.3 OSRM: Orthogonal Subspaces for Robust Model Merging
- **Full Title:** *Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging*
- **Authors:** Haobo Zhang et al.
- **Venue & Year:** Annual Meeting of the Association for Computational Linguistics (**ACL 2025**) (`arXiv:2505.22934`)
- **Mathematical Contribution:** Analyzes the parameter-data distribution interplay that causes cross-task interference in LoRA. Proposes constraining task LoRA subspaces to mutually orthogonal subspaces *prior to or during fine-tuning* using sample covariance matrices, ensuring task updates are geometrically disjoint by construction.
- **Shared Orthogonal Basis:** Yes, but enforces *mutually orthogonal task subspaces*, rather than projecting all tasks into one *shared* low-rank subspace.
- **Sign Consensus:** No. Focuses on subspace separation to prevent interference.
- **Representation:** LoRA adapters.
- **Exact Difference from OSSC:** OSRM enforces mutual orthogonality across task adapters during initialization/training to eliminate cross-talk; OSSC projects independently trained adapters post-hoc into a *single shared* subspace and performs sign consensus.

---

### 2.4 CtM: Compress-then-Merge
- **Full Title:** *Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter*
- **Authors:** Zhengbao He, Ruiqi Ding, Zhehao Huang, Ruikai Yang, Tao Li, Xiaolin Huang
- **Venue & Year:** International Conference on Machine Learning (**ICML 2026**) ([PMLR Proceedings v306](https://proceedings.mlr.press/v306/he26h.html), `arXiv:2606.03723`)
- **Mathematical Contribution:** Solves the rank-explosion problem of "Merge-then-Compress" (MtC) by enforcing a rank-$r$ bottleneck *before* merging. Computes shared $r$-dimensional left and right subspaces using only LoRA factor matrices ($B_i, A_i$), projects adapters into compact $r \times r$ coordinate spaces, and performs merging in this low-dimensional core.
- **Shared Orthogonal Basis:** Yes. Left and right shared subspaces extracted from LoRA factors.
- **Sign Consensus:** Can apply linear combination or TIES in the reduced space.
- **Representation:** Strictly factorized LoRA matrices.
- **Exact Difference from OSSC:** CtM computes shared left and right projections directly from LoRA factors and executes standard merging in an $r \times r$ coordinate bottleneck. OSSC Formulation 2 uses a similar two-sided projection $U_R^T \Delta W_i V_R$ but pairs it specifically with EMR sign consensus.

---

### 2.5 Representation Geometry of LoRA Model Merging
- **Full Title:** *On the Representation Geometry of LoRA Model Merging*
- **Authors:** ACL 2026 Authors
- **Venue & Year:** Findings of the Association for Computational Linguistics (**ACL 2026 Findings**) ([Paper Link](https://aclanthology.org/2026.findings-acl.261.pdf))
- **Mathematical Contribution:** Proves a fundamental geometric trade-off: SVD-based alignment isolates globally shared consensus directions but attenuates task-specific variations, whereas CUR decomposition preserves localized task-specific features from actual parameter rows/columns. Proposes a hybrid SVD+CUR merging scheme.
- **Shared Orthogonal Basis:** Yes (for the SVD component).
- **Sign Consensus:** Examines weight interference and sign preservation.
- **Representation:** LoRA matrices and dense updates.
- **Exact Difference from OSSC:** Analyzes why pure SVD alignment harms task-specific knowledge and why orthogonal projection alone is insufficient without localized column/row preservation.

---

### 2.6 TARA: Preference-Aligned LoRA Merging
- **Full Title:** *Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy*
- **Authors:** CVPR / Preprint Authors
- **Venue & Year:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR 2025**) / arXiv
- **Mathematical Contribution:** Addresses directional anisotropy and subspace coverage degradation during LoRA merging by optimizing merging coefficients under a preference-aligned cross-entropy objective.
- **Shared Orthogonal Basis:** No.
- **Sign Consensus:** No. Uses preference alignment.
- **Exact Difference from OSSC:** Focuses on preference weighting and anisotropy rather than SVD projection or sign consensus.

---

### 2.7 TIES-Merging
- **Full Title:** *Resolving Interference When Merging Models*
- **Authors:** Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, Mohit Bansal
- **Venue & Year:** Advances in Neural Information Processing Systems (**NeurIPS 2023**) (`arXiv:2306.01708`)
- **Mathematical Contribution:** Introduced the Trim, Elect Sign, and Disjoint Merge pipeline in parameter space to resolve parameter redundancy and sign disagreement interference.
- **Shared Orthogonal Basis:** No.
- **Sign Consensus:** Yes (seminal coordinate-wise sign consensus method).
- **Representation:** Raw parameter space.
- **Exact Difference from OSSC:** Operates strictly on raw weights; does not perform subspace decomposition.

---

## 3. Comparative Taxonomy Table

| Publication / Method | Venue & Year | Shared Orthogonal Basis? | Sign Consensus? | Input: LoRA or Dense? | Core Algorithmic Focus | Primary Difference from OSSC |
|---|---|---|---|---|---|---|
| **Task Arithmetic** (Ilharco et al.) | ICLR 2023 | No | No | Dense / FFT | Linear combination in weight space | Baseline addition; no alignment or consensus |
| **TIES-Merging** (Yadav et al.) | NeurIPS 2023 | No | Yes | Dense / FFT | Coordinate-wise Trim, Elect Sign, Disjoint Average | Operates on raw weights, no basis rotation |
| **EMR-Merging** (Huang et al.) | NeurIPS 2024 Spotlight | No | Yes | Dense / FFT | Elect backbone, Directional Mask, Rescale magnitude | Operates on raw weights, no SVD alignment |
| **KnOTS** (Stoica et al.) | ICLR 2025 | Yes (one-sided $U_R$) | Modular (Supports TIES) | LoRA / Dense | Horizontal SVD stacking, right vector partition, aligned merge | **OSSC Formulation 1 is KnOTS instantiated with EMR** |
| **OSRM** (Zhang et al.) | ACL 2025 | Yes (mutually orthogonal) | No | LoRA | Pre-training orthogonal subspace constraint | Mutually orthogonal per-task spaces; not post-hoc shared |
| **CtM** (He et al.) | ICML 2026 | Yes (two-sided $U, V$) | Modular (Linear/TIES) | LoRA factors | Shared low-rank subspace projection before merge | Two-sided factor projection; does not use EMR |
| **ACL 2026 Findings** | ACL 2026 Findings | Yes (SVD) + CUR | No | LoRA | SVD consensus vs. CUR localized individuality trade-off | Highlights SVD's suppression of task-specific features |
| **TARA** | CVPR 2025 | No | No | LoRA | Preference alignment to resolve directional anisotropy | Metric optimization; no orthogonal projection |
| **OSSC (Proposed)** | Proposed | Yes ($U_R$ or $U_R, V_R$) | Yes (EMR) | LoRA | SVD basis projection followed by EMR sign consensus | Combination of KnOTS SVD alignment with EMR operator |
