# Formal Go / No-Go Decision Report: Orthogonal Subspace Sign Consensus (OSSC)

## 1. Executive Decision

### **FINAL VERDICT: NO-GO (As a Standalone Novel Algorithm)**
### **ALTERNATIVE STATUS: CONDITIONAL GO (As an Empirical Ablation Baseline within KnOTS)**

---

## 2. Core Audit Findings & Justification

### 2.1 Fatal Novelty Overlap (Why it is a NO-GO as a "Novel Algorithm")
1. **Direct Equivalence to KnOTS + EMR:**
   OSSC Formulation 1 is mathematically indistinguishable from the official KnOTS framework (Stoica et al., ICLR 2025) instantiated with the EMR merging operator (Huang et al., NeurIPS 2024). KnOTS explicitly designed and open-sourced the exact horizontal concatenation ($W_{\text{cat}}$), joint SVD decomposition ($U_R \Sigma_R V_R^T$), right singular vector partitioning ($V_{R, i}^T$), aligned coordinate merging ($Z_i = \Sigma_R V_{R, i}^T$), and rank-$R$ LoRA factor reconstruction ($B = U_R, A = Z_{\text{merged}}$).
2. **KnOTS is Modular by Design:**
   Stoica et al. explicitly decoupled the SVD alignment step from the choice of inner merging operator $\mathcal{M}$ (demonstrating KnOTS+TA, KnOTS+TIES, and KnOTS+DARE). Substituting $\mathcal{M} = \text{EMR}$ is an ordinary hyperparameter/operator substitution, not a distinct scientific invention.
3. **Reviewer Rejection Risk:**
   Submitting OSSC as an independent novel method to a top-tier machine learning venue (ICML, NeurIPS, ICLR) would lead to immediate rejection for failure to cite and differentiate from KnOTS and CtM.

---

### 2.2 Severe Mathematical & Conceptual Flaws

1. **Non-Invariance Under Eigenspace Rotations:**
   In neural networks, singular value spectra frequently exhibit flat tails where consecutive singular values are nearly equal ($\sigma_j \approx \sigma_{j+1}$). By the Davis-Kahan $\sin \Theta$ theorem, the resulting singular vectors are non-unique and can rotate arbitrarily under floating-point precision changes or weight perturbations. Because EMR executes coordinate-wise sign consensus and maximum-magnitude selection, rotating the coordinate basis completely alters the elected signs and masks.
2. **Category Error on Sign Semantics:**
   In raw parameter space, weights have localized physical meaning: a weight $W_{j, k}$ controls the excitatory or inhibitory connection between input feature $k$ and neuron $j$. Sign consensus resolves functional antagonism on specific connections. In SVD space, coordinate $Z_{i, r, c}$ is an abstract linear combination across hundreds of neurons. Applying coordinate-wise sign voting to abstract spectral projections lacks theoretical justification.
3. **Additive Information Loss:**
   Truncating to rank $R$ adds a permanent subspace residual error $\sum_{j > R} \sigma_j^2$ that is additive to the cross-task interference error. Outlier tasks with subtle, unique knowledge (e.g. GSM8K math reasoning) suffer disproportionately higher truncation loss.

---

## 3. Decision Matrix Summary

| Criterion | Evaluation | Risk Level | Details |
|---|---|---|---|
| **Mathematical Correctness** | Valid derivations, but flawed rotation invariance | High | SVD sign flips cancel, but eigenspace rotations break coordinate-wise EMR. |
| **Algorithmic Novelty** | Zero algorithmic novelty | Critical | Formulation 1 is identical to KnOTS + EMR. Formulation 2 is CtM + EMR. |
| **Theoretical Justification** | Weak | High | Coordinate-wise sign consensus lacks meaning in abstract singular bases. |
| **Computational Feasibility** | Highly feasible | Low | Can be computed efficiently on CPU via Gram matrix formulation. |
| **Experimental Utility** | Useful as a negative control / ablation | Moderate | Serves to test whether EMR outperforms TIES in SVD space. |

---

## 4. Conditions for a "Conditional Go" (Empirical Benchmarking Only)

If the research team desires to experimentally test this combination, it must proceed under the following strict conditions:

1. **Retire the Name "OSSC-Merging":**
   Cease labeling this combination as an independent algorithm ("OSSC"). Label it transparently in all experimental logs and reports as:
   $$\textbf{KnOTS + EMR}$$
2. **Hypothesis Demotion:**
   Frame the investigation not as "proving OSSC superiority," but as answering:
   *"Does replacing TIES with EMR inside the KnOTS SVD-aligned coordinate space offer any empirical advantage across diverse multi-task benchmarks?"*
3. **Mandatory Inclusion of Negative Controls:**
   Must include Baseline 8 (Random Orthogonal Basis + EMR) and Baseline 5 (KnOTS + Task Arithmetic) to verify whether any observed performance changes stem from true spectral alignment or simply from unprincipled coordinate dispersion.
4. **Spectral Gap Diagnostic:**
   Log the spectral condition and singular value gaps $\delta_j = |\sigma_j - \sigma_{j+1}|$ across all layers to quantify whether basis instability degrades downstream reasoning tasks.

---

## 5. Final Recommendation to the Senior Researcher

Do **not** invest resources in developing a standalone paper, codebase, or marketing around "OSSC-Merging." 
Instead, incorporate **KnOTS + EMR** directly as an ablation baseline within our ongoing, mathematically principled **Activation-Distribution-Aware LoRA Merging** investigation.
