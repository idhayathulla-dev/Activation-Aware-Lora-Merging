# Stage E & Stage F: Repository Inspection Report

## 1. Inspection Overview
- **Repository Path:** `c:\Users\idhay\AIML Research\activation-aware-lora-merging`
- **Date:** 2026-11-16
- **Objective:** Evaluate readiness for Stage E (Independent LoRA Adapter Training) and Stage F (Experimental Provenance & Reproducibility Audit).

---

## 2. Inventory of Reusable Components
1. **Model Loading (`src/model_loader.py`):**
   - Clean separation of base model and tokenizer initialization.
   - Robust offline mock fallback (`MockQwenModel`, `MockTokenizer`) preserving exact Qwen2.5-0.5B-Instruct architecture ($d_{in}=896, \text{vocab}=151,936$, target modules `q_proj`, `v_proj`).
   - `verify_forward_pass` diagnostics checking logits shape and numerical finiteness.
2. **Preprocessing & Task Prompt Formatting (`src/data_preprocessing.py`):**
   - Task prompt formatters for BoolQ, GSM8K, and ARC-Easy.
   - Causal LM label masking (`-100` for prompt tokens, valid token IDs for target completion).
   - Answer normalization for GSM8K (`normalize_numeric_answer`) and ARC-Easy (`normalize_arc_label`).
   - Token length tracking and truncation statistics.
3. **Matrix Utilities & Merging Modules (`src/matrix_utils.py`, `src/merging/`):**
   - Stage C verified implementations for Task Arithmetic, TIES, KnOTS, and Activation-Aware KnOTS.
4. **Existing Unit Tests (`tests/`):**
   - 20 / 20 tests passing on local environment.

---

## 3. Missing Components
1. **Training Framework (`src/training/`):**
   - No trainer module, optimizer configuration, learning rate scheduler, or validation loss tracking.
   - No checkpoint manager conforming to PEFT saving conventions (`adapter_config.json`, `adapter_model.safetensors`, `training_metadata.json`).
2. **Provenance & Split Integrity Auditing:**
   - No explicit separation between `final` mode (enforcing authentic HuggingFace datasets) and `development` mode (permitting synthetic mocks).
   - No content-based SHA-256 fingerprinting for individual examples and dataset partitions.
   - No artifact generators for `artifacts/provenance/dataset_manifest.json`, `split_integrity_report.json`, and `split_integrity_report.md`.
3. **Execution Scripts:**
   - Need `scripts/audit_datasets.py`.
   - Need `scripts/train_adapter.py`.
   - Need `scripts/train_all_adapters.py` (with `--preflight`, `--development`, `--allow-synthetic`).
   - Need `scripts/verify_adapters.py`.

---

## 4. Potential Data-Integrity Problems
1. **Index-Only Fingerprints (Root Cause Identified):**
   - The Stage D implementation of `compute_split_fingerprint` in `src/dataset_loader.py` serialized only the integer partition indices (e.g. `[0, 1, 2, ...]`).
   - Because all three synthetic datasets contained 600 examples partitioned with `seed=42`, the list of indices was identical across all three tasks, producing identical fingerprint prefixes (`185e92a7d48fcee7` for calibration and `93be0e0689f82b88` for evaluation).
   - This was an index-only hash, not a content hash. Content-based hashing incorporating dataset name, config, example IDs, normalized input, and normalized target must replace this.
2. **Silent Synthetic Fallback Risk:**
   - In Stage D, `load_task_dataset` automatically caught exceptions and fell back to synthetic mocks.
   - In `final` mode, this must be strictly forbidden: missing or unreachable real Hugging Face datasets must immediately raise a fatal exception.

---

## 5. Required Modifications & Backward Compatibility
- Refactor `src/dataset_loader.py` to:
  - Add explicit `mode="final"` (default) vs `mode="development"`.
  - In `final` mode, raise `RuntimeError` if real data cannot be loaded; never fall back silently.
  - Implement `compute_content_sha256` hashing example text and metadata.
  - Generate full provenance manifests.
- Maintain existing functions (`partition_dataset_indices`, `TASK_SCHEMAS`, `validate_dataset_schema`) to preserve complete backward compatibility with Stage D tests.

---

## 6. Implementation Sequence
1. **Phase 1 (Stage F):** Implement content-based dataset provenance, strict mode validation, split integrity checking, and `scripts/audit_datasets.py`.
2. **Phase 2 (Stage E):** Implement `src/training/` (config, utils, metrics, checkpoint manager, trainer) with exact LoRA rank=8, alpha=16, causal LM loss masking.
3. **Phase 3 (CLI):** Implement `scripts/train_adapter.py`, `scripts/train_all_adapters.py` (with `--preflight`), and `scripts/verify_adapters.py`.
4. **Phase 4 (Tests):** Build comprehensive unit tests across provenance, split integrity, training config, label masking, checkpoint management, and adapter verification.
5. **Phase 5 (Audit & Preflight):** Run test suite, execute training preflight, and produce Google Colab GPU execution guide.
