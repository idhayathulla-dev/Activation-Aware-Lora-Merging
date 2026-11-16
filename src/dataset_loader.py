"""
Dataset Loader: Loads benchmark datasets, verifies schemas, and enforces strictly disjoint splits
preserving official benchmark evaluation partitions according to research integrity standards.

Policy:
- Official evaluation/test partitions are NEVER pooled into training sources.
- Official evaluation partitions are 100% preserved (never silently truncated or deleted).
- Training duplicates that match evaluation/validation examples are excluded from the training pool.
- Calibration examples (128) are strictly carved out from permitted source training partitions.
- Exact dataset commit SHAs and exclusion records are logged for complete experimental provenance.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


# Pinned resolved commit SHAs for official benchmark reproducibility
PINNED_RESOLVED_COMMITS = {
    "google/boolq": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
}

PREPROCESSING_VERSION = "2.1.0"


def resolve_dataset_commit(dataset_name: str) -> str:
    """Resolves the exact Git commit SHA for a Hugging Face dataset."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(dataset_name)
        if getattr(info, "sha", None):
            return str(info.sha)
    except Exception:
        pass
    return PINNED_RESOLVED_COMMITS.get(dataset_name, "unknown_commit")


# ==============================================================================
# 1. Deterministic Synthetic Mock Data Generators (Development / Unit Tests)
# ==============================================================================

def generate_mock_boolq_data(num_samples: int = 600, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates synthetic BoolQ examples conforming to google/boolq schema."""
    rng = np.random.RandomState(seed)
    data = []
    passages = [
        "Photosynthesis is the process used by plants to convert light into chemical energy.",
        "The Earth revolves around the Sun in an elliptical orbit taking approximately 365 days.",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure at sea level.",
        "A leap year contains 366 days instead of the usual 365 days in the Gregorian calendar.",
        "The human heart has four chambers: two atria and two ventricles that pump blood."
    ]
    for i in range(num_samples):
        p_idx = rng.randint(0, len(passages))
        ans = bool(rng.randint(0, 2))
        data.append({
            "id": f"boolq_synth_{i:04d}",
            "question": f"Is fact {i} related to passage {p_idx}?",
            "passage": passages[p_idx],
            "answer": ans,
            "title": f"Topic {p_idx}"
        })
    return data


def generate_mock_gsm8k_data(num_samples: int = 600, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates synthetic GSM8K examples conforming to openai/gsm8k schema."""
    rng = np.random.RandomState(seed)
    data = []
    for i in range(num_samples):
        a = rng.randint(2, 20)
        b = rng.randint(3, 15)
        ans = a * b
        data.append({
            "id": f"gsm8k_synth_{i:04d}",
            "question": f"In transaction {i}, Janet buys {a} boxes of pencils with {b} pencils each. How many pencils does she have?",
            "answer": f"In transaction {i}, Janet has {a} boxes. Each has {b} pencils. So {a} * {b} = {ans}.\n#### {ans}"
        })
    return data


def generate_mock_arc_data(num_samples: int = 600, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates synthetic ARC-Easy examples conforming to allenai/ai2_arc schema."""
    rng = np.random.RandomState(seed)
    data = []
    keys = ["A", "B", "C", "D"]
    for i in range(num_samples):
        correct_idx = rng.randint(0, 4)
        data.append({
            "id": f"arc_synth_{i:04d}",
            "question": f"Which process describes science phenomenon {i}?",
            "choices": {
                "text": ["Evaporation", "Condensation", "Precipitation", "Sublimation"],
                "label": ["A", "B", "C", "D"]
            },
            "answerKey": keys[correct_idx]
        })
    return data


# ==============================================================================
# 2. Schema Validation
# ==============================================================================

TASK_SCHEMAS = {
    "boolq": {
        "required_fields": ["question", "passage", "answer"],
        "types": {"question": str, "passage": str, "answer": bool}
    },
    "gsm8k": {
        "required_fields": ["question", "answer"],
        "types": {"question": str, "answer": str}
    },
    "arc_easy": {
        "required_fields": ["question", "choices", "answerKey"],
        "types": {"question": str, "choices": dict, "answerKey": str}
    },
}

def validate_dataset_schema(dataset: List[Dict[str, Any]], task_name: str) -> bool:
    """Validates that dataset examples conform to expected task schema."""
    if not dataset:
        raise ValueError(f"Dataset for task '{task_name}' is empty.")
    schema = TASK_SCHEMAS.get(task_name)
    if not schema:
        raise ValueError(f"No schema registered for task '{task_name}'.")

    for i, ex in enumerate(dataset[:20]):
        for field in schema["required_fields"]:
            if field not in ex:
                raise ValueError(f"Example {i} in task '{task_name}' missing required field '{field}'.")
            expected_type = schema["types"][field]
            if not isinstance(ex[field], expected_type):
                raise TypeError(
                    f"Example {i} field '{field}' has type {type(ex[field])}, expected {expected_type}."
                )
    return True


# ==============================================================================
# 3. Text Normalization, Canonical Representation, and Cryptographic Fingerprints
# ==============================================================================

def normalize_text_for_hash(text: str) -> str:
    """Normalizes text by lowercasing and standardizing whitespace."""
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.strip().lower().split())


def extract_canonical_input(example: Dict[str, Any], task_name: str) -> Tuple[str, str]:
    """Extracts normalized (primary_input, secondary_context) tuple."""
    if task_name == "boolq":
        q = normalize_text_for_hash(example.get("question", ""))
        p = normalize_text_for_hash(example.get("passage", ""))
        return (q, p)
    elif task_name == "gsm8k":
        q = normalize_text_for_hash(example.get("question", ""))
        return (q, "")
    elif task_name in ["arc_easy", "ai2_arc"]:
        q = normalize_text_for_hash(example.get("question", ""))
        choices = example.get("choices", {})
        texts = " ".join(choices.get("text", []))
        c = normalize_text_for_hash(texts)
        return (q, c)
    else:
        q = normalize_text_for_hash(example.get("input_text", example.get("question", str(example))))
        return (q, "")


def extract_canonical_target(example: Dict[str, Any], task_name: str) -> str:
    """Extracts normalized target label for comparison."""
    if task_name == "boolq":
        ans = example.get("answer", False)
        return "yes" if ans else "no"
    elif task_name == "gsm8k":
        raw_ans = str(example.get("answer", ""))
        if "####" in raw_ans:
            return normalize_text_for_hash(raw_ans.split("####")[-1])
        return normalize_text_for_hash(raw_ans)
    elif task_name in ["arc_easy", "ai2_arc"]:
        return str(example.get("answerKey", "")).strip().upper()
    else:
        return normalize_text_for_hash(str(example.get("label", example.get("answer", ""))))


def compute_example_content_sha256(
    example: Dict[str, Any],
    task_name: str = "generic",
    dataset_name: str = "generic",
    dataset_config: Optional[str] = None
) -> str:
    """
    Computes a cryptographic SHA-256 fingerprint for an individual example based on:
    - task_name, dataset_name, dataset_config
    - stable example ID
    - normalized input text (passage/question/choices/input_text)
    - normalized target answer (label/answer/answerKey)
    """
    ex_id = str(example.get("id", ""))
    cinp = extract_canonical_input(example, task_name)
    ctgt = extract_canonical_target(example, task_name)
    payload = f"{task_name}|{dataset_name}|{dataset_config or ''}|{ex_id}|{cinp[0]}|{cinp[1]}|{ctgt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_partition_content_sha256(
    examples: List[Dict[str, Any]],
    task_name: str = "generic",
    dataset_name: str = "generic",
    dataset_config: Optional[str] = None
) -> str:
    """
    Computes a SHA-256 fingerprint for an entire partition by aggregating
    the individual content hashes of all examples in sorted order.
    """
    if not examples:
        return hashlib.sha256(b"empty_partition").hexdigest()

    hashes = [
        compute_example_content_sha256(ex, task_name, dataset_name, dataset_config)
        for ex in examples
    ]
    sorted_hashes = sorted(hashes)
    combined = "\n".join(sorted_hashes).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def compute_split_fingerprint(indices: List[int]) -> str:
    """Computes SHA-256 fingerprint for a list of partition indices."""
    serialized = ",".join(str(idx) for idx in indices).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


# ==============================================================================
# 4. Cross-Split Duplicate Detection & Training-Side Exclusion Engine
# ==============================================================================

def filter_train_against_eval_duplicates(
    train_examples: List[Dict[str, Any]],
    eval_splits_dict: Dict[str, List[Dict[str, Any]]],
    task_name: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Identifies and excludes training examples that duplicate any example in evaluation or validation.

    Guarantees:
    - Official evaluation examples are NEVER deleted or modified.
    - If a training example duplicates an official evaluation example, it is excluded from training.
    - Detects whether the duplicate has matching target or conflicting target.
    - Records comprehensive exclusion metadata.
    """
    eval_lookup: Dict[Tuple[str, str], List[Tuple[str, int, str, Dict[str, Any]]]] = {}
    for split_name, exs in eval_splits_dict.items():
        for idx, ex in enumerate(exs):
            cinp = extract_canonical_input(ex, task_name)
            ctgt = extract_canonical_target(ex, task_name)
            if cinp not in eval_lookup:
                eval_lookup[cinp] = []
            eval_lookup[cinp].append((split_name, idx, ctgt, ex))

    clean_train = []
    excluded_duplicates = []

    for tr_idx, ex in enumerate(train_examples):
        cinp = extract_canonical_input(ex, task_name)
        ctgt = extract_canonical_target(ex, task_name)

        if cinp in eval_lookup:
            matches = eval_lookup[cinp]
            matched_split, matched_idx, matched_tgt, matched_ex = matches[0]
            is_conflicting = (ctgt != matched_tgt)
            record = {
                "train_index": tr_idx,
                "train_id": str(ex.get("id", f"train_{tr_idx}")),
                "matched_split": matched_split,
                "matched_index": matched_idx,
                "matched_id": str(matched_ex.get("id", f"{matched_split}_{matched_idx}")),
                "question": str(ex.get("question", ""))[:120],
                "train_target": ctgt,
                "eval_target": matched_tgt,
                "is_conflicting": is_conflicting,
                "exclusion_reason": "conflict_with_evaluation" if is_conflicting else "duplicate_of_official_evaluation"
            }
            excluded_duplicates.append(record)
        else:
            clean_train.append(ex)

    return clean_train, excluded_duplicates


# ==============================================================================
# 5. Deterministic Disjoint Split Partitioning (Legacy / Mock compatibility)
# ==============================================================================

def partition_dataset_indices(
    total_count: int,
    calib_count: int = 128,
    val_count: int = 100,
    eval_count: int = 200,
    seed: int = 42
) -> Dict[str, List[int]]:
    """Deterministically partitions single-pool indices for mock generators."""
    if total_count < (calib_count + val_count + eval_count):
        raise ValueError(
            f"Total dataset size ({total_count}) too small for requested splits "
            f"(calib: {calib_count}, val: {val_count}, eval: {eval_count})."
        )

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(total_count).tolist()

    eval_indices = sorted(shuffled[:eval_count])
    calib_indices = sorted(shuffled[eval_count : eval_count + calib_count])
    val_indices = sorted(shuffled[eval_count + calib_count : eval_count + calib_count + val_count])
    train_indices = sorted(shuffled[eval_count + calib_count + val_count:])

    return {
        "train": train_indices,
        "calibration": calib_indices,
        "validation": val_indices,
        "evaluation": eval_indices,
    }


# ==============================================================================
# 6. Split Integrity and Leakage Verification
# ==============================================================================

def verify_split_integrity(
    split_datasets: Dict[str, List[Dict[str, Any]]],
    task_name: str = "generic",
    dataset_name: str = "generic",
    dataset_config: Optional[str] = None
) -> Dict[str, Any]:
    """
    Performs strict leakage and integrity checks across partitions:
    - No overlapping content hashes between any partitions
    - No duplicate example IDs across partitions
    - No duplicate canonical inputs between training/calib/val and evaluation
    - No conflicting targets across partitions
    """
    partition_names = list(split_datasets.keys())
    partition_hashes: Dict[str, Set[str]] = {}
    partition_ids: Dict[str, Set[str]] = {}
    partition_inputs: Dict[str, Dict[Tuple[str, str], str]] = {}

    leakage_detected = False
    violations = []

    for name, examples in split_datasets.items():
        hashes = set()
        ids = set()
        inputs = {}

        for idx, ex in enumerate(examples):
            h = compute_example_content_sha256(ex, task_name, dataset_name, dataset_config)
            hashes.add(h)

            ex_id = str(ex.get("id", f"{name}_{idx}"))
            ids.add(ex_id)

            cinp = extract_canonical_input(ex, task_name)
            ctgt = extract_canonical_target(ex, task_name)
            inputs[cinp] = ctgt

        partition_hashes[name] = hashes
        partition_ids[name] = ids
        partition_inputs[name] = inputs

    # Check pairwise disjointness
    for i in range(len(partition_names)):
        for j in range(i + 1, len(partition_names)):
            p1 = partition_names[i]
            p2 = partition_names[j]

            # Content hash overlap
            overlap_hashes = partition_hashes[p1] & partition_hashes[p2]
            if overlap_hashes:
                leakage_detected = True
                violations.append(f"Content hash overlap between '{p1}' and '{p2}': {len(overlap_hashes)} examples.")

            # Input overlap
            common_inputs = set(partition_inputs[p1].keys()) & set(partition_inputs[p2].keys())
            if common_inputs:
                leakage_detected = True
                conflicts = [cinp for cinp in common_inputs if partition_inputs[p1][cinp] != partition_inputs[p2][cinp]]
                if conflicts:
                    violations.append(
                        f"Conflicting duplicate labels between '{p1}' and '{p2}': {len(conflicts)} examples."
                    )
                else:
                    violations.append(
                        f"Canonical input overlap between '{p1}' and '{p2}': {len(common_inputs)} examples."
                    )

    return {
        "task_name": task_name,
        "integrity_passed": not leakage_detected,
        "leakage_detected": leakage_detected,
        "violations": violations,
        "partition_sizes": {p: len(examples) for p, examples in split_datasets.items()}
    }

# Backwards compatibility alias
verify_split_disjointness = verify_split_integrity


# ==============================================================================
# 7. Unified Task Dataset Loading Interface (Benchmark Preservation Policy)
# ==============================================================================

def load_task_dataset(
    task_name: str,
    calib_count: int = 128,
    val_count: Optional[int] = 100,
    eval_count: Optional[int] = None,
    seed: int = 42,
    use_mock: bool = False,
    mode: str = "final",
    allow_synthetic: bool = False,
    require_real_huggingface_data: bool = True,
    metadata_save_dir: Optional[str] = "results/processed"
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Loads and partitions task dataset adhering to official benchmark preservation policy.

    Policy:
    - BoolQ: Training from official train. 128 calib, 100 val reserved from train. Evaluation = official validation.
    - GSM8K: Training from official train. 128 calib, 100 val reserved from train. Evaluation = official test.
    - ARC-Easy: Training from official train. 128 calib reserved from train. Validation = official validation. Evaluation = official test.
    - Cross-split duplicate handling: Training examples duplicating evaluation/val are excluded from training. Evaluation remains 100% intact.
    """
    is_final_mode = (mode == "final")
    permit_synthetic = allow_synthetic or (mode == "development")

    dataset_name = ""
    dataset_config = None
    source_split_names: Dict[str, str] = {}
    source = "huggingface"
    is_synthetic = False

    if task_name == "boolq":
        dataset_name = "google/boolq"
        dataset_config = None
        source_split_names = {"train": "train", "evaluation": "validation"}
    elif task_name == "gsm8k":
        dataset_name = "openai/gsm8k"
        dataset_config = "main"
        source_split_names = {"train": "train", "evaluation": "test"}
    elif task_name in ["arc_easy", "ai2_arc"]:
        dataset_name = "allenai/ai2_arc"
        dataset_config = "ARC-Easy"
        source_split_names = {"train": "train", "validation": "validation", "evaluation": "test"}
    else:
        raise ValueError(f"Unknown task: '{task_name}'")

    raw_splits: Dict[str, List[Dict[str, Any]]] = {}
    resolved_commit = resolve_dataset_commit(dataset_name)

    if not use_mock and require_real_huggingface_data:
        try:
            from datasets import load_dataset
            print(f"[DatasetLoader] Loading Hugging Face dataset '{dataset_name}' (config: {dataset_config})...")
            ds = load_dataset(dataset_name, dataset_config) if dataset_config else load_dataset(dataset_name)

            # Confirm all required source splits exist
            for target_role, hf_split_name in source_split_names.items():
                if hf_split_name not in ds:
                    raise RuntimeError(
                        f"Missing required source split '{hf_split_name}' for task '{task_name}' in dataset '{dataset_name}'."
                    )
                raw_splits[target_role] = [dict(x) for x in ds[hf_split_name]]

        except Exception as e:
            if is_final_mode and not permit_synthetic:
                raise RuntimeError(
                    f"FINAL MODE FAILURE: Cannot load real dataset '{dataset_name}' ({e}). "
                    f"Synthetic fallback is forbidden in final mode. Ensure network access and 'datasets' package."
                )
            print(f"[DatasetLoader] Real dataset load failed ({e}). Permitted fallback to mock generator (mode={mode})...")
            use_mock = True

    if use_mock or not raw_splits:
        if is_final_mode and not permit_synthetic:
            raise RuntimeError(
                f"FINAL MODE VIOLATION: Synthetic mock dataset requested for '{task_name}', "
                f"but allow_synthetic=False and mode='final'."
            )
        is_synthetic = True
        source = "synthetic_mock"
        resolved_commit = "synthetic_dev_generator"
        if task_name == "boolq":
            raw_mock = generate_mock_boolq_data(num_samples=600, seed=seed)
        elif task_name == "gsm8k":
            raw_mock = generate_mock_gsm8k_data(num_samples=600, seed=seed)
        elif task_name in ["arc_easy", "ai2_arc"]:
            raw_mock = generate_mock_arc_data(num_samples=600, seed=seed)

        # Mock split allocation for unit tests
        e_n = eval_count if isinstance(eval_count, int) else 200
        v_n = val_count if isinstance(val_count, int) else 100
        c_n = calib_count if isinstance(calib_count, int) else 128
        raw_splits["evaluation"] = raw_mock[:e_n]
        raw_splits["calibration"] = raw_mock[e_n : e_n + c_n]
        raw_splits["validation"] = raw_mock[e_n + c_n : e_n + c_n + v_n]
        raw_splits["train"] = raw_mock[e_n + c_n + v_n :]

    # Validate schemas of loaded splits
    for role, exs in raw_splits.items():
        validate_dataset_schema(exs, task_name)

    # --------------------------------------------------------------------------
    # Official Benchmark Preservation & Disjoint Partition Allocation
    # --------------------------------------------------------------------------
    original_source_counts = {role: len(exs) for role, exs in raw_splits.items()}
    excluded_duplicates: List[Dict[str, Any]] = []

    if not is_synthetic:
        # 1. Official Evaluation Split is 100% PRESERVED
        final_evaluation = raw_splits["evaluation"]

        # 2. For ARC-Easy: Official Validation Split is 100% PRESERVED
        if task_name in ["arc_easy", "ai2_arc"]:
            final_validation = raw_splits["validation"]
            eval_lookup_splits = {"test": final_evaluation, "validation": final_validation}
        else:
            final_validation = []
            eval_lookup_splits = {"evaluation": final_evaluation}

        # 3. Detect and Exclude Training-side Duplicates matching evaluation/validation
        source_train = raw_splits["train"]
        clean_train_candidates, excluded_duplicates = filter_train_against_eval_duplicates(
            train_examples=source_train,
            eval_splits_dict=eval_lookup_splits,
            task_name=task_name
        )

        # 4. Partition clean training candidates using deterministic seed
        rng = np.random.RandomState(seed)
        permuted_idx = rng.permutation(len(clean_train_candidates)).tolist()

        # Calibration: 128 examples carved out of clean train
        calib_idx = permuted_idx[:calib_count]
        final_calibration = [clean_train_candidates[i] for i in calib_idx]

        if task_name in ["arc_easy", "ai2_arc"]:
            # Remaining clean train goes to train partition
            train_idx = permuted_idx[calib_count:]
            final_train = [clean_train_candidates[i] for i in train_idx]
        else:
            # Reserve val_count (100) from train
            v_count = val_count if val_count is not None else 100
            val_idx = permuted_idx[calib_count : calib_count + v_count]
            train_idx = permuted_idx[calib_count + v_count :]
            final_validation = [clean_train_candidates[i] for i in val_idx]
            final_train = [clean_train_candidates[i] for i in train_idx]

        split_datasets = {
            "train": final_train,
            "calibration": final_calibration,
            "validation": final_validation,
            "evaluation": final_evaluation
        }

    else:
        # Development / synthetic mode splits
        split_datasets = {
            "train": raw_splits["train"],
            "calibration": raw_splits["calibration"],
            "validation": raw_splits["validation"],
            "evaluation": raw_splits["evaluation"]
        }

    # --------------------------------------------------------------------------
    # Leakage and Integrity Verification
    # --------------------------------------------------------------------------
    integrity_audit = verify_split_integrity(
        split_datasets=split_datasets,
        task_name=task_name,
        dataset_name=dataset_name,
        dataset_config=dataset_config
    )
    if not integrity_audit["integrity_passed"]:
        raise ValueError(f"Integrity check failed for task '{task_name}': {integrity_audit['violations']}")

    # Content-based SHA-256 fingerprints
    content_fingerprints = {
        split: compute_partition_content_sha256(
            examples=split_datasets[split],
            task_name=task_name,
            dataset_name=dataset_name,
            dataset_config=dataset_config
        )
        for split in split_datasets
    }

    metadata = {
        "task_name": task_name,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_revision": resolved_commit,
        "preprocessing_version": PREPROCESSING_VERSION,
        "source": source,
        "source_splits": source_split_names,
        "is_synthetic": is_synthetic,
        "seed": seed,
        "original_source_counts": original_source_counts,
        "excluded_duplicates_count": len(excluded_duplicates),
        "excluded_duplicates": excluded_duplicates,
        "split_sizes": {split: len(data) for split, data in split_datasets.items()},
        "content_fingerprints": content_fingerprints,
        "integrity_audit": integrity_audit
    }

    if metadata_save_dir:
        save_path = Path(metadata_save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        meta_file = save_path / f"dataset_splits_{task_name}.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return split_datasets, metadata
