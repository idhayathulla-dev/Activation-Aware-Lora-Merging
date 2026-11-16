"""
Unit tests for Model Loader, Dataset Loader, and Data Preprocessing (Stage D).
"""

import pytest
import torch
import numpy as np

from src.model_loader import load_model_and_tokenizer, verify_forward_pass
from src.dataset_loader import (
    load_task_dataset,
    partition_dataset_indices,
    validate_dataset_schema,
    generate_mock_boolq_data,
    generate_mock_gsm8k_data,
    generate_mock_arc_data,
)
from src.data_preprocessing import (
    format_boolq_prompt,
    format_boolq_target,
    format_gsm8k_prompt,
    extract_gsm8k_answer,
    normalize_numeric_answer,
    format_arc_prompt,
    normalize_arc_label,
    extract_prompt_and_target,
    preprocess_dataset,
)


@pytest.fixture
def rng_seed():
    torch.manual_seed(42)
    np.random.seed(42)


# ==============================================================================
# 1. Model Loading and Forward Pass Verification
# ==============================================================================

def test_model_loading_and_metadata(rng_seed):
    """Test model loading and verify architecture metadata conformance."""
    model, tokenizer, meta = load_model_and_tokenizer(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        prefer_mock=True
    )
    print("\n[test_model_loading_and_metadata]")
    print(f"  Architecture     : {meta['architecture']}")
    print(f"  Hidden Size      : {meta['hidden_size']}")
    print(f"  Vocab Size       : {meta['vocab_size']}")
    print(f"  Target Modules   : {meta['target_modules']}")
    print(f"  Device           : {meta['device']}")

    assert meta["hidden_size"] == 896
    assert meta["vocab_size"] == 151936
    assert "q_proj" in meta["target_modules"]
    assert "v_proj" in meta["target_modules"]


def test_forward_pass_verification(rng_seed):
    """Test model forward pass execution and check logits finiteness."""
    model, tokenizer, _ = load_model_and_tokenizer(prefer_mock=True)
    test_text = "Passage: The sky is blue. Question: Is the sky blue? Answer: yes"
    diag = verify_forward_pass(model, tokenizer, test_text=test_text)

    print("\n[test_forward_pass_verification]")
    print(f"  Input Tokens  : {diag['input_tokens']}")
    print(f"  Logits Shape  : {diag['logits_shape']}")
    print(f"  Is Finite     : {diag['is_finite']}")
    print(f"  Latency (ms)  : {diag['latency_ms']}")

    assert diag["verified"] is True
    assert diag["is_finite"] is True
    assert diag["logits_shape"][-1] == 151936


# ==============================================================================
# 2. Dataset Schema Validation & Split Disjointness
# ==============================================================================

def test_dataset_schemas_validation():
    """Verify schemas for BoolQ, GSM8K, and ARC-Easy."""
    boolq_data = generate_mock_boolq_data(10)
    gsm8k_data = generate_mock_gsm8k_data(10)
    arc_data = generate_mock_arc_data(10)

    assert validate_dataset_schema(boolq_data, "boolq") is True
    assert validate_dataset_schema(gsm8k_data, "gsm8k") is True
    assert validate_dataset_schema(arc_data, "arc_easy") is True

    # Check error on invalid schema
    with pytest.raises(ValueError):
        invalid_data = [{"invalid_key": "val"}]
        validate_dataset_schema(invalid_data, "boolq")


def test_split_disjointness_and_zero_leakage():
    """
    Requirement 2: Strictly verify that train, calibration, validation,
    and evaluation splits have ZERO overlap (no data leakage).
    """
    total_count = 600
    calib_count = 128
    val_count = 100
    eval_count = 200

    splits = partition_dataset_indices(
        total_count=total_count,
        calib_count=calib_count,
        val_count=val_count,
        eval_count=eval_count,
        seed=42
    )

    s_train = set(splits["train"])
    s_calib = set(splits["calibration"])
    s_val = set(splits["validation"])
    s_eval = set(splits["evaluation"])

    print("\n[test_split_disjointness_and_zero_leakage]")
    print(f"  Train Count : {len(s_train)}")
    print(f"  Calib Count : {len(s_calib)} (Target: 128)")
    print(f"  Val Count   : {len(s_val)}")
    print(f"  Eval Count  : {len(s_eval)}")

    assert len(s_calib) == 128, f"Calibration count must be exactly 128, got {len(s_calib)}"
    assert len(s_train & s_calib) == 0, "Train and Calibration overlap!"
    assert len(s_train & s_eval) == 0, "Train and Evaluation overlap!"
    assert len(s_calib & s_eval) == 0, "Calibration and Evaluation overlap!"
    assert len(s_val & s_eval) == 0, "Validation and Evaluation overlap!"
    assert len(s_train) + len(s_calib) + len(s_val) + len(s_eval) == total_count


def test_deterministic_split_reproducibility():
    """Verify that identical random seeds produce identical split indices."""
    splits_1 = partition_dataset_indices(500, calib_count=128, val_count=50, eval_count=100, seed=42)
    splits_2 = partition_dataset_indices(500, calib_count=128, val_count=50, eval_count=100, seed=42)
    splits_diff = partition_dataset_indices(500, calib_count=128, val_count=50, eval_count=100, seed=99)

    assert splits_1["train"] == splits_2["train"]
    assert splits_1["calibration"] == splits_2["calibration"]
    assert splits_1["evaluation"] == splits_2["evaluation"]
    assert splits_1["train"] != splits_diff["train"]


# ==============================================================================
# 3. Prompt Formatting and Answer Extraction
# ==============================================================================

def test_boolq_formatting():
    """Test BoolQ prompt and target extraction."""
    ex_true = {"passage": "Dogs are domesticated mammals.", "question": "Is a dog a mammal", "answer": True}
    ex_false = {"passage": "Ice is solid water.", "question": "Is ice liquid water?", "answer": False}

    p_true, t_true, r_true = extract_prompt_and_target(ex_true, "boolq")
    p_false, t_false, r_false = extract_prompt_and_target(ex_false, "boolq")

    assert "Passage: Dogs are domesticated mammals." in p_true
    assert "Question: Is a dog a mammal?" in p_true
    assert t_true == "yes"
    assert t_false == "no"
    assert r_true is True
    assert r_false is False


def test_gsm8k_formatting_and_numeric_isolation():
    """
    Test GSM8K formatting:
    Prompt must NOT expose the answer.
    Answer extraction must isolate numeric target cleanly.
    """
    raw_answer = "He bought 3 books for $15 each. Total = 3 * 15 = 45.\n#### 45"
    ex = {"question": "How much did he spend?", "answer": raw_answer}

    prompt, target, numeric = extract_prompt_and_target(ex, "gsm8k")

    # Prompt must NOT leak answer
    assert "45" not in prompt
    assert "Total =" not in prompt
    assert "Question: How much did he spend?" in prompt
    assert "####" not in prompt

    # Target contains reasoning
    assert "The answer is 45." in target
    assert numeric == "45"

    # Numeric normalization tests
    assert normalize_numeric_answer("$1,250.00") == "1250"
    assert normalize_numeric_answer("  36.5% ") == "36.5"
    assert normalize_numeric_answer("42.") == "42"


def test_arc_formatting_and_label_standardization():
    """Test ARC-Easy prompt formatting and standard choice letter mapping."""
    ex_alpha = {
        "question": "What is water freezing point?",
        "choices": {"text": ["0 C", "100 C", "50 C", "25 C"], "label": ["A", "B", "C", "D"]},
        "answerKey": "A"
    }
    ex_numeric = {
        "question": "What is water boiling point?",
        "choices": {"text": ["0 C", "100 C", "50 C", "25 C"], "label": ["1", "2", "3", "4"]},
        "answerKey": "2"
    }

    p_alpha, t_alpha, r_alpha = extract_prompt_and_target(ex_alpha, "arc_easy")
    p_numeric, t_numeric, r_numeric = extract_prompt_and_target(ex_numeric, "arc_easy")

    assert "(A) 0 C" in p_alpha
    assert t_alpha == "A"
    assert "(B) 100 C" in p_numeric
    assert t_numeric == "B"  # Mapped from '2' to 'B'


# ==============================================================================
# 4. Tokenization, Label Masking, and Truncation Tracking
# ==============================================================================

def test_preprocessing_and_label_masking(rng_seed):
    """
    Test preprocessing for causal LM training:
    Prompt tokens must be masked with -100 so loss is computed only on completion.
    """
    _, tokenizer, _ = load_model_and_tokenizer(prefer_mock=True)
    examples = [
        {"passage": "Passage text A", "question": "Question A?", "answer": True},
        {"passage": "Passage text B", "question": "Question B?", "answer": False},
    ]

    processed = preprocess_dataset(
        examples=examples,
        task_name="boolq",
        tokenizer=tokenizer,
        max_seq_length=64,
        is_training=True
    )

    stats = processed["statistics"]
    print("\n[test_preprocessing_and_label_masking]")
    print(f"  Total Processed : {stats['total_examples']}")
    print(f"  Mean Length     : {stats['mean_token_length']}")
    print(f"  Truncation Rate : {stats['truncation_rate']}")

    assert stats["total_examples"] == 2
    assert stats["truncation_count"] == 0

    # Verify label masking: first tokens are -100
    labels = processed["labels"][0]
    assert -100 in labels
    # Not all labels are -100 (completion tokens must have valid token ids)
    non_masked = [l for l in labels if l != -100]
    assert len(non_masked) > 0, "All tokens were masked in labels!"
