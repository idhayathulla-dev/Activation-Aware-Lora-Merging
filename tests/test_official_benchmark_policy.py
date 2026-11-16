"""
Unit tests specifically verifying the 7 rules of the Official Benchmark Preservation Policy:

1. Official test examples never enter the training partition.
2. Calibration examples come only from the permitted source training partition.
3. Official validation/test partitions are preserved (100% of benchmark examples retained).
4. Cross-split duplicate handling preserves evaluation examples and excludes training duplicates.
5. Conflicting duplicate targets are properly flagged.
6. Actual dataset revisions (commit SHAs) are recorded and not placeholders.
7. Final mode continues to reject synthetic data.
"""

import pytest
from unittest.mock import patch
from src.dataset_loader import (
    load_task_dataset,
    filter_train_against_eval_duplicates,
    extract_canonical_input,
    extract_canonical_target,
    resolve_dataset_commit,
    PINNED_RESOLVED_COMMITS,
    PREPROCESSING_VERSION,
)


# ==============================================================================
# Rule 1 & 2: Evaluation Isolation & Calibration Source Verification
# ==============================================================================

def test_official_test_examples_never_enter_training():
    """
    Test 1: Official evaluation/test examples NEVER enter the training partition.
    Any candidate in train matching an evaluation example must be excluded.
    """
    eval_examples = [
        {"id": "eval_0", "question": "What is the speed of light?", "answerKey": "A", "choices": {"text": ["3e8 m/s"], "label": ["A"]}},
        {"id": "eval_1", "question": "Is water wet?", "answerKey": "B", "choices": {"text": ["Yes"], "label": ["B"]}},
    ]
    train_candidates = [
        {"id": "tr_0", "question": "What is the capital of France?", "answerKey": "C", "choices": {"text": ["Paris"], "label": ["C"]}},
        {"id": "tr_1", "question": "What is the speed of light?", "answerKey": "A", "choices": {"text": ["3e8 m/s"], "label": ["A"]}}, # Duplicates eval_0
        {"id": "tr_2", "question": "What is 2 + 2?", "answerKey": "D", "choices": {"text": ["4"], "label": ["D"]}},
    ]

    clean_train, excluded = filter_train_against_eval_duplicates(
        train_examples=train_candidates,
        eval_splits_dict={"evaluation": eval_examples},
        task_name="arc_easy"
    )

    # tr_1 must be stripped from clean_train
    assert len(clean_train) == 2
    assert [x["id"] for x in clean_train] == ["tr_0", "tr_2"]
    
    # Excluded metadata must record the exact duplicate
    assert len(excluded) == 1
    assert excluded[0]["train_id"] == "tr_1"
    assert excluded[0]["matched_id"] == "eval_0"
    assert excluded[0]["matched_split"] == "evaluation"


def test_calibration_from_permitted_training_only():
    """
    Test 2: Calibration examples come strictly from the permitted source training partition.
    They never overlap with the evaluation partition.
    """
    # Test across all tasks using mock/development mode
    for task in ["boolq", "gsm8k", "arc_easy"]:
        splits, meta = load_task_dataset(
            task_name=task,
            calib_count=128,
            val_count=100,
            eval_count=200,
            mode="development",
            allow_synthetic=True,
            require_real_huggingface_data=False
        )

        calib_inputs = {extract_canonical_input(x, task) for x in splits["calibration"]}
        eval_inputs = {extract_canonical_input(x, task) for x in splits["evaluation"]}
        train_inputs = {extract_canonical_input(x, task) for x in splits["train"]}

        # Zero overlap with evaluation
        assert len(calib_inputs.intersection(eval_inputs)) == 0
        # Calibration size exactly 128
        assert len(splits["calibration"]) == 128
        # Calibration and train are disjoint
        assert len(calib_inputs.intersection(train_inputs)) == 0


# ==============================================================================
# Rule 3: 100% Preservation of Official Benchmark Partitions
# ==============================================================================

def test_official_benchmark_partitions_preserved():
    """
    Test 3: Official evaluation partitions must preserve 100% of benchmark examples.
    Mocking the raw HF dataset return to assert the exact size is retained.
    """
    mock_train = [{"id": f"tr_{i}", "question": f"Train Q {i}", "answer": f"#### {i}"} for i in range(500)]
    mock_eval = [{"id": f"eval_{i}", "question": f"Eval Q {i}", "answer": f"#### {i}"} for i in range(1319)]

    class MockDatasetDict(dict):
        pass

    mock_hf = MockDatasetDict({"train": mock_train, "test": mock_eval})

    with patch("datasets.load_dataset", return_value=mock_hf):
        splits, meta = load_task_dataset(
            task_name="gsm8k",
            calib_count=128,
            val_count=100,
            mode="development",
            allow_synthetic=False,
            require_real_huggingface_data=True
        )

        # 100% of the 1319 test examples must be in evaluation split
        assert len(splits["evaluation"]) == 1319
        assert splits["evaluation"] == mock_eval
        # Calibration carved out of train
        assert len(splits["calibration"]) == 128
        # Validation carved out of train
        assert len(splits["validation"]) == 100
        # Remaining train
        assert len(splits["train"]) == 500 - 128 - 100


# ==============================================================================
# Rule 4 & 5: Duplicate Handling & Target Conflict Flagging
# ==============================================================================

def test_cross_split_duplicate_handling_preserves_eval():
    """
    Test 4: When duplicates occur, evaluation is 100% untouched; duplicate train example is removed.
    """
    eval_examples = [
        {"question": "What is 10 * 10?", "answer": "100 #### 100"},
        {"question": "What is 5 * 5?", "answer": "25 #### 25"},
    ]
    train_examples = [
        {"question": "What is 10 * 10?", "answer": "100 #### 100"}, # Identical duplicate
        {"question": "What is 1 * 1?", "answer": "1 #### 1"},
    ]

    clean_train, excluded = filter_train_against_eval_duplicates(
        train_examples=train_examples,
        eval_splits_dict={"test": eval_examples},
        task_name="gsm8k"
    )

    assert len(eval_examples) == 2  # Eval set untouched
    assert len(clean_train) == 1
    assert clean_train[0]["question"] == "What is 1 * 1?"
    assert len(excluded) == 1
    assert excluded[0]["is_conflicting"] is False


def test_conflicting_duplicate_targets_flagged():
    """
    Test 5: Conflicting duplicate targets (same question, different label) are flagged as conflicting.
    """
    eval_examples = [
        {"question": "Is the sky blue?", "passage": "The sky is blue on a sunny day.", "answer": True}
    ]
    train_examples = [
        # Same question and passage, but corrupted/conflicting label
        {"question": "Is the sky blue?", "passage": "The sky is blue on a sunny day.", "answer": False},
        {"question": "Is snow cold?", "passage": "Snow is made of ice crystals.", "answer": True}
    ]

    clean_train, excluded = filter_train_against_eval_duplicates(
        train_examples=train_examples,
        eval_splits_dict={"validation": eval_examples},
        task_name="boolq"
    )

    assert len(clean_train) == 1
    assert clean_train[0]["question"] == "Is snow cold?"
    assert len(excluded) == 1
    assert excluded[0]["is_conflicting"] is True
    assert excluded[0]["train_target"] == "no"
    assert excluded[0]["eval_target"] == "yes"


# ==============================================================================
# Rule 6 & 7: Commit SHA Resolution & Final Mode Rejection of Synthetic Data
# ==============================================================================

def test_actual_dataset_revisions_recorded():
    """
    Test 6: Valid 40-character hex commit SHAs are recorded, not placeholders.
    """
    for ds_name in ["google/boolq", "openai/gsm8k", "allenai/ai2_arc"]:
        sha = resolve_dataset_commit(ds_name)
        assert len(sha) == 40
        assert all(c in "0123456789abcdefABCDEF" for c in sha)
        assert sha != "verified_hf"
        assert sha == PINNED_RESOLVED_COMMITS[ds_name]


def test_final_mode_rejects_synthetic_data():
    """
    Test 7: Final mode strictly rejects synthetic data and raises RuntimeError if real data fails.
    """
    with patch("datasets.load_dataset", side_effect=Exception("HF Network Down")):
        with pytest.raises(RuntimeError, match="FINAL MODE FAILURE|FINAL MODE VIOLATION"):
            load_task_dataset(
                task_name="boolq",
                mode="final",
                allow_synthetic=False,
                require_real_huggingface_data=True
            )
