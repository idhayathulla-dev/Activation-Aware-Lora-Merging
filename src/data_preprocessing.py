"""
Data Preprocessing: Prompt formatting, label extraction, and tokenization.

Handles task-specific prompt formatting for BoolQ, GSM8K, and ARC-Easy, preserves ground-truth labels,
masks prompt tokens for causal LM loss computation, and tracks token lengths and truncation rates.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import torch


# ==============================================================================
# 1. Task-Specific Prompt Formatters and Extractors
# ==============================================================================

def format_boolq_prompt(passage: str, question: str) -> str:
    """Formats BoolQ passage and question into standard instruction prompt."""
    p_clean = passage.strip()
    q_clean = question.strip()
    if not q_clean.endswith("?"):
        q_clean += "?"
    return f"Passage: {p_clean}\nQuestion: {q_clean}\nAnswer with yes or no:\n"


def format_boolq_target(answer: bool) -> str:
    """Formats BoolQ boolean target into 'yes' or 'no'."""
    return "yes" if answer else "no"


def format_gsm8k_prompt(question: str) -> str:
    """
    Formats GSM8K grade-school math problem into reasoning prompt.
    CRITICAL: Does NOT include the solution or answer text in the prompt.
    """
    q_clean = question.strip()
    return f"Question: {q_clean}\nPlease reason step by step and conclude with the final numerical answer.\nAnswer:\n"


def extract_gsm8k_answer(raw_answer: str) -> Tuple[str, str]:
    """
    Extracts the full reasoning text and ground-truth numeric value from GSM8K answer.
    GSM8K ground-truth answer format: "... step-by-step reasoning ... #### <numeric_answer>"
    """
    parts = raw_answer.split("####")
    if len(parts) >= 2:
        reasoning = parts[0].strip()
        numeric = parts[-1].strip()
    else:
        reasoning = raw_answer.strip()
        # Fallback regex search for trailing numbers
        numbers = re.findall(r"-?\d+(?:\.\d+)?", raw_answer)
        numeric = numbers[-1] if numbers else ""
    return reasoning, normalize_numeric_answer(numeric)


def normalize_numeric_answer(raw: str) -> str:
    """
    Normalizes numeric answers by stripping whitespace, currency symbols, commas, and trailing zeroes.
    """
    cleaned = raw.strip().replace(",", "").replace("$", "").replace("%", "")
    # Remove trailing period if present
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    # Check if integer
    try:
        val = float(cleaned)
        if val.is_integer():
            return str(int(val))
        return str(val)
    except ValueError:
        return cleaned


def format_arc_prompt(question: str, choices: Dict[str, List[str]]) -> Tuple[str, str]:
    """
    Formats ARC-Easy multiple-choice science question with standard choice letters.

    Args:
        question: Question stem.
        choices: Dictionary containing 'text': List[str] and 'label': List[str].

    Returns:
        Tuple of (formatted_prompt, standard_letter_labels_dict).
    """
    q_clean = question.strip()
    choice_texts = choices.get("text", [])
    choice_labels = choices.get("label", [])

    # Standardize numeric labels 1, 2, 3, 4 to A, B, C, D if needed
    label_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
    formatted_choices = []
    
    for lbl, txt in zip(choice_labels, choice_texts):
        std_lbl = label_map.get(lbl, lbl).strip().upper()
        formatted_choices.append(f"({std_lbl}) {txt.strip()}")

    choices_block = "\n".join(formatted_choices)
    prompt = f"Question: {q_clean}\nChoices:\n{choices_block}\nAnswer:\n"
    return prompt


def normalize_arc_label(raw_label: str) -> str:
    """Standardizes ARC-Easy answer key to uppercase letter."""
    label_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
    return label_map.get(raw_label.strip(), raw_label.strip()).upper()


# ==============================================================================
# 2. Unified Example Extraction
# ==============================================================================

def extract_prompt_and_target(example: Dict[str, Any], task_name: str) -> Tuple[str, str, Any]:
    """
    Extracts prompt text, generation target, and raw ground-truth evaluation metadata.

    Args:
        example: Dictionary from raw dataset.
        task_name: 'boolq', 'gsm8k', or 'arc_easy'.

    Returns:
        Tuple of (prompt_text, target_text, raw_eval_target).
    """
    if task_name == "boolq":
        passage = example.get("passage", "")
        question = example.get("question", "")
        answer_bool = example.get("answer", False)
        prompt = format_boolq_prompt(passage, question)
        target = format_boolq_target(answer_bool)
        raw_target = answer_bool

    elif task_name == "gsm8k":
        question = example.get("question", "")
        raw_answer = example.get("answer", "")
        reasoning, numeric = extract_gsm8k_answer(raw_answer)
        prompt = format_gsm8k_prompt(question)
        # Training target includes full reasoning chain and numeric answer
        target = f"{reasoning}\nThe answer is {numeric}."
        raw_target = numeric

    elif task_name in ["arc_easy", "ai2_arc"]:
        question = example.get("question", "")
        choices = example.get("choices", {"text": [], "label": []})
        raw_key = example.get("answerKey", "A")
        prompt = format_arc_prompt(question, choices)
        target = normalize_arc_label(raw_key)
        raw_target = target

    elif "input_text" in example and "label" in example:
        prompt = str(example["input_text"])
        target = str(example["label"])
        raw_target = example["label"]

    else:
        raise ValueError(f"Unknown task_name: '{task_name}'")

    return prompt, target, raw_target


# ==============================================================================
# 3. Tokenization and Attention Mask Processing
# ==============================================================================

def preprocess_dataset(
    examples: List[Dict[str, Any]],
    task_name: str,
    tokenizer: Any,
    max_seq_length: int = 512,
    is_training: bool = True
) -> Dict[str, Any]:
    """
    Preprocesses a list of examples for causal language modeling.

    When is_training=True:
        Concatenates prompt + target, and masks prompt tokens with label -100 so loss
        is only computed on the completion tokens.

    When is_training=False (e.g., calibration or evaluation):
        Tokenizes the prompt only (for generation / input activation collection),
        while recording the ground truth target.

    Args:
        examples: List of example dictionaries.
        task_name: 'boolq', 'gsm8k', or 'arc_easy'.
        tokenizer: PreTrainedTokenizer or MockTokenizer.
        max_seq_length: Max sequence truncation length.
        is_training: Whether preprocessing for training loss or inference/calibration.

    Returns:
        Dict containing input_ids, attention_mask, labels, and statistics metadata.
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    token_lengths = []
    truncation_count = 0

    eos_token = getattr(tokenizer, "eos_token", "") or ""
    eos_token_id = getattr(tokenizer, "eos_token_id", 151643)
    pad_token_id = getattr(tokenizer, "pad_token_id", eos_token_id)

    processed_examples = []

    for ex in examples:
        prompt, target, raw_target = extract_prompt_and_target(ex, task_name)

        if is_training:
            # Full sequence: prompt + " " + target + eos_token
            full_text = f"{prompt} {target} {eos_token}".strip()
            prompt_tokens = tokenizer(prompt, truncation=False)["input_ids"]
            if isinstance(prompt_tokens[0], list):
                prompt_tokens = prompt_tokens[0]

            full_tokens = tokenizer(full_text, truncation=False)["input_ids"]
            if isinstance(full_tokens[0], list):
                full_tokens = full_tokens[0]

            seq_len = len(full_tokens)
            token_lengths.append(seq_len)
            is_truncated = seq_len > max_seq_length
            if is_truncated:
                truncation_count += 1
                full_tokens = full_tokens[:max_seq_length]

            # Label mask: -100 for prompt tokens, target tokens keep their token ids
            prompt_len = min(len(prompt_tokens), len(full_tokens))
            labels = [-100] * prompt_len + full_tokens[prompt_len:]

            all_input_ids.append(full_tokens)
            all_attention_mask.append([1] * len(full_tokens))
            all_labels.append(labels)

        else:
            # Calibration or generation input: prompt only
            prompt_tokens = tokenizer(prompt, truncation=False)["input_ids"]
            if isinstance(prompt_tokens[0], list):
                prompt_tokens = prompt_tokens[0]

            seq_len = len(prompt_tokens)
            token_lengths.append(seq_len)
            is_truncated = seq_len > max_seq_length
            if is_truncated:
                truncation_count += 1
                prompt_tokens = prompt_tokens[:max_seq_length]

            all_input_ids.append(prompt_tokens)
            all_attention_mask.append([1] * len(prompt_tokens))
            all_labels.append([-100] * len(prompt_tokens))

        processed_examples.append({
            "prompt": prompt,
            "target": target,
            "raw_target": raw_target,
            "seq_len": seq_len,
            "is_truncated": is_truncated,
        })

    # Summary statistics
    total_ex = len(examples)
    max_len = max(token_lengths) if token_lengths else 0
    mean_len = float(sum(token_lengths) / total_ex) if total_ex > 0 else 0.0
    truncation_rate = float(truncation_count / total_ex) if total_ex > 0 else 0.0

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
        "examples": processed_examples,
        "statistics": {
            "total_examples": total_ex,
            "max_token_length": max_len,
            "mean_token_length": round(mean_len, 2),
            "truncation_count": truncation_count,
            "truncation_rate": round(truncation_rate, 4),
            "max_seq_length": max_seq_length,
        }
    }
