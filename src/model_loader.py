"""
Model Loader: Initializes and audits base models and tokenizers.

Provides robust loading for Qwen/Qwen2.5-0.5B-Instruct, records model architecture metadata,
supports CPU and CUDA devices, and verifies forward pass execution.
"""

import time
from typing import Any, Dict, Optional, Tuple
import torch


class MockQwenConfig:
    """Minimal mock configuration representing Qwen2.5-0.5B-Instruct architecture."""
    def __init__(self):
        self.model_type = "qwen2"
        self.hidden_size = 896
        self.intermediate_size = 4864
        self.num_hidden_layers = 24
        self.num_attention_heads = 14
        self.num_key_value_heads = 2
        self.vocab_size = 151936
        self.torch_dtype = "float32"
        self._name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"


class MockQwenModel(torch.nn.Module):
    """
    Lightweight CPU mock causal language model conforming to Qwen2ForCausalLM interface.
    Used for local CPU smoke testing and unit tests without downloading large weights.
    """
    def __init__(self, config: Optional[MockQwenConfig] = None):
        super().__init__()
        self.config = config or MockQwenConfig()
        self.embed_tokens = torch.nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        # Target modules q_proj and v_proj as in Qwen2 attention
        self.q_proj = torch.nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=True)
        self.k_proj = torch.nn.Linear(self.config.hidden_size, self.config.hidden_size // 7, bias=True)
        self.v_proj = torch.nn.Linear(self.config.hidden_size, self.config.hidden_size // 7, bias=True)
        self.o_proj = torch.nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
        self.lm_head = torch.nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        batch_size, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)
        # Pass through q_proj and v_proj
        q = self.q_proj(x)
        v = self.v_proj(x)
        v_expanded = v.repeat(1, 1, 7) if v.size(-1) != q.size(-1) else v
        logits = self.lm_head(q + v_expanded)
        
        loss = None
        if labels is not None:
            # Shift labels for causal LM loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        class ModelOutput:
            def __init__(self, logits, loss):
                self.logits = logits
                self.loss = loss
        return ModelOutput(logits=logits, loss=loss)


class MockTokenizer:
    """Mock tokenizer matching Qwen2 tokenizer behavior for offline testing."""
    def __init__(self):
        self.vocab_size = 151936
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token_id = 151643
        self.eos_token_id = 151643
        self.model_max_length = 32768

    def __call__(
        self,
        text,
        max_length: int = 512,
        padding: Any = False,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs
    ):
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts = text
            single = False

        batch_input_ids = []
        batch_attention_mask = []

        for t in texts:
            # Deterministic pseudo-tokens from string hash / words
            words = t.split()
            token_ids = [abs(hash(w)) % (self.vocab_size - 100) + 1 for w in words]
            if not token_ids:
                token_ids = [self.eos_token_id]
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            batch_input_ids.append(token_ids)

        if padding:
            max_len = max(len(ids) for ids in batch_input_ids)
            if padding == "max_length" or padding is True:
                max_len = max(max_len, max_length if padding == "max_length" else max_len)
            padded_ids = []
            masks = []
            for ids in batch_input_ids:
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [self.pad_token_id] * pad_len)
                masks.append([1] * len(ids) + [0] * pad_len)
            batch_input_ids = padded_ids
            batch_attention_mask = masks
        else:
            batch_attention_mask = [[1] * len(ids) for ids in batch_input_ids]

        if return_tensors == "pt":
            input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
            attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
            if single and not padding:
                pass
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask}

    def decode(self, token_ids, skip_special_tokens: bool = True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join(f"token_{tid}" for tid in token_ids if tid != self.pad_token_id)


def load_model_and_tokenizer(
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
    revision: str = "main",
    torch_dtype: Optional[str] = None,
    device: Optional[str] = None,
    use_mock_fallback: bool = True,
    prefer_mock: bool = False,
    local_files_only: bool = False
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Loads base model and tokenizer, supporting CPU, CUDA, and offline mock fallback.

    Args:
        model_name_or_path: Hugging Face model identifier or local checkpoint path.
        revision: Model commit hash or revision branch.
        torch_dtype: Desired precision ("float32", "bfloat16", "float16"). Defaults to auto.
        device: "cpu", "cuda", or auto-detect.
        use_mock_fallback: Fallback to MockQwenModel if internet or HF checkpoint is unreachable.
        prefer_mock: Directly use MockQwenModel (avoids remote HF network timeouts during smoke tests).
        local_files_only: Only attempt loading from local cache without contacting huggingface.co.

    Returns:
        Tuple of (model, tokenizer, metadata_dict).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch_dtype is None:
        selected_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    elif torch_dtype == "bfloat16":
        selected_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    elif torch_dtype == "float16":
        selected_dtype = torch.float16 if device == "cuda" else torch.float32
    else:
        selected_dtype = torch.float32

    model = None
    tokenizer = None
    is_mock = False

    if prefer_mock:
        tokenizer = MockTokenizer()
        model = MockQwenModel()
        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        is_mock = True
    else:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[ModelLoader] Attempting to load '{model_name_or_path}' (rev: {revision})...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                revision=revision,
                trust_remote_code=True,
                local_files_only=local_files_only
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                revision=revision,
                torch_dtype=selected_dtype,
                device_map=device if device == "cuda" else None,
                trust_remote_code=True,
                local_files_only=local_files_only
            )
            if device == "cpu":
                model = model.to("cpu")

        except Exception as e:
            if use_mock_fallback:
                print(f"[ModelLoader] HF model load failed ({e}). Initializing offline MockQwenModel for testing...")
                tokenizer = MockTokenizer()
                model = MockQwenModel()
                if device == "cuda" and torch.cuda.is_available():
                    model = model.cuda()
                is_mock = True
            else:
                raise RuntimeError(f"Failed to load model '{model_name_or_path}': {e}")

    # Record model and tokenizer audit metadata
    cfg = getattr(model, "config", MockQwenConfig())
    metadata = {
        "model_name_or_path": model_name_or_path,
        "revision": revision,
        "architecture": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "hidden_size": getattr(cfg, "hidden_size", 896),
        "intermediate_size": getattr(cfg, "intermediate_size", 4864),
        "num_layers": getattr(cfg, "num_hidden_layers", 24),
        "num_attention_heads": getattr(cfg, "num_attention_heads", 14),
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", 2),
        "vocab_size": getattr(cfg, "vocab_size", 151936),
        "target_modules": ["q_proj", "v_proj"],
        "device": str(device),
        "torch_dtype": str(selected_dtype),
        "is_mock": is_mock,
    }

    return model, tokenizer, metadata


def verify_forward_pass(
    model: Any,
    tokenizer: Any,
    test_text: str = "Question: What is 2 + 2? Answer: 4"
) -> Dict[str, Any]:
    """
    Verifies that the model can successfully execute a forward pass without NaN or Inf values.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer instance.
        test_text: Input prompt text for forward verification.

    Returns:
        Dictionary of verification diagnostics (logits shape, finiteness, forward latency).
    """
    model.eval()
    inputs = tokenizer(test_text, return_tensors="pt")
    
    # Move inputs to model device
    param = next(model.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    logits = outputs.logits
    is_finite = bool(torch.all(torch.isfinite(logits)).item())
    logits_shape = list(logits.shape)

    diagnostics = {
        "verified": is_finite,
        "input_tokens": int(input_ids.shape[1]),
        "logits_shape": logits_shape,
        "is_finite": is_finite,
        "latency_ms": round(latency_ms, 2),
        "device": str(device),
    }

    if not is_finite:
        raise ValueError("Forward pass produced NaN or infinite logits.")

    return diagnostics
