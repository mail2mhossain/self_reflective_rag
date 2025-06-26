# python -m dox_pipeline.text_utils
import tiktoken
from transformers import AutoConfig

from dox_pipeline.config import encoding

# Known OpenAI model context windows (tokens)
_OPENAI_MAX_TOKENS = {
    "gpt-3.5-turbo":       4096,
    "gpt-3.5-turbo-16k":  16384,
    "gpt-4":               8192,
    "gpt-4-32k":         32768,
}


# Initialize tokenizer once
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

def num_tokens(text: str) -> int:
    return len(enc.encode(text))

def detect_context_window(model_name: str, fallback: int = 4096) -> int:
    """Return the model's maximum context length (tokens)."""

    # first try our OpenAI lookup
    if model_name in _OPENAI_MAX_TOKENS:
        return _OPENAI_MAX_TOKENS[model_name]
        
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return int(getattr(cfg, "max_position_embeddings", fallback))
