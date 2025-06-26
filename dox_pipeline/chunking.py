# python -m dox_pipeline.chunking
from dox_pipeline.text_utils import detect_context_window

def suggest_chunk_sizes(
    model_name: str,
    chunks_per_query: int = 6,
    reserve_pct: float = 0.30,
    parent_cap: int = 2048,
    embedding_limit: int = 512,
) -> tuple[int, int, int, int]:
    """
    Compute (parent_size, child_size, parent_overlap, child_overlap).
    """
    ctx = detect_context_window(model_name)
    usable = int(ctx * (1 - reserve_pct))
    parent_size = min(usable // chunks_per_query, parent_cap)
    child_size  = min(parent_size, int(embedding_limit * 0.8), 512)
    parent_overlap = max(1, round(parent_size * 0.10))   # 10%
    child_overlap  = max(1, round(child_size  * 0.15))   # 15%
    return parent_size, child_size, parent_overlap, child_overlap
