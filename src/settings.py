import os
from typing import Mapping, Optional


def embedding_settings(env: Optional[Mapping[str, str]] = None) -> tuple[str, str]:
    """(mode, local model name) shared by SOP ingestion and retrieval so both hit the same index."""
    env = os.environ if env is None else env
    mode = (env.get("EMBEDDING_MODEL") or "LOCAL").strip().upper()
    local_model = (env.get("LOCAL_EMBEDDING_MODEL") or "BAAI/bge-m3").strip()
    return mode, local_model
