import os
import re
from dataclasses import dataclass
from typing import Optional, Dict

# Try to import python-dotenv
try:
    from dotenv import load_dotenv
    # Load .env immediately if available, aligning with project habit
    load_dotenv()
except ImportError:
    pass


@dataclass
class RagConfig:
    backend: str = "faiss"  # faiss|pinecone|qdrant|chroma
    collection: str = "default"
    top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120
    # Embeddings/LLM
    embeddings_provider: Optional[str] = None  # anyroute|openai|hash
    anyroute_api_key: Optional[str] = None
    anyroute_base_url: Optional[str] = None
    anyroute_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_embeddings_model: str = "text-embedding-3-small"
    # Storage paths
    rag_dir: str = ".rag_store"


_PLACEHOLDER_PATTERNS = [
    r"^sk-your-.*-key-here$",
    r"^sk-your-openai-api-key-here$",
    r"^your-.*-api-key-here$",
    r"^your_api_key$",
    r"^api_key_here$",
    r"^<.*>$",
    r"^\[.*\]$",
    r"^\{.*\}$",
]

# Mapping of known OpenAI-compatible providers to their defaults
# This allows using project-standard keys (e.g. DEEPSEEK_API_KEY) with RAG automatically.
_COMPATIBLE_PROVIDERS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "",  # Let server decide or user override
    },
    "openrouter": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "",
    },
    # Note: Gemini and Anthropic are not strictly OpenAI-compatible for embeddings (paths differ),
    # so we do not auto-map them to AnyRoute to avoid runtime errors unless explicitly configured.
}


def _is_placeholder(value: Optional[str]) -> bool:
    if not value or not isinstance(value, str):
        return True
    v = value.strip().lower()
    if not v:
        return True
    for p in _PLACEHOLDER_PATTERNS:
        if re.match(p, v):
            return True
    # Common keywords that indicate examples
    for k in ("placeholder", "example", "sample", "demo", "insert", "replace", "change-me"):
        if k in v:
            return True
    return False


def get_default_config() -> RagConfig:
    backend = os.getenv("RAG_BACKEND", "faiss").lower()
    collection = os.getenv("RAG_COLLECTION", "default")
    rag_dir = os.getenv("RAG_DIR", ".rag_store")
    top_k = int(os.getenv("TOP_K", "5"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Embeddings provider selection
    embeddings_provider = None
    
    # 1. AnyRoute (Explicit RAG config) - Highest Priority
    anyroute_api_key = os.getenv("ANYROUTE_API_KEY")
    anyroute_base = os.getenv("ANYROUTE_BASE_URL")
    anyroute_model = os.getenv("ANYROUTE_MODEL")
    
    # 2. OpenAI (Native support)
    openai_key = os.getenv("OPENAI_API_KEY")

    # Logic to determine provider
    if (anyroute_api_key and anyroute_base) and not (_is_placeholder(anyroute_api_key) or _is_placeholder(anyroute_base)):
        embeddings_provider = "anyroute"
    elif openai_key and not _is_placeholder(openai_key):
        embeddings_provider = "openai"
    else:
        # 3. Try Auto-mapping compatible providers (DeepSeek, OpenRouter, etc.)
        for name, defaults in _COMPATIBLE_PROVIDERS.items():
            key_val = os.getenv(defaults["env_key"])
            if key_val and not _is_placeholder(key_val):
                embeddings_provider = "anyroute"
                anyroute_api_key = key_val
                # Use provider default base URL if explicit ANYROUTE_BASE_URL is missing
                anyroute_base = anyroute_base or defaults["base_url"]
                # Use provider default model if explicit ANYROUTE_MODEL is missing
                if not anyroute_model and defaults["default_model"]:
                    anyroute_model = defaults["default_model"]
                break
        
        # 4. Fallback
        if not embeddings_provider:
             embeddings_provider = "hash"  # deterministic offline fallback

    return RagConfig(
        backend=backend,
        collection=collection,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embeddings_provider=embeddings_provider,
        anyroute_api_key=None if _is_placeholder(anyroute_api_key) else anyroute_api_key,
        anyroute_base_url=None if _is_placeholder(anyroute_base) else anyroute_base,
        anyroute_model=anyroute_model,
        openai_api_key=None if _is_placeholder(openai_key) else openai_key,
        rag_dir=rag_dir,
    )
