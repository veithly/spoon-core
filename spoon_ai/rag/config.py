import os
from dataclasses import dataclass
from typing import Optional


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


def get_default_config() -> RagConfig:
    backend = os.getenv("RAG_BACKEND", "faiss").lower()
    collection = os.getenv("RAG_COLLECTION", "default")
    rag_dir = os.getenv("RAG_DIR", ".rag_store")
    top_k = int(os.getenv("TOP_K", "5"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Embeddings provider selection: anyroute, openai, or fallback hash
    embeddings_provider = None
    anyroute_api_key = os.getenv("ANYROUTE_API_KEY")
    anyroute_base = os.getenv("ANYROUTE_BASE_URL")
    anyroute_model = os.getenv("ANYROUTE_MODEL")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anyroute_api_key and anyroute_base:
        embeddings_provider = "anyroute"
    elif openai_key:
        embeddings_provider = "openai"
    else:
        embeddings_provider = "hash"  # deterministic offline fallback

    return RagConfig(
        backend=backend,
        collection=collection,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embeddings_provider=embeddings_provider,
        anyroute_api_key=anyroute_api_key,
        anyroute_base_url=anyroute_base,
        anyroute_model=anyroute_model,
        openai_api_key=openai_key,
        rag_dir=rag_dir,
    )

