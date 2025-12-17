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


from spoon_ai.llm.config import ConfigurationManager

def get_default_config() -> RagConfig:
    backend = os.getenv("RAG_BACKEND", "faiss").lower()
    collection = os.getenv("RAG_COLLECTION", "default")
    rag_dir = os.getenv("RAG_DIR", ".rag_store")
    top_k = int(os.getenv("TOP_K", "5"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Use LLM ConfigurationManager for standardized provider detection
    config_manager = ConfigurationManager()
    
    # 1. Determine active provider
    # Try ANYROUTE_API_KEY explicitly first (legacy RAG priority)
    anyroute_key = os.getenv("ANYROUTE_API_KEY")
    # Use static method from ConfigurationManager
    if anyroute_key and not ConfigurationManager._is_placeholder_value(anyroute_key):
        embeddings_provider = "anyroute"
        anyroute_base = os.getenv("ANYROUTE_BASE_URL", "https://api.openai.com/v1") # Default generic
        anyroute_model = os.getenv("ANYROUTE_MODEL")
        openai_key = None
    else:
        # Fallback to LLM module's intelligent selection
        # This picks defaults based on available API keys (OpenAI > Anthropic > OpenRouter...)
        # Note: Anthropic/Gemini are not directly supported for embeddings here unless mapped
        provider = config_manager.get_default_provider()
        
        # Load full config for the selected provider
        try:
            llm_config = config_manager.load_provider_config(provider)
        except Exception:
            llm_config = None

        embeddings_provider = "hash" # Default fallback
        anyroute_key = None
        anyroute_base = None
        anyroute_model = None
        openai_key = None

        if llm_config:
            if provider == "openai":
                embeddings_provider = "openai"
                openai_key = llm_config.api_key
            elif provider in ("deepseek", "openrouter", "anyroute"):
                # Map compatible OpenAI-like providers to AnyRoute client
                embeddings_provider = "anyroute"
                anyroute_key = llm_config.api_key
                anyroute_base = llm_config.base_url
                
                # Check for explicit override or intelligent default
                env_model = os.getenv("ANYROUTE_MODEL")
                if env_model:
                    anyroute_model = env_model
                elif provider == "openrouter" and "embedding" not in llm_config.model.lower():
                    # OpenRouter: Default to openai/text-embedding-3-small if main model is not an embedding model
                    anyroute_model = "openai/text-embedding-3-small"
                else:
                    anyroute_model = llm_config.model
    
    return RagConfig(
        backend=backend,
        collection=collection,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embeddings_provider=embeddings_provider,
        anyroute_api_key=anyroute_key,
        anyroute_base_url=anyroute_base,
        anyroute_model=anyroute_model,
        openai_api_key=openai_key,
        rag_dir=rag_dir,
    )
