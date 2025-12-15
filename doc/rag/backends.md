Backends & Embeddings
=====================

Vector Stores
-------------

The RAG registry supports the following backends via `RAG_BACKEND`:

- `faiss` (default): Local and offline friendly. Provides a real FAISS adapter (falls back to in-memory cosine retrieval if `faiss` is not installed).
- `pinecone`: Cloud vector database (requires `PINECONE_API_KEY`, default index name `spoon-rag`, can be specified via `RAG_PINECONE_INDEX`). Real adapter provided.
- `qdrant`: Local/Cloud (requires `pip install qdrant-client` and service at `http://localhost:6333` by default). Real adapter provided.
  - Supports embedded mode: `QDRANT_URL=:memory:` or set `QDRANT_PATH=:memory:` to run locally without a service.
- `chroma`: Local (requires `pip install chromadb`, default persistence directory `${RAG_DIR:-.rag_store}/chroma`). Real adapter provided.

Note: To ensure stable offline testing, `faiss` still defaults to in-memory vector store behavior. Other backends require installation/configuration as needed, while the adapter layer maintains a unified API.

Embeddings
----------

- AnyRoute (OpenAI-compatible): set `ANYROUTE_API_KEY` and `ANYROUTE_BASE_URL` (optional `ANYROUTE_MODEL`).
- OpenAI: set `OPENAI_API_KEY` (uses `text-embedding-3-small` by default).
- Hash (fallback): deterministic offline embedding for tests and demos (no env needed).

Backend Smoke Tests
-------------------

- FAISS:
  ```bash
  # Automatically falls back to in-memory implementation if faiss is not installed
  RAG_BACKEND=faiss RAG_FAKE_QA=1 python examples/smoke/rag_faiss_smoke.py
  ```

- Pinecone:
  ```bash
  export PINECONE_API_KEY=...
  RAG_BACKEND=pinecone RAG_FAKE_QA=1 python examples/smoke/rag_pinecone_smoke.py
  ```

- Qdrant (requires local/remote service and `qdrant-client`)
  ```bash
  pip install qdrant-client
  export QDRANT_URL=http://localhost:6333  # if needed
  RAG_BACKEND=qdrant RAG_FAKE_QA=1 python examples/smoke/rag_qdrant_smoke.py
  ```

- Chroma (requires `chromadb`)
  ```bash
  pip install chromadb
  RAG_BACKEND=chroma RAG_FAKE_QA=1 python examples/smoke/rag_chroma_smoke.py
  ```
