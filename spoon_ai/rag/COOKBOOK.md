# RAG Quick Start Cookbook

This cookbook provides a quick guide to using the RAG (Retrieval-Augmented Generation) system within `spoon-core`.

## 1. Installation

Ensure you have the necessary dependencies. The RAG system supports multiple backends.

### Basic (FAISS / In-Memory)
```bash
pip install faiss-cpu  # For FAISS backend
# or no extra install for in-memory testing
```

### Advanced Backends
```bash
pip install chromadb      # For Chroma
pip install pinecone-client # For Pinecone
pip install qdrant-client # For Qdrant
```

---

## 2. Basic Usage

Here is a complete example of how to ingest documents, retrieve them, and assume a QA session.

### Step 1: Initialize Components

```python
import os
from spoon_ai.rag import (
    get_default_config,
    get_vector_store,
    get_embedding_client,
    RagIndex,
    RagRetriever,
    RagQA
)
from spoon_ai.chat import ChatBot

# Set environment variables for configuration
# os.environ["RAG_BACKEND"] = "faiss"  # Default
os.environ["OPENAI_API_KEY"] = "sk-..." 

# Load config and components
cfg = get_default_config()
store = get_vector_store(cfg.backend)
embed = get_embedding_client(cfg.embeddings_provider)
```

### Step 2: Ingest Documents

You can ingest local text files, PDFs, or URLs.

```python
# Initialize Indexer
index = RagIndex(config=cfg, store=store, embeddings=embed)

# Ingest a directory of files or a list of files/URLs
# Returns the number of chunks added
count = index.ingest(["./my_documents", "https://example.com/article"])
print(f"Ingested {count} chunks.")
```

### Step 3: Retrieve & Search

```python
# Initialize Retriever
retriever = RagRetriever(config=cfg, store=store, embeddings=embed)

# Search
chunks = retriever.retrieve("How do I use SpoonAI?", top_k=3)

for c in chunks:
    print(f"[{c.score:.2f}] {c.text[:100]}... (Source: {c.metadata.get('source')})")
```

### Step 4: Question Answering (QA)

Combine retrieval with an LLM to answer questions.

```python
# Initialize QA Engine
llm = ChatBot() # Automatically uses configured LLM provider
qa = RagQA(config=cfg, llm=llm)

# Ask
result = await qa.answer("How do I use SpoonAI?", chunks)

print("Answer:", result.answer)
print("Citations:")
for cite in result.citations:
    print(f"- {cite.marker} {cite.source}")
```

---

## 3. Configuration

You can configure the RAG system using environment variables or by passing a modified `RagConfig` object.

| Variable | Description | Default |
|----------|-------------|---------|
| `RAG_BACKEND` | Vector store backend (`faiss`, `chroma`, `pinecone`, `qdrant`) | `faiss` |
| `RAG_COLLECTION` | Name of the collection to us | `default` |
| `OPENAI_API_KEY` | Key for OpenAI Embeddings & LLM | None |
| `ANYROUTE_API_KEY` | Key for AnyRoute/DeepSeek/OpenRouter | None |
| `TOP_K` | Default number of chunks to retrieve | 5 |
| `CHUNK_SIZE` | Size of text chunks | 800 |

## 4. Using with Agents (Tools)

The RAG system exposes standard tools that can be registered with any Agent.

```python
from spoon_ai.tools.rag_tools import RAGIngestTool, RAGSearchTool, RAGQATool

tools = [
    RAGIngestTool(),
    RAGSearchTool(),
    RAGQATool(llm=chatbot) # Inject shared LLM
]

# Register tools with your agent...
```
