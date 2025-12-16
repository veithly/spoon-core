from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from .base import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, *, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or os.getenv("CHROMA_DIR", os.path.join(os.getenv("RAG_DIR", ".rag_store"), "chroma"))
        self._client = None
        self._collections = {}

    def _client_or_raise(self):
        if self._client is None:
            try:
                import chromadb
            except Exception as e:
                raise RuntimeError("chromadb package is required for Chroma backend") from e
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def _get_collection(self, name: str):
        if name in self._collections:
            return self._collections[name]
        client = self._client_or_raise()
        col = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        self._collections[name] = col
        return col

    def add(self, *, collection: str, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]) -> None:
        col = self._get_collection(collection)
        col.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query(self, *, collection: str, query_embeddings: List[List[float]], top_k: int = 5, filter: Optional[Dict] = None) -> List[List[Tuple[str, float, Dict]]]:
        col = self._get_collection(collection)
        # Chroma >=1.3 disallows requesting "ids" in include; request metadatas+distances only.
        res = col.query(query_embeddings=query_embeddings, n_results=top_k, include=["metadatas", "distances"])
        out: List[List[Tuple[str, float, Dict]]] = []
        q = len(query_embeddings)
        for i in range(q):
            mds = res.get("metadatas", [[]])[i]
            dists = res.get("distances", [[]])[i]
            triples: List[Tuple[str, float, Dict]] = []
            # Some Chroma versions return ids even if not requested; otherwise synthesize
            ids = res.get("ids", [[]])
            ids_i = ids[i] if i < len(ids) else []
            for j, (dist, md) in enumerate(zip(dists, mds)):
                # Convert distance to a similarity-like score; keep ordering
                try:
                    d = float(dist)
                except Exception:
                    d = 0.0
                score = 1.0 / (1.0 + max(d, 0.0))
                fallback_id = f"chroma-{i}-{j}"
                id_ = ids_i[j] if j < len(ids_i) else (md.get("id") if isinstance(md, dict) and md.get("id") else fallback_id)
                triples.append((id_, score, md or {}))
            out.append(triples)
        return out

    def delete_collection(self, collection: str) -> None:
        try:
            client = self._client_or_raise()
            client.delete_collection(collection)
            self._collections.pop(collection, None)
        except Exception:
            pass
