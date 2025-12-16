"""Graph Agent demo using the RAG component (ingest -> retrieve -> answer).

Offline by default (hash embeddings + in-memory store).
"""

import asyncio
from typing import Dict, Any, List

from spoon_ai.graph import StateGraph
from spoon_ai.agents.graph_agent import GraphAgent
from spoon_ai.chat import ChatBot

from spoon_ai.rag import (
    get_default_config,
    get_embedding_client,
    get_vector_store,
    RagIndex,
    RagRetriever,
    RagQA,
)


def build_pipeline():
    cfg = get_default_config()
    store = get_vector_store(cfg.backend)
    embed = get_embedding_client(
        cfg.embeddings_provider,
        openai_api_key=cfg.openai_api_key,
        openai_model=cfg.openai_embeddings_model,
        anyroute_api_key=cfg.anyroute_api_key,
        anyroute_base_url=cfg.anyroute_base_url,
        anyroute_model=cfg.anyroute_model,
    )

    index = RagIndex(config=cfg, store=store, embeddings=embed)
    retriever = RagRetriever(config=cfg, store=store, embeddings=embed)
    # Offline-friendly: if RAG_FAKE_QA=1, avoid initializing ChatBot
    import os
    llm = None if os.getenv("RAG_FAKE_QA") == "1" else ChatBot()
    qa = RagQA(config=cfg, llm=llm)

    def ingest_node(state: Dict[str, Any]) -> Dict[str, Any]:
        inputs = state.get("inputs", [])
        n = index.ingest(inputs)
        return {"ingested": n}

    def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("question", "")
        chunks = retriever.retrieve(q)
        return {"chunks": chunks, "context": retriever.build_context(chunks)}

    async def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("question", "")
        chunks = state.get("chunks", [])
        res = await qa.answer(q, chunks)
        return {"answer": res.answer, "citations": res.citations}

    graph = StateGraph(dict)
    graph.add_node("RAGIngestNode", ingest_node)
    graph.add_node("RAGRetrieveNode", retrieve_node)
    graph.add_node("RAGAnswerNode", answer_node)
    graph.add_edge("RAGIngestNode", "RAGRetrieveNode")
    graph.add_edge("RAGRetrieveNode", "RAGAnswerNode")
    graph.set_entry_point("RAGIngestNode")
    return graph.compile()


async def main():
    compiled = build_pipeline()
    agent = GraphAgent(name="rag-graph", graph=compiled.graph)
    initial_state = {
        "inputs": ["./doc"],
        "question": "How do I install the SDK?",
    }
    result = await compiled.ainvoke(initial_state)
    print("\n== RAG Graph Agent Demo ==")
    print("Answer:", result.get("answer"))
    print("Citations:", result.get("citations"))


if __name__ == "__main__":
    asyncio.run(main())
