from langchain_core.documents import Document
import os
from typing import List
import uuid
from langchain_openai import OpenAIEmbeddings
class ChromaClient:
    def __init__(self, config_dir: str):
        try:
            import chromadb
        except ImportError:
            raise ImportError("Chroma is not installed. Please install it with 'pip install chromadb'.")
        
        self.client = chromadb.PersistentClient(path=os.path.join(config_dir, "spoon_ai.db"))
        self.collection = self.client.get_or_create_collection("spoon_ai")
        
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the collection"""
        embeddings = OpenAIEmbeddings()
        for doc in documents:
            # TODO: parallelize this
            doc_embedding = embeddings.embed_query(doc.page_content)
            self.collection.add(
                ids=[doc.metadata.get("id", str(uuid.uuid4()))],
                documents=[doc.page_content],
                metadatas=[doc.metadata],
                embeddings=[doc_embedding]
            )
        
    def query(self, query: str, k: int = 10) -> List[Document]:
        """Query the collection"""
        embeddings = OpenAIEmbeddings()
        query_embedding = embeddings.embed_query(query)
        results = self.collection.query(query_embedding, n_results=k)
        docs = []
        for i in range(len(results["documents"][0])):
            docs.append(Document(page_content=results["documents"][0][i], metadata=results["metadatas"][0][i]))
        return docs

    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(self.collection.name)
