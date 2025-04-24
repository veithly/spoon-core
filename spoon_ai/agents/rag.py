from typing import List, Optional, Dict, Any

from logging import getLogger

logger = getLogger(__name__)

DEBUG = False

def debug_log(message):
    if DEBUG:
        logger.debug(message)

class RetrievalMixin:
    """Mixin class for retrieval-augmented generation functionality"""
    
    def initialize_retrieval_client(self):
        """Initialize the retrieval client if it doesn't exist"""
        if not hasattr(self, 'retrieval_client') or self.retrieval_client is None:
            from spoon_ai.retrieval.chroma import ChromaClient
            debug_log("Initializing retrieval client")
            self.retrieval_client = ChromaClient(str(self.config_dir))
    
    def add_documents(self, documents):
        """Add documents to the retrieval system"""
        self.initialize_retrieval_client()
        
        self.retrieval_client.add_documents(documents)
        debug_log(f"Added {len(documents)} documents to retrieval system for agent {self.name}")
    
    def retrieve_relevant_documents(self, query, k=5):
        """Retrieve relevant documents for a query"""
        self.initialize_retrieval_client()
        
        try:
            docs = self.retrieval_client.query(query, k=k)
            debug_log(f"Retrieved {len(docs)} documents for query: {query}...")
            return docs
        except Exception as e:
            debug_log(f"Error retrieving documents: {e}")
            return []
    
    def get_context_from_query(self, query):
        """Get context string from relevant documents for a query"""
        relevant_docs = self.retrieve_relevant_documents(query)
        context_str = ""
        debug_log(f"Retrieved {len(relevant_docs)} relevant documents")
        
        if relevant_docs:
            context_str = "\n\nRelevant context:\n"
            for i, doc in enumerate(relevant_docs):
                context_str += f"[Document {i+1}]\n{doc.page_content}\n\n"
                
        return context_str, relevant_docs