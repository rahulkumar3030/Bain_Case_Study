import os
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings

from .azure_client import AzureOpenAIEmbeddingClient
load_dotenv()
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_file = os.getenv("CONFIG_YAML_PATH", "config.yaml")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)
    
class HybridRetriever:
    """Hybrid retriever with semantic search, BM25, and reranking."""
    
    def __init__(self, collection_name: str = "acme_hr_docs"):
        """
        Initialize hybrid retriever.
        
        Args:
            collection_name: ChromaDB collection name
        """
        config = load_config()
        
        self.base_dir = Path(os.getcwd())
        self.chroma_db_dir = self.base_dir / config['paths']['data_folder'] / config['paths']['chroma_db_folder']
        self.collection_name = collection_name
        
        # Config values
        self.retrieval_k = config['rag']['retrieval_k']
        self.reranker_top_k = config['rag']['reranker_top_k']
        self.semantic_weight = config['rag']['semantic_weight']
        self.bm25_weight = config['rag']['bm25_weight']
        self.reranker_model = config['rag']['reranker_model']
        
        # Initialize clients
        self.embedding_client = AzureOpenAIEmbeddingClient()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_db_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_collection(name=self.collection_name)    

if __name__ == "__main__":
    """Test retriever."""
    retriever = HybridRetriever()
    
    test_query = "How can I claim my expenses?"
    
    embs = retriever.embedding_client.create_embeddings([test_query])
    # print("embs (raw):", type(embs), len(embs))
    # print("embs[0] type/len:", type(embs[0]), len(embs[0]))
    # print("first 5 values:", embs[0][:5])
    query_embedding = embs[0]
    semantic_results = retriever.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=5  # get more because we will fuse later
                    )
    print(f"Semantic results for '{test_query}': {semantic_results}")
    # dummy_embedding = [0.0] * retriever.collection.dimension
    keyword_results = retriever.collection.query(
        query_embeddings=[query_embedding],
        query_texts=[test_query],
        n_results=3  # BM25 keyword search
    )
    print(f"Keyword results for '{test_query}': {keyword_results}")