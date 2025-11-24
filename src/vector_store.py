"""
Vector Store Module
Manages ChromaDB vector database for document storage and retrieval
"""

import yaml
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional, Any
import json


class VectorStore:
    """ChromaDB vector store for financial documents"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize ChromaDB vector store"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path(__file__).parent.parent
        self.collection_name = self.config['vector_db']['collection_name']
        self.persist_directory = self.project_root / self.config['vector_db']['persist_directory']
        self.top_k = self.config['vector_db']['top_k']
        
        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        print(f"ChromaDB initialized. Collection: {self.collection_name}")
        print(f"Persist directory: {self.persist_directory}")
    
    def add_documents(self, chunks: List[Dict], embeddings: List[List[float]]):
        """
        Add documents to the vector store
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors
        """
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        ids = [chunk['chunk_id'] for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Successfully added {len(chunks)} chunks to vector store")
    
    def query(
        self,
        query_text: str,
        query_embedding: List[float],
        n_results: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Query the vector store
        
        Args:
            query_text: Query text string
            query_embedding: Query embedding vector
            n_results: Number of results to return (default: top_k from config)
            filter_dict: Optional metadata filters (e.g., {"bank": "Wells Fargo"})
        
        Returns:
            List of retrieved chunks with metadata
        """
        n_results = n_results or self.top_k
        
        # Build Chroma where-clause from simple metadata filters.
        # Chroma expects a single top-level operator (e.g. "$and").
        where_clause = None
        if filter_dict:
            if len(filter_dict) == 1:
                # Single field filter, pass through directly, e.g. {"bank": "Wells Fargo"}
                where_clause = filter_dict
            else:
                # Multiple fields: wrap in an $and operator
                and_clauses = []
                for key, value in filter_dict.items():
                    and_clauses.append({key: value})
                where_clause = {"$and": and_clauses}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause
        )
        
        # Format results
        retrieved_chunks = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                retrieved_chunks.append({
                    'chunk_id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'similarity': 1 - results['distances'][0][i] if 'distances' in results else None
                })
        
        return retrieved_chunks
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'persist_directory': str(self.persist_directory)
        }
    
    def delete_collection(self):
        """Delete the collection (use with caution)"""
        self.client.delete_collection(name=self.collection_name)
        print(f"Collection {self.collection_name} deleted")


if __name__ == "__main__":
    # Test vector store
    store = VectorStore()
    stats = store.get_collection_stats()
    print(f"Collection stats: {stats}")

