"""
Embeddings Module
Generates embeddings using Sentence Transformers
"""

import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np


class EmbeddingGenerator:
    """Generate embeddings for document chunks"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize embedding generator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        model_name = self.config['embeddings']['model']
        self.batch_size = self.config['embeddings']['batch_size']
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
        
        Returns:
            numpy array of embeddings
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True  # For cosine similarity
        )
        print("Embeddings generated successfully")
        return embeddings
    
    def generate_embeddings_for_chunks(self, chunks: List[Dict]) -> tuple:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
        
        Returns:
            Tuple of (embeddings, chunk_texts)
        """
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.generate_embeddings(chunk_texts)
        
        return embeddings, chunk_texts


if __name__ == "__main__":
    # Test embedding generation
    generator = EmbeddingGenerator()
    test_texts = [
        "Net interest income for the quarter was $12.5 billion.",
        "Total revenue increased by 8% compared to prior year."
    ]
    embeddings = generator.generate_embeddings(test_texts)
    print(f"Generated embeddings shape: {embeddings.shape}")

