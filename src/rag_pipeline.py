"""
RAG Pipeline Module
Orchestrates retrieval and generation for question answering
"""

import yaml
from pathlib import Path
from typing import List, Dict, Optional
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from models import ModelManager


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize RAG pipeline"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(config_path)
        self.vector_store = VectorStore(config_path)
        self.model_manager = ModelManager(config_path)
    
    def _infer_metadata_filter(self, query: str) -> Optional[Dict]:
        """
        Infer bank and quarter filters from the query text.
        This helps restrict retrieval to the most relevant documents.
        """
        q_lower = query.lower()
        metadata: Dict[str, str] = {}

        # Infer bank
        if "wells fargo" in q_lower or "wells" in q_lower:
            metadata["bank"] = "Wells Fargo"
        elif "bank of america" in q_lower or "bofa" in q_lower or "boa" in q_lower:
            metadata["bank"] = "Bank of America"

        # Infer quarter
        if "q1" in q_lower or "q 1" in q_lower or "first quarter" in q_lower:
            metadata["quarter"] = "Q1"
        elif "q2" in q_lower or "q 2" in q_lower or "second quarter" in q_lower:
            metadata["quarter"] = "Q2"
        elif "q3" in q_lower or "q 3" in q_lower or "third quarter" in q_lower:
            metadata["quarter"] = "Q3"
        elif "q4" in q_lower or "q 4" in q_lower or "fourth quarter" in q_lower:
            metadata["quarter"] = "Q4"

        return metadata if metadata else None
    
    def retrieve_context(self, query: str, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User query
            filter_dict: Optional metadata filters
        
        Returns:
            List of retrieved chunks
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings([query])[0].tolist()
        
        # Query vector store
        retrieved_chunks = self.vector_store.query(
            query_text=query,
            query_embedding=query_embedding,
            filter_dict=filter_dict
        )
        
        return retrieved_chunks
    
    def generate_answer(
        self,
        query: str,
        model_name: str,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Generate answer for a query using specified model.

        If no filter_dict is provided, attempt to infer a bank/quarter
        filter from the query text to improve retrieval quality.
        """
        # Infer metadata filter if not explicitly provided
        if filter_dict is None:
            filter_dict = self._infer_metadata_filter(query)

        # Retrieve context
        retrieved_chunks = self.retrieve_context(query, filter_dict)
        
        if not retrieved_chunks:
            return {
                'query': query,
                'answer': "No relevant context found in the knowledge base.",
                'model': model_name,
                'retrieved_chunks': [],
                'context': ""
            }
        
        # Combine retrieved chunks into context
        context_parts = []
        for chunk in retrieved_chunks:
            source_info = f"[Source: {chunk['metadata'].get('bank', 'Unknown')} {chunk['metadata'].get('quarter', 'Unknown')}]"
            context_parts.append(f"{chunk['text']} {source_info}")
        
        context = "\n\n".join(context_parts)
        
        # Get model and generate response
        model = self.model_manager.get_model(model_name)
        if not model:
            return {
                'query': query,
                'answer': f"Error: Model {model_name} not found.",
                'model': model_name,
                'retrieved_chunks': retrieved_chunks,
                'context': context
            }
        
        answer = model.generate_response(query, context)
        
        return {
            'query': query,
            'answer': answer,
            'model': model_name,
            'retrieved_chunks': retrieved_chunks,
            'context': context
        }
    
    def process_questions(
        self,
        questions: List[str],
        model_names: Optional[List[str]] = None
    ) -> Dict[str, List[Dict]]:
        """
        Process multiple questions with multiple models
        
        Args:
            questions: List of question strings
            model_names: List of model display names (default: all models)
        
        Returns:
            Dictionary mapping model names to their responses
        """
        if model_names is None:
            model_names = list(self.model_manager.get_all_models().keys())
        
        results = {model_name: [] for model_name in model_names}
        
        print(f"\nProcessing {len(questions)} questions with {len(model_names)} models...")
        
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}/{len(questions)}: {question[:60]}...")
            
            for model_name in model_names:
                print(f"  Generating answer with {model_name}...")
                response = self.generate_answer(question, model_name)
                results[model_name].append(response)
        
        return results


if __name__ == "__main__":
    # Test RAG pipeline
    pipeline = RAGPipeline()
    test_query = "What was the revenue for Wells Fargo in Q1?"
    result = pipeline.generate_answer(test_query, "Llama-3")
    print(f"\nQuery: {result['query']}")
    print(f"Answer: {result['answer']}")

