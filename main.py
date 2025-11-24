"""
Main Execution Script
Orchestrates the entire Financial RAG pipeline
"""

import json
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_processor import PDFProcessor
from document_processor import DocumentProcessor
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from evaluator import Evaluator
from output_formatter import OutputFormatter


def load_config():
    """Load configuration"""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)


def build_knowledge_base(config, rebuild=False):
    """
    Build the knowledge base from PDFs
    
    Args:
        config: Configuration dictionary
        rebuild: Whether to rebuild even if knowledge base exists
    """
    kb_path = Path(config['paths']['knowledge_base']) / "chroma_db"
    
    # Check if knowledge base already exists
    if kb_path.exists() and not rebuild:
        print("Knowledge base already exists. Use rebuild=True to rebuild.")
        return
    
    print("="*80)
    print("BUILDING KNOWLEDGE BASE")
    print("="*80)
    
    # Step 1: Process PDFs
    print("\n[Step 1/4] Processing PDF files...")
    pdf_processor = PDFProcessor()
    documents = pdf_processor.process_all_pdfs()
    
    # Step 2: Process documents into chunks
    print("\n[Step 2/4] Chunking documents...")
    doc_processor = DocumentProcessor()
    chunks = doc_processor.process_documents(documents)
    
    # Step 3: Generate embeddings
    print("\n[Step 3/4] Generating embeddings...")
    embedding_generator = EmbeddingGenerator()
    embeddings, chunk_texts = embedding_generator.generate_embeddings_for_chunks(chunks)
    
    # Step 4: Build vector store
    print("\n[Step 4/4] Building vector store...")
    vector_store = VectorStore()
    vector_store.add_documents(chunks, embeddings.tolist())
    
    stats = vector_store.get_collection_stats()
    print(f"\n✓ Knowledge base built successfully!")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Collection: {stats['collection_name']}")


def load_questions(config):
    """Load questions from JSON file"""
    questions_path = Path(config['paths']['questions'])
    
    if not questions_path.exists():
        print(f"Error: Questions file not found at {questions_path}")
        return []
    
    with open(questions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('questions', [])


def load_ground_truth(config):
    """Load ground truth answers"""
    gt_path = Path(config['evaluation']['ground_truth_file'])
    
    if not gt_path.exists():
        print(f"Warning: Ground truth file not found at {gt_path}")
        return {}
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main execution function"""
    print("="*80)
    print("FINANCIAL DOCUMENT Q&A SYSTEM")
    print("="*80)
    
    # Load configuration
    config = load_config()
    
    # Build knowledge base
    build_knowledge_base(config, rebuild=False)
    
    # Load questions
    print("\n" + "="*80)
    print("LOADING QUESTIONS")
    print("="*80)
    questions = load_questions(config)
    
    if not questions:
        print("No questions found. Exiting.")
        return
    
    print(f"Loaded {len(questions)} questions")
    
    # Limit to first 10 questions if more exist
    if len(questions) > 10:
        print(f"Using first 10 questions for evaluation")
        questions = questions[:10]
    
    # Initialize RAG pipeline
    print("\n" + "="*80)
    print("INITIALIZING RAG PIPELINE")
    print("="*80)
    pipeline = RAGPipeline()
    
    # Process questions with all models
    print("\n" + "="*80)
    print("GENERATING ANSWERS")
    print("="*80)
    results = pipeline.process_questions(questions)
    
    # Evaluate results
    print("\n" + "="*80)
    print("EVALUATING RESULTS")
    print("="*80)
    ground_truth = load_ground_truth(config)
    evaluator = Evaluator()
    evaluation_results = evaluator.evaluate_all(results, questions, ground_truth)
    
    # Display results
    print("\n" + "="*80)
    print("DISPLAYING RESULTS")
    print("="*80)
    formatter = OutputFormatter()
    formatter.display_all_results(results, evaluation_results)
    
    print("\n" + "="*80)
    print("PROCESS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

