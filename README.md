# Financial Document Q&A System

A Retrieval-Augmented Generation (RAG) system for answering questions about financial reports from Wells Fargo and Bank of America.

## Overview

This system processes financial PDF reports, creates a vectorized knowledge base, and uses three open-source LLMs (Llama-3, Mistral-7B, Phi-3-mini) to answer domain-specific financial questions. It evaluates responses using BLEU & ROUGE metrics.

## Features

- **PDF Processing**: Extracts text from financial reports using pdfplumber
- **Intelligent Chunking**: 300-token chunks with 50-token overlap for focused retrieval
- **Vector Database**: ChromaDB for efficient similarity search
- **Multiple LLMs**: Supports Llama-3, Mistral-7B, and Phi-3-mini via Ollama
- **Comprehensive Evaluation**: BLEU & ROUGE metrics
- **Formatted Output**: Beautiful tabular display of results

## Project Structure

```
financial-qa/
├── data/
│   ├── raw/              # Extracted text from PDFs
│   ├── processed/        # Chunked documents
│   └── knowledge_base/   # ChromaDB vector store
├── src/
│   ├── pdf_processor.py      # PDF text extraction
│   ├── document_processor.py  # Text cleaning and chunking
│   ├── embeddings.py          # Sentence Transformers embeddings
│   ├── vector_store.py        # ChromaDB management
│   ├── rag_pipeline.py        # RAG orchestration
│   ├── models.py              # Ollama model wrappers
│   ├── evaluator.py           # Evaluation metrics
│   └── output_formatter.py    # Table formatting
├── questions/
│   ├── financial_questions.json      # Domain-specific questions
│   └── ground_truth_answers.json    # Reference answers
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
└── main.py               # Entry point
```

## Prerequisites

1. **Python 3.9+**
2. **Ollama** installed and running
3. **Ollama Models** pulled:
   ```bash
   ollama pull llama3:8b
   ollama pull mistral:7b
   ollama pull phi3:mini
   ```

## Installation

1. Clone or navigate to the project directory:
   ```bash
   cd financial-qa
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure PDF files are in the project root:
   - Wells Fargo: `q1-earnings-wells.pdf`, `q2-earnings-wells.pdf`, `q3-earnings-wells.pdf`, `q4-earnings-wells.pdf`
   - Bank of America: `q1-earnings-bofa.pdf`, `q2_earnings_bofa.pdf`, `q3_earnings_bofa.pdf`, `q4_earnings_bofa.pdf`

## Usage

Run the main script:

```bash
python main.py
```

The system will:
1. Process PDF files and extract text
2. Chunk documents (300 tokens, 50 overlap)
3. Generate embeddings
4. Build ChromaDB vector store
5. Load questions
6. Generate answers using all three models
7. Evaluate responses
8. Display formatted results in tables

## Configuration

Edit `config.yaml` to customize:
- Model names and settings
- Chunking parameters (chunk_size, chunk_overlap)
- Vector database settings
- Evaluation metrics

## Output

The system displays:
- **3 Q&A Tables**: One for each model showing questions and answers
- **BLEU Scores Table**: BLEU scores for each model
- **ROUGE Scores Table**: ROUGE-1, ROUGE-2, ROUGE-L scores
- **METEOR Scores Table**: METEOR scores for each model

## Evaluation Metrics

- **BLEU**: Measures n-gram overlap with reference answers
- **ROUGE**: Evaluates recall-oriented metrics (ROUGE-1, ROUGE-2, ROUGE-L)
- **METEOR**: Measures semantic overlap using precision/recall with synonym and stem matching

## Notes

- The knowledge base is built once and reused. Delete `data/knowledge_base/chroma_db/` to rebuild.
- Ground truth answers should be manually updated in `questions/ground_truth_answers.json` for accurate evaluation.
- First run will take longer as it processes all PDFs and generates embeddings.

## Troubleshooting

1. **Ollama models not found**: Run `ollama pull <model_name>` for each model
2. **PDF processing errors**: Ensure PDFs are readable and not corrupted
3. **Memory issues**: Reduce batch_size in config.yaml or process fewer documents

