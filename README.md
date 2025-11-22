<!-- b74e0786-e8aa-4f57-8910-d532dde3ad16 be22bfaf-01c2-4f9a-95e7-4422c8fb9bc1 -->
# Financial Document Q&A System - Optimized RAG Implementation

## System Architecture

**Tech Stack:**

- Python 3.9+
- Ollama (for LLM inference)
- LangChain (RAG orchestration)
- FAISS (vector store)
- Sentence Transformers (embeddings)
- RAGAS (evaluation framework)
- NLTK/rouge-score (BLEU/ROUGE metrics)
- BeautifulSoup/requests (web scraping)
- pandas/rich (tabular output)

## Implementation Plan

### 1. Project Structure

```
financial-qa/
├── data/
│   ├── raw/              # Downloaded documents
│   ├── processed/        # Chunked and cleaned text
│   └── knowledge_base/ # Final processed corpus
├── src/
│   ├── data_collection.py    # Web scraping for Yahoo Finance & SEC EDGAR
│   ├── document_processor.py # Text cleaning, chunking
│   ├── embeddings.py         # Sentence Transformers setup
│   ├── vector_store.py       # FAISS index creation
│   ├── rag_pipeline.py       # Main RAG orchestration
│   ├── models.py             # Ollama model wrappers
│   ├── evaluator.py          # RAGAS, BLEU, ROUGE evaluation
│   └── output_formatter.py   # Tabular console output
├── questions/
│   └── financial_questions.json  # 10+ domain-specific questions
├── config.yaml            # Configuration (model names, paths, etc.)
├── requirements.txt
└── main.py               # Entry point
```

### 2. Data Collection & Knowledge Base Creation

**File: `src/data_collection.py`**

- SEC EDGAR scraper: Download 10-K filings using `sec-edgar-downloader` or direct API
- Yahoo Finance scraper: Extract news articles from provided URLs
- Text extraction: Parse HTML/PDF to plain text
- Storage: Save raw documents to `data/raw/`

**File: `src/document_processor.py`**

- Text cleaning: Remove headers, footers, special characters
- Chunking strategy: 
  - Overlapping chunks (512 tokens, 50 token overlap)
  - Preserve document metadata (source, date, company)
- Tokenization: Prepare for embedding
- Output: Processed chunks to `data/processed/`

### 3. RAG Pipeline Implementation

**File: `src/embeddings.py`**

- Model: `sentence-transformers/all-MiniLM-L6-v2` (or `all-mpnet-base-v2` for better quality)
- Generate embeddings for all document chunks
- Store embeddings for FAISS indexing

**File: `src/vector_store.py`**

- FAISS index creation (L2 distance or cosine similarity)
- Save/load index functionality
- Retrieval: Top-k (k=5) most relevant chunks per query

**File: `src/rag_pipeline.py`**

- Query processing: Reformulate if needed
- Retrieval: Get top-k chunks from FAISS
- Context assembly: Combine retrieved chunks with query
- Prompt template: Financial domain-specific prompts
- Response generation: Pass to LLM via Ollama

### 4. Model Integration (Ollama)

**File: `src/models.py`**

- Ollama client wrapper for three models:
  - `llama3` (or `llama3:8b`)
  - `mistral:7b`
  - `phi3:mini`
- Unified interface: `generate_response(query, context)`
- Error handling and retry logic
- Temperature/sampling parameters per model

**Setup requirement:** Ensure Ollama is installed and models are pulled:

```bash
ollama pull llama3:8b
ollama pull mistral:7b
ollama pull phi3:mini
```

### 5. Evaluation Framework

**File: `src/evaluator.py`**

- **RAGAS metrics:**
  - Faithfulness (factual consistency)
  - Context Precision
  - Context Recall
  - Answer Relevancy
- **BLEU score:** Compare generated answers (requires reference answers)
- **ROUGE scores:** ROUGE-L, ROUGE-1, ROUGE-2
- Ground truth: Create reference answers for 10 questions (manual or from authoritative sources)

**Note:** For RAGAS, we'll need ground truth answers. Plan includes creating a reference answers file.

### 6. Output Formatting

**File: `src/output_formatter.py`**

- Use `rich` library for formatted tables
- **Table 1:** Llama-3 Q&A (10 rows: Question | Answer)
- **Table 2:** Mistral-7B Q&A (10 rows: Question | Answer)
- **Table 3:** Phi-3-mini Q&A (10 rows: Question | Answer)
- **Table 4:** RAGAS Metrics (Model | Faithfulness | Context Precision | Context Recall | Answer Relevancy)
- **Table 5:** BLEU Scores (Model | BLEU Score)
- **Table 6:** ROUGE Scores (Model | ROUGE-1 | ROUGE-2 | ROUGE-L)

### 7. Domain-Specific Questions

**File: `questions/financial_questions.json`**

Create 10+ questions covering:

- Revenue trends and financial performance
- Risk factors and disclosures
- Market analysis and competitive positioning
- Regulatory compliance
- Strategic initiatives
- Financial ratios and metrics

### 8. Main Execution Flow

**File: `main.py`**

1. Initialize data collection (if needed)
2. Process documents and create knowledge base
3. Build FAISS vector store
4. Load questions from JSON
5. For each model:

   - Run RAG pipeline for all questions
   - Store responses

6. Evaluate all responses (RAGAS, BLEU, ROUGE)
7. Format and display results in tables

### 9. Configuration

**File: `config.yaml`**

- Model names (Ollama)
- Embedding model
- Chunk size and overlap
- Top-k retrieval
- Evaluation settings
- Data paths

## Key Optimizations

1. **Efficient chunking:** Overlapping chunks preserve context across boundaries
2. **Metadata preservation:** Track document source for citation in answers
3. **Query reformulation:** Enhance queries for better retrieval
4. **Batch processing:** Process multiple questions efficiently
5. **Caching:** Cache embeddings and vector store to avoid recomputation
6. **Error handling:** Robust handling for API failures, missing documents

## Dependencies

- `ollama` (Python client)
- `langchain`, `langchain-community`
- `faiss-cpu` (or `faiss-gpu`)
- `sentence-transformers`
- `ragas`
- `nltk`, `rouge-score`
- `beautifulsoup4`, `requests`
- `sec-edgar-downloader` (optional)
- `rich`, `pandas`
- `pyyaml`

## Evaluation Strategy

- Create ground truth answers for 10 questions (manual annotation or authoritative sources)
- Run RAGAS evaluation with retrieved contexts
- Compute BLEU/ROUGE against ground truth
- Compare metrics across models to identify best performer

### To-dos

- [ ] Create project structure, requirements.txt, and config.yaml with all necessary dependencies and configuration
- [ ] Implement web scrapers for SEC EDGAR and Yahoo Finance URLs to collect financial documents
- [ ] Build document processor for text cleaning, chunking (512 tokens, 50 overlap), and metadata preservation
- [ ] Implement Sentence Transformers embeddings and FAISS vector store creation with save/load functionality
- [ ] Create Ollama client wrapper for three models (Llama-3, Mistral-7B, Phi-3-mini) with unified interface
- [ ] Build RAG pipeline with query processing, FAISS retrieval (top-k=5), context assembly, and prompt templates
- [ ] Create 10+ domain-specific financial questions covering revenue trends, risk factors, market analysis, etc.
- [ ] Implement RAGAS (faithfulness, context precision/recall, answer relevancy), BLEU, and ROUGE evaluation metrics
- [ ] Create tabular output formatter using rich library for Q&A tables and metrics tables
- [ ] Build main.py to orchestrate data collection, RAG execution across models, evaluation, and formatted output
