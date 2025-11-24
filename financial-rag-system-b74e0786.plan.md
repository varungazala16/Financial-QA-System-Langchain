<!-- b74e0786-e8aa-4f57-8910-d532dde3ad16 35584450-cb92-4775-bdda-27f53f5f5810 -->
# Financial Document Q&A System - Optimized RAG Implementation

## System Architecture

**Tech Stack:**

- Python 3.9+
- Ollama (for LLM inference)
- LangChain (RAG orchestration)
- ChromaDB (vector database)
- Sentence Transformers (embeddings)
- pdfplumber (PDF text extraction)
- RAGAS (evaluation framework)
- NLTK/rouge-score (BLEU/ROUGE metrics)
- pandas/rich (tabular output)
- tiktoken (token counting)

## Implementation Plan

### 1. Project Structure

```
financial-qa/
├── data/
│   ├── raw/              # Downloaded documents
│   ├── processed/        # Chunked and cleaned text
│   └── knowledge_base/ # Final processed corpus
├── src/
│   ├── pdf_processor.py      # PDF text extraction using pdfplumber
│   ├── document_processor.py # Text cleaning, chunking
│   ├── embeddings.py         # Sentence Transformers setup
│   ├── vector_store.py       # ChromaDB vector database
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

**Document Structure:**

- Wells Fargo: Q1, Q2, Q3, Q4 financial reports (4 PDFs)
        - `q1-2024-earnings-wells.pdf`
        - `q2-2024-earnings-wells.pdf`
        - `q3-2024-earnings-wells.pdf`
        - `q4-2024-earnings-wells.pdf`
- Bank of America: Q1, Q2, Q3, Q4 financial reports (4 PDFs)
        - `q1-earnings-bofa.pdf`
        - `q2_earnings_bofa.pdf`
        - `q3_earnings_bofa.pdf`
        - `q4_earnings_bofa.pdf`
- Total: 8 PDF documents (already available in project folder)

**File: `src/pdf_processor.py`**

- **PDF Text Extraction:** Use `pdfplumber` library
        - Primary choice: `pdfplumber` - excellent for financial documents
                - Preserves table structure and formatting
                - Handles multi-column layouts well
                - Extracts text with positional information
                - Better table extraction than PyPDF2
        - Alternative fallback: `pymupdf` (fitz) for complex layouts
- Process all 8 PDFs from project root directory
- Extract text, tables, and metadata from each PDF
- Preserve document structure (sections, headings)
- Handle table extraction: Convert PDF tables to structured text format
- Store raw extracted text to `data/raw/` with naming: `{bank}_{quarter}.txt`
- Extract metadata: bank name, quarter, year, document type

**File: `src/document_processor.py`**

- **Text Cleaning:**
        - Remove excessive whitespace and line breaks
        - Normalize special characters (currency symbols, percentages)
        - Preserve numerical data (financial figures, dates)
        - Handle tables: Convert to structured text or preserve as-is
        - Remove headers/footers (if identifiable patterns exist)

- **Chunking Strategy (Detailed):**

**Primary Method: Semantic Chunking with Overlap**

        - **Chunk Size:** 100 tokens (focused, topic-specific chunks)
                - Reason: Smaller chunks improve Information Retrieval (IR) efficiency
                - Each chunk focuses on a specific topic or concept
                - Better precision in retrieval - reduces noise from unrelated content
                - More granular matching to user queries
                - Financial documents benefit from focused chunks that capture specific metrics or statements
        - **Chunk Overlap:** 50 tokens (50% overlap)
                - Reason: High overlap ensures no information loss at boundaries
                - Critical for financial data where context spans multiple sentences
                - Prevents splitting related financial figures across chunks
                - Ensures continuity for multi-sentence financial statements
        - **Tokenizer:** Use `tiktoken` (GPT-4 tokenizer) for accurate token counting

**Secondary Method: Section-Aware Chunking**

        - Detect document sections (using headings, formatting)
        - Preserve section boundaries when possible
        - Chunk within sections, but allow overlap across section boundaries
        - Sections to identify: Income Statement, Balance Sheet, Cash Flow, Risk Factors, MD&A

**Metadata Preservation:**

        - Bank name (Wells Fargo / Bank of America)
        - Quarter (Q1, Q2, Q3, Q4)
        - Document section (if identifiable)
        - Page number (approximate)
        - Chunk index within document
        - Source file path

- **Chunking Parameters Explained:**
  ```
  chunk_size = 100 tokens
 - Focused chunks improve IR efficiency by targeting specific topics
 - Each chunk contains a focused concept, metric, or statement
 - Better precision: Retrieval returns more relevant, topic-specific content
 - Reduces noise: Smaller chunks avoid mixing unrelated information
 - Optimal for financial Q&A where queries target specific metrics or facts
  
  chunk_overlap = 50 tokens
 - 50% overlap ensures comprehensive coverage
 - Prevents splitting related financial figures across chunk boundaries
 - High overlap compensates for small chunk size
 - Ensures continuity for multi-sentence financial statements
 - Critical for maintaining context in focused chunks
  
  max_chunks_per_doc = None (process all)
 - Financial reports are dense; all content is potentially relevant
  
  separator = "\n\n" (paragraph breaks)
 - Preserves document structure
 - Natural break points for financial data
  ```

- **Special Handling:**
        - **Tables:** Extract and format as text with clear delimiters
        - **Financial Figures:** Preserve exact formatting (dollars, percentages)
        - **Dates:** Normalize to consistent format
        - **Acronyms:** Expand common financial terms (EBITDA, ROE, etc.) in metadata

- **Output:**
        - Processed chunks to `data/processed/chunks.jsonl` (JSON Lines format)
        - Each line: `{"text": "...", "metadata": {...}, "chunk_id": "..."}`
        - Create chunk index mapping: `data/processed/chunk_index.json`

### 3. RAG Pipeline Implementation

**File: `src/embeddings.py`**

- Model: `sentence-transformers/all-mpnet-base-v2` (768-dim embeddings)
- Batch embedding generation (batch_size=32)
- Progress tracking for large document sets
- Embedding normalization for cosine similarity
- Generate embeddings for all document chunks before storing in ChromaDB

**File: `src/vector_store.py`**

- **Vector Database: ChromaDB** (local, persistent, easy setup)
  - Alternative options: Qdrant (if cloud needed), Weaviate (if more features needed)
  - ChromaDB chosen for: simplicity, local persistence, metadata filtering, free/open-source
- Initialize ChromaDB collection: `financial_reports`
- Store embeddings with metadata (bank, quarter, section, etc.)
- Persistent storage: `data/knowledge_base/chroma_db/`
- Retrieval: Top-k (k=5) most relevant chunks per query
- Return chunks with similarity scores and metadata
- Filtering capability: Filter by bank or quarter using ChromaDB metadata filters
- Query interface: `query(query_text, n_results=5, filter_dict=None)`

**File: `src/rag_pipeline.py`**

- Query processing: Reformulate if needed
- Retrieval: Get top-k chunks from ChromaDB
- Context assembly: Combine retrieved chunks with query
- Prompt template: Financial domain-specific prompts
- Response generation: Pass to LLM via Ollama

### 4. Knowledge Base Structure

**Final Knowledge Base Components:**

```
data/knowledge_base/
├── chroma_db/               # ChromaDB persistent storage
│   ├── chroma.sqlite3       # ChromaDB database
│   └── ...                  # ChromaDB internal files
├── chunk_map.json           # Chunk ID to text mapping (backup)
├── metadata.json            # Full metadata for all chunks (backup)
└── statistics.json          # KB stats (total chunks, docs, etc.)
```

**Chunk Format Example:**

```json
{
  "chunk_id": "wells_fargo_q1_chunk_42",
  "text": "Net interest income for the quarter was $12.5 billion, an increase of 8% compared to the prior year quarter...",
  "metadata": {
    "bank": "Wells Fargo",
    "quarter": "Q1",
    "year": "2024",
    "section": "Income Statement",
    "page": 15,
    "source_file": "wells_fargo_q1.pdf",
    "chunk_index": 42
  },
  "embedding_index": 142
}
```

### 5. Model Integration (Ollama)

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

### 6. Evaluation Framework

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

### 7. Output Formatting

**File: `src/output_formatter.py`**

- Use `rich` library for formatted tables
- **Table 1:** Llama-3 Q&A (10 rows: Question | Answer)
- **Table 2:** Mistral-7B Q&A (10 rows: Question | Answer)
- **Table 3:** Phi-3-mini Q&A (10 rows: Question | Answer)
- **Table 4:** RAGAS Metrics (Model | Faithfulness | Context Precision | Context Recall | Answer Relevancy)
- **Table 5:** BLEU Scores (Model | BLEU Score)
- **Table 6:** ROUGE Scores (Model | ROUGE-1 | ROUGE-2 | ROUGE-L)

### 8. Domain-Specific Questions

**File: `questions/financial_questions.json`**

Create 10+ questions covering:

- Revenue trends and financial performance
- Risk factors and disclosures
- Market analysis and competitive positioning
- Regulatory compliance
- Strategic initiatives
- Financial ratios and metrics

### 9. Main Execution Flow

**File: `main.py`**

1. Process PDF documents and create knowledge base
2. Build ChromaDB vector database
3. Load questions from JSON
5. For each model:

            - Run RAG pipeline for all questions
            - Store responses

6. Evaluate all responses (RAGAS, BLEU, ROUGE)
7. Format and display results in tables

### 10. Configuration

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
- `chromadb` (vector database)
- `sentence-transformers`
- `pdfplumber` (PDF text extraction)
- `ragas`
- `nltk`, `rouge-score`
- `rich`, `pandas`
- `pyyaml`
- `tiktoken` (token counting for chunking)

## Evaluation Strategy

- Create ground truth answers for 10 questions (manual annotation or authoritative sources)
- Run RAGAS evaluation with retrieved contexts
- Compute BLEU/ROUGE against ground truth
- Compare metrics across models to identify best performer