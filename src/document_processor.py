"""
Document Processing Module
Handles text cleaning, chunking, and metadata preservation
"""

import re
import json
import yaml
import tiktoken
from pathlib import Path
from typing import List, Dict, Any


class DocumentProcessor:
    """Process and chunk documents for RAG"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize document processor"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path(__file__).parent.parent
        self.chunk_size = self.config['chunking']['chunk_size']
        self.chunk_overlap = self.config['chunking']['chunk_overlap']
        self.separator = self.config['chunking']['separator']
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Preserve currency symbols and percentages
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s\$\%\.\,\-\(\)\:]', ' ', text)
        
        # Remove excessive spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces with overlap
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
        
        Returns:
            List of chunk dictionaries
        """
        # Clean text first
        cleaned_text = self.clean_text(text)
        
        # Split by paragraphs first
        paragraphs = cleaned_text.split(self.separator)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = self.count_tokens(para)
            
            # If paragraph itself is larger than chunk size, split it further
            if para_tokens > self.chunk_size:
                # Split large paragraph into sentences
                sentences = re.split(r'[.!?]+\s+', para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sent_tokens = self.count_tokens(sentence)
                    
                    if current_tokens + sent_tokens > self.chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(self._create_chunk(chunk_text, metadata, chunk_index))
                        chunk_index += 1
                        
                        # Start new chunk with overlap
                        overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                        current_chunk = overlap_text + [sentence]
                        current_tokens = self.count_tokens(' '.join(current_chunk))
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sent_tokens
            else:
                # Check if adding this paragraph would exceed chunk size
                if current_tokens + para_tokens > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, metadata, chunk_index))
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + [para]
                    current_tokens = self.count_tokens(' '.join(current_chunk))
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens
        
        # Add final chunk if exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, metadata, chunk_index))
        
        return chunks
    
    def _get_overlap_text(self, text_list: List[str], overlap_tokens: int) -> List[str]:
        """Get overlap text from the end of current chunk"""
        if not text_list:
            return []
        
        # Reverse the list and collect text until we reach overlap_tokens
        overlap_list = []
        overlap_count = 0
        
        for item in reversed(text_list):
            item_tokens = self.count_tokens(item)
            if overlap_count + item_tokens <= overlap_tokens:
                overlap_list.insert(0, item)
                overlap_count += item_tokens
            else:
                # If we can't fit the whole item, try to split it
                if overlap_count < overlap_tokens:
                    # Take partial sentence if needed
                    words = item.split()
                    for word in reversed(words):
                        word_tokens = self.count_tokens(word)
                        if overlap_count + word_tokens <= overlap_tokens:
                            overlap_list.insert(0, word)
                            overlap_count += word_tokens
                        else:
                            break
                break
        
        return overlap_list
    
    def _create_chunk(self, text: str, metadata: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata"""
        chunk_id = f"{metadata['bank'].lower().replace(' ', '_')}_{metadata['quarter'].lower()}_chunk_{chunk_index}"
        
        return {
            'chunk_id': chunk_id,
            'text': text,
            'metadata': {
                'bank': metadata['bank'],
                'quarter': metadata['quarter'],
                'source_file': metadata.get('source_file', ''),
                'chunk_index': chunk_index,
                **metadata.get('additional_metadata', {})
            }
        }
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process all documents and create chunks"""
        all_chunks = []
        processed_dir = self.project_root / self.config['paths']['processed_data']
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {len(documents)} documents into chunks...")
        
        for doc in documents:
            print(f"Chunking: {doc['bank']} {doc['quarter']}")
            
            chunks = self.chunk_text(doc['text'], doc['metadata'])
            all_chunks.extend(chunks)
            
            print(f"  Created {len(chunks)} chunks")
        
        # Save chunks to JSONL file
        chunks_file = processed_dir / "chunks.jsonl"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        # Save chunk index
        chunk_index = {chunk['chunk_id']: i for i, chunk in enumerate(all_chunks)}
        index_file = processed_dir / "chunk_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_index, f, indent=2)
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        print(f"Chunks saved to: {chunks_file}")
        
        return all_chunks


if __name__ == "__main__":
    # Test the processor
    processor = DocumentProcessor()
    test_text = "This is a test document. " * 50
    test_metadata = {
        'bank': 'Wells Fargo',
        'quarter': 'Q1',
        'source_file': 'test.pdf'
    }
    chunks = processor.chunk_text(test_text, test_metadata)
    print(f"Created {len(chunks)} chunks from test text")

