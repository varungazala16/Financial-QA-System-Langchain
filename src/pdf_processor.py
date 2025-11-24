"""
PDF Text Extraction Module
Extracts text from financial report PDFs using pdfplumber
"""

import pdfplumber
import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple


class PDFProcessor:
    """Process PDF files and extract text content"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize PDF processor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path(__file__).parent.parent
        self.pdf_files = self._get_pdf_files()
    
    def _get_pdf_files(self) -> List[Dict[str, str]]:
        """Get list of PDF files with metadata"""
        pdf_files = []
        
        # Wells Fargo PDFs
        for pdf_name in self.config['pdf_files']['wells']:
            pdf_path = self.project_root / pdf_name
            if pdf_path.exists():
                quarter = self._extract_quarter(pdf_name)
                pdf_files.append({
                    'path': str(pdf_path),
                    'bank': 'Wells Fargo',
                    'quarter': quarter,
                    'filename': pdf_name
                })
        
        # Bank of America PDFs
        for pdf_name in self.config['pdf_files']['bofa']:
            pdf_path = self.project_root / pdf_name
            if pdf_path.exists():
                quarter = self._extract_quarter(pdf_name)
                pdf_files.append({
                    'path': str(pdf_path),
                    'bank': 'Bank of America',
                    'quarter': quarter,
                    'filename': pdf_name
                })
        
        return pdf_files
    
    def _extract_quarter(self, filename: str) -> str:
        """Extract quarter from filename"""
        if 'q1' in filename.lower():
            return 'Q1'
        elif 'q2' in filename.lower():
            return 'Q2'
        elif 'q3' in filename.lower():
            return 'Q3'
        elif 'q4' in filename.lower():
            return 'Q4'
        return 'Unknown'
    
    def extract_text(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        """
        Extract text and tables from PDF
        
        Returns:
            Tuple of (text_content, tables_list)
        """
        text_content = []
        tables_list = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    text = page.extract_text()
                    if text:
                        text_content.append(f"--- Page {page_num} ---\n{text}\n")
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            table_text = self._format_table(table)
                            tables_list.append({
                                'page': page_num,
                                'table': table_text
                            })
                            text_content.append(f"\n--- Table on Page {page_num} ---\n{table_text}\n")
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return "", []
        
        return "\n".join(text_content), tables_list
    
    def _format_table(self, table: List[List]) -> str:
        """Format table as structured text"""
        if not table:
            return ""
        
        formatted_rows = []
        for row in table:
            if row:
                # Filter out None values and join with delimiter
                clean_row = [str(cell) if cell else "" for cell in row]
                formatted_rows.append(" | ".join(clean_row))
        
        return "\n".join(formatted_rows)
    
    def process_all_pdfs(self) -> List[Dict]:
        """Process all PDF files and return extracted content"""
        processed_docs = []
        raw_data_dir = self.project_root / self.config['paths']['raw_data']
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {len(self.pdf_files)} PDF files...")
        
        for pdf_info in self.pdf_files:
            print(f"Processing: {pdf_info['filename']}")
            
            text_content, tables = self.extract_text(pdf_info['path'])
            
            # Save raw text
            output_filename = f"{pdf_info['bank'].lower().replace(' ', '_')}_{pdf_info['quarter'].lower()}.txt"
            output_path = raw_data_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            processed_docs.append({
                'bank': pdf_info['bank'],
                'quarter': pdf_info['quarter'],
                'filename': pdf_info['filename'],
                'text': text_content,
                'tables': tables,
                'raw_text_path': str(output_path),
                'metadata': {
                    'bank': pdf_info['bank'],
                    'quarter': pdf_info['quarter'],
                    'source_file': pdf_info['filename']
                }
            })
        
        print(f"Successfully processed {len(processed_docs)} PDF files")
        return processed_docs


if __name__ == "__main__":
    processor = PDFProcessor()
    docs = processor.process_all_pdfs()
    print(f"\nTotal documents processed: {len(docs)}")

