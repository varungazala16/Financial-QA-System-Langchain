"""
Output Formatter Module
Creates formatted tables for displaying results
"""

from rich.console import Console
from rich.table import Table
from rich import box
from typing import List, Dict


class OutputFormatter:
    """Format and display results in tables"""
    
    def __init__(self):
        """Initialize formatter"""
        self.console = Console()
    
    def create_qa_table(self, model_name: str, responses: List[Dict]) -> Table:
        """
        Create Q&A table for a model
        
        Args:
            model_name: Name of the model
            responses: List of response dictionaries
        
        Returns:
            Rich Table object
        """
        table = Table(
            title=f"{model_name} - Questions & Answers",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("#", style="magenta", width=3, justify="right")
        table.add_column("Question", style="cyan", width=50, overflow="fold")
        table.add_column("Answer", style="green", width=100, overflow="fold", no_wrap=False)
        
        total = len(responses)
        for idx, response in enumerate(responses, 1):
            question = response.get('query', '')
            answer = response.get('answer', '')
            
            # Truncate question if too long for display
            if len(question) > 100:
                question = question[:97] + "..."
            # Don't truncate answers - show full text (table will handle overflow)
            # Answers can be long, let the table's overflow="fold" handle it
            
            # Label within the cells so each block is clearly identifiable
            q_text = f"Q{idx}: {question}"
            a_text = f"A{idx}: {answer}"
            
            table.add_row(str(idx), q_text, a_text)
            
            # Add an empty spacer row between Q&A pairs for better readability
            if idx < total:
                table.add_row("", "", "")
        
        return table

    def create_bleu_table(self, evaluation_results: Dict[str, Dict]) -> Table:
        """
        Create BLEU scores table
        
        Args:
            evaluation_results: Dictionary of evaluation results per model
        
        Returns:
            Rich Table object
        """
        table = Table(
            title="BLEU Scores",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Model", style="cyan", width=15)
        table.add_column("BLEU Score", style="yellow", justify="right")
        
        for model_name, results in evaluation_results.items():
            bleu = results.get('bleu', 0.0)
            table.add_row(model_name, f"{bleu:.4f}")
        
        return table

    def create_meteor_table(self, evaluation_results: Dict[str, Dict]) -> Table:
        """
        Create METEOR scores table
        
        Args:
            evaluation_results: Dictionary of evaluation results per model
        
        Returns:
            Rich Table object
        """
        table = Table(
            title="METEOR Scores",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Model", style="cyan", width=15)
        table.add_column("METEOR Score", style="yellow", justify="right")
        
        for model_name, results in evaluation_results.items():
            meteor = results.get('meteor', 0.0)
            table.add_row(model_name, f"{meteor:.4f}")
        
        return table
    
    def create_rouge_table(self, evaluation_results: Dict[str, Dict]) -> Table:
        """
        Create ROUGE scores table
        
        Args:
            evaluation_results: Dictionary of evaluation results per model
        
        Returns:
            Rich Table object
        """
        table = Table(
            title="ROUGE Scores",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Model", style="cyan", width=15)
        table.add_column("ROUGE-1", style="yellow", justify="right")
        table.add_column("ROUGE-2", style="yellow", justify="right")
        table.add_column("ROUGE-L", style="yellow", justify="right")
        
        for model_name, results in evaluation_results.items():
            rouge = results.get('rouge', {})
            table.add_row(
                model_name,
                f"{rouge.get('rouge1', 0.0):.4f}",
                f"{rouge.get('rouge2', 0.0):.4f}",
                f"{rouge.get('rougeL', 0.0):.4f}"
            )
        
        return table
    
    def display_all_results(
        self,
        results: Dict[str, List[Dict]],
        evaluation_results: Dict[str, Dict]
    ):
        """
        Display all results in formatted tables
        
        Args:
            results: Dictionary mapping model names to their responses
            evaluation_results: Dictionary of evaluation results per model
        """
        self.console.print("\n" + "="*100, style="bold blue")
        self.console.print("FINANCIAL DOCUMENT Q&A SYSTEM - RESULTS", style="bold blue", justify="center")
        self.console.print("="*100 + "\n", style="bold blue")
        
        # Display Q&A tables for each model
        for model_name, responses in results.items():
            qa_table = self.create_qa_table(model_name, responses)
            self.console.print(qa_table)
            self.console.print()
        
        # Display evaluation metrics
        if evaluation_results:
            bleu_table = self.create_bleu_table(evaluation_results)
            self.console.print(bleu_table)
            self.console.print()
            
            rouge_table = self.create_rouge_table(evaluation_results)
            self.console.print(rouge_table)
            self.console.print()
            
            meteor_table = self.create_meteor_table(evaluation_results)
            self.console.print(meteor_table)
            self.console.print()


if __name__ == "__main__":
    # Test formatter
    formatter = OutputFormatter()
    print("Output formatter initialized successfully")

