"""
Evaluation Module
Implements BLEU, ROUGE, and METEOR metrics for evaluation
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk


# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass


class Evaluator:
    """Evaluate RAG responses using multiple metrics"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize evaluator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path(__file__).parent.parent
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
    
    def load_ground_truth(self) -> Dict[str, str]:
        """Load ground truth answers"""
        gt_path = self.project_root / self.config['evaluation']['ground_truth_file']
        
        if not gt_path.exists():
            print(f"Warning: Ground truth file not found at {gt_path}")
            return {}
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def evaluate_bleu(self, generated_answers: List[str], reference_answers: List[str]) -> float:
        """
        Evaluate using BLEU score
        
        Args:
            generated_answers: List of generated answers
            reference_answers: List of reference answers
        
        Returns:
            Average BLEU score
        """
        if len(generated_answers) != len(reference_answers):
            print("Warning: Mismatch in number of answers")
            return 0.0
        
        bleu_scores = []
        
        for gen, ref in zip(generated_answers, reference_answers):
            # Tokenize
            gen_tokens = gen.lower().split()
            ref_tokens = ref.lower().split()
            
            # Calculate BLEU
            try:
                score = sentence_bleu(
                    [ref_tokens],
                    gen_tokens,
                    smoothing_function=self.smoothing
                )
                bleu_scores.append(score)
            except:
                bleu_scores.append(0.0)
        
        return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    def evaluate_meteor(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> float:
        """
        Evaluate using METEOR score
        
        Args:
            generated_answers: List of generated answers
            reference_answers: List of reference answers
        
        Returns:
            Average METEOR score
        """
        if len(generated_answers) != len(reference_answers):
            print("Warning: Mismatch in number of answers for METEOR")
            return 0.0
        
        meteor_scores = []
        for gen, ref in zip(generated_answers, reference_answers):
            try:
                # meteor_score expects list of references and a hypothesis
                score = meteor_score([ref], gen)
                meteor_scores.append(score)
            except Exception:
                meteor_scores.append(0.0)
        
        return sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    
    def evaluate_rouge(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate using ROUGE scores
        
        Args:
            generated_answers: List of generated answers
            reference_answers: List of reference answers
        
        Returns:
            Dictionary of ROUGE scores
        """
        if len(generated_answers) != len(reference_answers):
            print("Warning: Mismatch in number of answers")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for gen, ref in zip(generated_answers, reference_answers):
            scores = self.rouge_scorer.score(ref, gen)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
            'rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
            'rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
        }
    
    def evaluate_all(
        self,
        results: Dict[str, List[Dict]],
        questions: List[str],
        ground_truth: Optional[Dict[str, str]] = None
    ) -> Dict[str, Dict]:
        """
        Evaluate all models' responses
        
        Args:
            results: Dictionary mapping model names to their responses
            questions: List of questions
            ground_truth: Optional dictionary mapping questions to ground truth answers
        
        Returns:
            Dictionary of evaluation results per model
        """
        evaluation_results = {}
        
        # Prepare ground truth answers if available
        reference_answers = None
        if ground_truth:
            reference_answers = [ground_truth.get(q, "") for q in questions]
        
        for model_name, responses in results.items():
            print(f"\nEvaluating {model_name}...")
            
            answers = [r['answer'] for r in responses]
            
            # BLEU evaluation (if ground truth available)
            bleu_score = 0.0
            if reference_answers:
                bleu_score = self.evaluate_bleu(answers, reference_answers)
            
            # ROUGE evaluation (if ground truth available)
            rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            if reference_answers:
                rouge_scores = self.evaluate_rouge(answers, reference_answers)
            
            # METEOR evaluation (if ground truth available)
            meteor_score_val = 0.0
            if reference_answers:
                meteor_score_val = self.evaluate_meteor(answers, reference_answers)
            
            evaluation_results[model_name] = {
                'bleu': bleu_score,
                'rouge': rouge_scores,
                'meteor': meteor_score_val
            }
        
        return evaluation_results


if __name__ == "__main__":
    # Test evaluator
    evaluator = Evaluator()
    print("Evaluator initialized successfully")

