"""
Ollama Model Wrapper
Provides unified interface for multiple LLM models via Ollama
"""

import ollama
from typing import Dict, Optional
import time


class OllamaModel:
    """Wrapper for Ollama LLM models"""
    
    def __init__(self, model_name: str, display_name: str, temperature: float = 0.1):
        """
        Initialize Ollama model
        
        Args:
            model_name: Ollama model name (e.g., "llama3:8b")
            display_name: Display name for the model
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.display_name = display_name
        self.temperature = temperature
        self.max_retries = 3
        self.retry_delay = 2
    
    def generate_response(self, query: str, context: str, max_retries: Optional[int] = None) -> str:
        """
        Generate response using the model
        
        Args:
            query: User query
            context: Retrieved context from vector store
            max_retries: Maximum number of retry attempts
        
        Returns:
            Generated response text
        """
        max_retries = max_retries or self.max_retries
        
        prompt = self._create_prompt(query, context)
        
        for attempt in range(max_retries):
            try:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': self.temperature,
                        'num_predict': 512  # Limit response length
                    }
                )
                return response['response'].strip()
            
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error generating response (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed to generate response after {max_retries} attempts: {str(e)}")
                    return f"Error: Unable to generate response. {str(e)}"
        
        return "Error: Unable to generate response."
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt template for financial Q&A"""
        prompt = f"""You are a financial analyst assistant. Answer the following question based on the provided financial document context.

Context from financial reports:
{context}

Question: {query}

Instructions:
- Answer based only on the provided context
- If the information is not in the context, say so
- Be precise with financial figures and numbers
- Cite the source (bank and quarter) when mentioning specific data
- Provide a clear, concise answer in 1–2 sentences
- When possible, begin your answer with the phrase: "According to [Bank] [Quarter] report, ..."
- Use wording that closely follows the phrasing and numbers given in the context

Answer:"""
        return prompt
    
    def check_model_available(self) -> bool:
        """Check if the model is available in Ollama"""
        try:
            models_response = ollama.list()
            
            # Handle different response structures
            models_list = []
            if isinstance(models_response, dict):
                if 'models' in models_response:
                    models_list = models_response['models']
                else:
                    # Try all values to find a list
                    for value in models_response.values():
                        if isinstance(value, list):
                            models_list = value
                            break
            elif isinstance(models_response, list):
                models_list = models_response
            
            # Extract model names - try multiple approaches
            model_names = []
            for model in models_list:
                if isinstance(model, dict):
                    # Try all possible keys for model name
                    name = (model.get('name') or 
                           model.get('model') or 
                           model.get('model_name') or
                           model.get('id') or
                           str(model).split("'")[1] if "'" in str(model) else None)
                    if name:
                        model_names.append(str(name))
                elif isinstance(model, str):
                    model_names.append(model)
                else:
                    # Try to convert to string and extract
                    model_str = str(model)
                    if ':' in model_str:
                        model_names.append(model_str)
            
            # Also try a direct test - attempt to use the model
            try:
                # Quick test generation to verify model works
                test_response = ollama.generate(
                    model=self.model_name,
                    prompt="test",
                    options={'num_predict': 1}
                )
                # If we get here without error, model exists
                return True
            except:
                # If direct test fails, fall back to name matching
                pass
            
            # Check if our model name matches (exact or base name match)
            base_name = self.model_name.split(':')[0]
            for name in model_names:
                # Exact match
                if self.model_name == name:
                    return True
                # Base name match (e.g., "llama3" matches "llama3:8b" or "llama3:latest")
                if name.startswith(base_name + ':') or name == base_name:
                    return True
                # Reverse check - our name starts with listed name
                if name and self.model_name.startswith(name.split(':')[0]):
                    return True
            
            return False
        except Exception as e:
            print(f"Error checking model availability for {self.model_name}: {str(e)}")
            # Try direct test as fallback
            try:
                ollama.generate(model=self.model_name, prompt="test", options={'num_predict': 1})
                return True
            except:
                return False


class ModelManager:
    """Manage multiple Ollama models"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize model manager with all configured models"""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        for model_config in self.config['models']:
            model_name = model_config['name']
            display_name = model_config['display_name']
            self.models[display_name] = OllamaModel(model_name, display_name)
        
        # Verify models are available
        self._verify_models()
    
    def _verify_models(self):
        """Verify all models are available - exits if any model is missing"""
        import sys
        
        print("Verifying Ollama models...")
        
        # First, show what models Ollama knows about
        try:
            models_response = ollama.list()
            print("\nModels detected by Ollama:")
            if isinstance(models_response, dict) and 'models' in models_response:
                for m in models_response['models']:
                    model_name = m.get('name', 'unknown')
                    print(f"  - {model_name}")
            print()
        except Exception as e:
            print(f"  (Could not list models: {e})\n")
        
        missing_models = []
        
        for display_name, model in self.models.items():
            if model.check_model_available():
                print(f"✓ {display_name} ({model.model_name}) is available")
            else:
                print(f"✗ {display_name} ({model.model_name}) is NOT available")
                missing_models.append((display_name, model.model_name))
        
        if missing_models:
            print("\n" + "="*80)
            print("ERROR: Required models are not available!")
            print("="*80)
            for display_name, model_name in missing_models:
                print(f"  Missing: {display_name} ({model_name})")
                print(f"  Run: ollama pull {model_name}")
            print("\nPlease install all required models before running the system.")
            print("="*80)
            sys.exit(1)
        
        print(f"\n✓ All {len(self.models)} models are available and ready to use.\n")
    
    def get_model(self, display_name: str) -> Optional[OllamaModel]:
        """Get a model by display name"""
        return self.models.get(display_name)
    
    def get_all_models(self) -> Dict[str, OllamaModel]:
        """Get all models"""
        return self.models


if __name__ == "__main__":
    # Test model manager
    manager = ModelManager()
    print(f"\nAvailable models: {list(manager.models.keys())}")

