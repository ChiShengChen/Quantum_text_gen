"""
Text generation module for quantum text models
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from tqdm import tqdm

from models.quantum_text_models import QuantumTextGenerator
from data.datasets import SimpleTokenizer

class QuantumTextGenerator_Module:
    """Text generation interface for quantum models"""
    
    def __init__(self, 
                 model: QuantumTextGenerator,
                 tokenizer: SimpleTokenizer,
                 device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Special token IDs
        self.pad_token_id = tokenizer.token_to_id('<PAD>')
        self.bos_token_id = tokenizer.token_to_id('<BOS>')
        self.eos_token_id = tokenizer.token_to_id('<EOS>')
        self.unk_token_id = tokenizer.token_to_id('<UNK>')
    
    def generate(self,
                prompt: str = "",
                max_length: int = 50,
                temperature: float = 1.0,
                top_k: int = 50,
                top_p: float = 0.9,
                repetition_penalty: float = 1.1,
                do_sample: bool = True,
                num_return_sequences: int = 1,
                early_stopping: bool = True) -> List[str]:
        """
        Generate text sequences
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or use greedy decoding
            num_return_sequences: Number of sequences to generate
            early_stopping: Whether to stop at EOS token
            
        Returns:
            List of generated text strings
        """
        self.model.eval()
        
        # Encode prompt
        if prompt:
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        else:
            # Start with BOS token
            input_ids = torch.tensor([[self.bos_token_id]], dtype=torch.long, device=self.device)
        
        # Expand for multiple sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
        
        generated_sequences = []
        
        with torch.no_grad():
            for _ in range(num_return_sequences):
                if num_return_sequences > 1:
                    current_input = input_ids[_:_+1]  # Single sequence
                else:
                    current_input = input_ids
                
                generated = self._generate_sequence(
                    current_input,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    early_stopping=early_stopping
                )
                
                # Decode to text
                generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
                generated_sequences.append(generated_text)
        
        return generated_sequences
    
    def _generate_sequence(self,
                          input_ids: torch.Tensor,
                          max_length: int,
                          temperature: float,
                          top_k: int,
                          top_p: float,
                          repetition_penalty: float,
                          do_sample: bool,
                          early_stopping: bool) -> torch.Tensor:
        """Generate a single sequence"""
        
        current_length = input_ids.size(1)
        generated = input_ids.clone()
        
        # Keep track of generated tokens for repetition penalty
        generated_tokens = set()
        
        for step in range(max_length - current_length):
            # Get next token logits
            if self.model.model_type == 'qrwkv':
                logits, _ = self.model(generated)
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            else:
                logits = self.model(generated)
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, repetition_penalty
                )
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply sampling strategies
            if do_sample:
                next_token = self._sample_token(next_token_logits, top_k, top_p)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Check for EOS token
            if early_stopping and next_token.item() == self.eos_token_id:
                break
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            generated_tokens.add(next_token.item())
        
        return generated
    
    def _apply_repetition_penalty(self,
                                 logits: torch.Tensor,
                                 generated: torch.Tensor,
                                 penalty: float) -> torch.Tensor:
        """Apply repetition penalty to logits"""
        # Get unique tokens in generated sequence
        unique_tokens = torch.unique(generated)
        
        # Apply penalty
        for token_id in unique_tokens:
            if logits[0, token_id] > 0:
                logits[0, token_id] = logits[0, token_id] / penalty
            else:
                logits[0, token_id] = logits[0, token_id] * penalty
        
        return logits
    
    def _sample_token(self,
                     logits: torch.Tensor,
                     top_k: int,
                     top_p: float) -> torch.Tensor:
        """Sample next token using top-k and nucleus sampling"""
        
        # Apply top-k filtering
        if top_k > 0:
            # Get top-k values and indices
            top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            # Set non-top-k values to -inf
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(1, top_k_indices, top_k_values)
            logits = logits_filtered
        
        # Apply nucleus (top-p) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Set filtered logits to -inf
            logits_filtered = sorted_logits.clone()
            logits_filtered[sorted_indices_to_remove] = float('-inf')
            
            # Scatter back to original positions
            logits.scatter_(1, sorted_indices, logits_filtered)
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def interactive_generation(self):
        """Interactive text generation session"""
        print("Quantum Text Generator - Interactive Mode")
        print("Type 'quit' to exit, 'help' for commands")
        print("-" * 50)
        
        while True:
            try:
                prompt = input("\nEnter prompt (or command): ").strip()
                
                if prompt.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif prompt.lower() == 'help':
                    self._print_help()
                    continue
                elif prompt.lower().startswith('set '):
                    self._handle_setting(prompt[4:])
                    continue
                
                if not prompt:
                    prompt = ""
                
                print(f"\nGenerating text for prompt: '{prompt}'")
                print("-" * 30)
                
                # Generate text
                generated_texts = self.generate(
                    prompt=prompt,
                    max_length=50,
                    temperature=1.0,
                    do_sample=True,
                    num_return_sequences=1
                )
                
                for i, text in enumerate(generated_texts):
                    print(f"Generated {i+1}: {text}")
                
            except KeyboardInterrupt:
                print("\nGeneration interrupted.")
                continue
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
    
    def _print_help(self):
        """Print help information"""
        help_text = """
Available commands:
- quit: Exit the generator
- help: Show this help message
- set <parameter> <value>: Set generation parameters
  
Available parameters:
- temperature: Controls randomness (0.1-2.0)
- max_length: Maximum generation length (10-200)
- top_k: Top-k sampling (1-100)
- top_p: Nucleus sampling (0.1-1.0)

Examples:
- set temperature 1.5
- set max_length 100
- set top_k 40
        """
        print(help_text)
    
    def _handle_setting(self, setting: str):
        """Handle parameter setting"""
        try:
            parts = setting.split()
            if len(parts) != 2:
                print("Usage: set <parameter> <value>")
                return
            
            param, value = parts
            param = param.lower()
            
            if param == 'temperature':
                self.temperature = float(value)
                print(f"Set temperature to {self.temperature}")
            elif param == 'max_length':
                self.max_length = int(value)
                print(f"Set max_length to {self.max_length}")
            elif param == 'top_k':
                self.top_k = int(value)
                print(f"Set top_k to {self.top_k}")
            elif param == 'top_p':
                self.top_p = float(value)
                print(f"Set top_p to {self.top_p}")
            else:
                print(f"Unknown parameter: {param}")
        
        except ValueError as e:
            print(f"Invalid value: {str(e)}")
    
    def batch_generate(self,
                      prompts: List[str],
                      batch_size: int = 4,
                      **generation_kwargs) -> List[str]:
        """Generate text for multiple prompts in batches"""
        
        all_generated = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i:i + batch_size]
            
            for prompt in batch_prompts:
                generated = self.generate(prompt, num_return_sequences=1, **generation_kwargs)
                all_generated.extend(generated)
        
        return all_generated
    
    def evaluate_generation_quality(self,
                                   test_prompts: List[str],
                                   reference_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate generation quality metrics"""
        
        generated_texts = self.batch_generate(test_prompts)
        
        metrics = {}
        
        # Calculate average length
        lengths = [len(text.split()) for text in generated_texts]
        metrics['avg_length'] = np.mean(lengths)
        metrics['std_length'] = np.std(lengths)
        
        # Calculate vocabulary diversity
        all_words = []
        for text in generated_texts:
            all_words.extend(text.lower().split())
        
        unique_words = set(all_words)
        metrics['vocab_diversity'] = len(unique_words) / len(all_words) if all_words else 0
        
        # Calculate repetition rate
        repetition_counts = []
        for text in generated_texts:
            words = text.lower().split()
            if len(words) > 1:
                unique_ratio = len(set(words)) / len(words)
                repetition_counts.append(1 - unique_ratio)
        
        metrics['repetition_rate'] = np.mean(repetition_counts) if repetition_counts else 0
        
        return metrics