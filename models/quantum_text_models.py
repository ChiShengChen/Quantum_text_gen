"""
Quantum text generation models - adapters for the three quantum models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import numpy as np

# Import the original quantum models
from qasa_model import QASAWrapper
from qksan_improved import EnhancedQuantumClassifier, build_enhanced_model
from qrwkv_model import QuantumRWKVModel, ModelConfig as QRWKVModelConfig
from qksan_fixed import create_fixed_qksan_model

class QuantumTextGenerator(nn.Module):
    """Base class for quantum text generators"""
    
    def __init__(self, vocab_size: int, model_type: str = 'qasa'):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_type = model_type
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 50, **kwargs) -> torch.Tensor:
        """Generate text sequences"""
        raise NotImplementedError

class QASATextGenerator(QuantumTextGenerator):
    """QASA model adapted for text generation"""
    
    def __init__(self, 
                 vocab_size: int,
                 hidden_dim: int = 128,
                 seq_len: int = 32,
                 n_qubits: int = 8,
                 device: str = 'cpu'):
        super().__init__(vocab_size, 'qasa')
        
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.device = device
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # QASA model for sequence processing
        self.qasa_model = QASAWrapper(
            input_dim=1,  # QASA expects scalar inputs per sequence position
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            output_dim=1  # Output scalar per sequence
        )
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        self.to(device)
    
    def forward(self, input_ids: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for text generation
        Args:
            input_ids: [batch_size, seq_len]
            timestep: optional timestep information
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, hidden_dim]
        
        # QASA expects [B, S] input, so we need to convert embedded features
        # We'll use a simple approach: sum the embeddings along the feature dimension
        # to get a scalar for each position, then process the sequence
        seq_features = embedded.sum(dim=-1)  # [batch_size, seq_len] - sum embeddings to scalars
        
        # Process through QASA model for each position
        outputs = []
        for i in range(seq_len):
            # Take sequence up to position i+1
            seq_input = seq_features[:, :i+1]  # [batch_size, i+1]
            
            # Pad or truncate to fixed length
            if seq_input.size(1) < self.seq_len:
                padding = torch.zeros(batch_size, self.seq_len - seq_input.size(1), device=self.device)
                seq_input = torch.cat([seq_input, padding], dim=1)
            elif seq_input.size(1) > self.seq_len:
                seq_input = seq_input[:, -self.seq_len:]
            
            # Process through QASA (expects [B, S] input)
            if timestep is not None:
                qasa_output = self.qasa_model(seq_input, timestep)  # [B]
            else:
                qasa_output = self.qasa_model(seq_input)  # [B]
            
            # Convert back to hidden_dim size by combining with original embedding
            # Use the last embedded vector for this position
            combined_output = embedded[:, i, :] * qasa_output.unsqueeze(-1)  # [B, hidden_dim]
            outputs.append(combined_output)
        
        # Stack outputs
        hidden_states = torch.stack(outputs, dim=1)  # [batch_size, seq_len, hidden_dim]
        
        # Project to vocabulary
        logits = self.output_proj(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def generate(self, 
                input_ids: torch.Tensor, 
                max_length: int = 50,
                temperature: float = 1.0,
                top_k: int = 50,
                do_sample: bool = True) -> torch.Tensor:
        """Generate text sequences"""
        self.eval()
        batch_size = input_ids.size(0)
        current_length = input_ids.size(1)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - current_length):
                # Get logits for next token
                logits = self.forward(generated)  # [batch_size, seq_len, vocab_size]
                next_token_logits = logits[:, -1, :] / temperature  # [batch_size, vocab_size]
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated

class QKSANTextGenerator(QuantumTextGenerator):
    """QKSAN model adapted for text generation"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 128,
                 n_qubits: int = 6,
                 model_variant: str = "fixed",
                 device: str = 'cpu'):
        super().__init__(vocab_size, 'qksan')
        
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        self.device = device
        
        # Use fixed QKSAN model for better stability
        if model_variant == "fixed":
            self.qksan_model = create_fixed_qksan_model(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                n_qubits=n_qubits,
                device=device
            )
        else:
            # Fallback to original model
            self.qksan_model = build_enhanced_model(
                model_variant, embedding_dim, n_qubits, vocab_size, device
            )
        
        self.to(device)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for next-token prediction
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Use the fixed QKSAN model's forward method
        return self.qksan_model(input_ids)
    
    def generate(self,
                input_ids: torch.Tensor,
                max_length: int = 50,
                temperature: float = 1.0,
                top_k: int = 50,
                do_sample: bool = True) -> torch.Tensor:
        """Generate text sequences"""
        self.eval()
        batch_size = input_ids.size(0)
        current_length = input_ids.size(1)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - current_length):
                # Get context (average of all previous tokens)
                embedded = self.embedding(generated)  # [batch_size, seq_len, embedding_dim]
                context = embedded.mean(dim=1)  # [batch_size, embedding_dim]
                
                # Get logits for next token
                logits = self.qksan_model(context) / temperature  # [batch_size, vocab_size]
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = float('-inf')
                    
                    # Sample from distribution
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated

class QRWKVTextGenerator(QuantumTextGenerator):
    """QRWKV model for text generation"""
    
    def __init__(self,
                 vocab_size: int,
                 n_embd: int = 256,
                 n_head: int = 8,
                 n_layer: int = 6,
                 block_size: int = 128,
                 n_qubits: int = 4,
                 device: str = 'cpu'):
        super().__init__(vocab_size, 'qrwkv')
        
        # Create QRWKV config
        config = QRWKVModelConfig(
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            block_size=block_size,
            vocab_size=vocab_size,
            n_qubits=n_qubits
        )
        
        # Create QRWKV model
        self.qrwkv_model = QuantumRWKVModel(config)
        self.config = config
        self.device = device
        
        self.to(device)
    
    def forward(self, input_ids: torch.Tensor, states: Optional[Any] = None) -> Tuple[torch.Tensor, Any]:
        """
        Forward pass
        Args:
            input_ids: [batch_size, seq_len]
            states: RWKV states from previous forward pass
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            new_states: Updated RWKV states
        """
        logits, new_states = self.qrwkv_model(input_ids, states)
        return logits, new_states
    
    def generate(self,
                input_ids: torch.Tensor,
                max_length: int = 50,
                temperature: float = 1.0,
                top_k: int = 50,
                do_sample: bool = True) -> torch.Tensor:
        """Generate text sequences"""
        self.eval()
        batch_size = input_ids.size(0)
        current_length = input_ids.size(1)
        
        generated = input_ids.clone()
        states = None
        
        with torch.no_grad():
            # Initial forward pass to get states
            _, states = self.forward(generated, states)
            
            for _ in range(max_length - current_length):
                # 修改：使用整個序列而不是只使用最後一個token
                # 這樣可以保持上下文信息
                logits, states = self.forward(generated, states)  # [batch_size, seq_len, vocab_size]
                next_token_logits = logits[:, -1, :] / temperature  # [batch_size, vocab_size]
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated

def create_quantum_text_model(model_type: str, vocab_size: int, **kwargs) -> QuantumTextGenerator:
    """Factory function to create quantum text models"""
    
    if model_type.lower() == 'qasa':
        return QASATextGenerator(vocab_size, **kwargs)
    elif model_type.lower() == 'qksan':
        # Map hidden_dim to embedding_dim for QKSAN
        if 'hidden_dim' in kwargs:
            kwargs['embedding_dim'] = kwargs.pop('hidden_dim')
        return QKSANTextGenerator(vocab_size, **kwargs)
    elif model_type.lower() == 'qrwkv':
        # Map hidden_dim to n_embd for QRWKV
        if 'hidden_dim' in kwargs:
            kwargs['n_embd'] = kwargs.pop('hidden_dim')
        return QRWKVTextGenerator(vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: qasa, qksan, qrwkv")