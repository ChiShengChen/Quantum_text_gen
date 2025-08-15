"""
Model configuration settings for quantum text generation models
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class QASAConfig:
    """Configuration for QASA (Quantum Attention Sequence Architecture) model"""
    input_dim: int = 64
    hidden_dim: int = 128
    seq_len: int = 32
    output_dim: int = 64
    n_qubits: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    device: str = 'cpu'

@dataclass
class QKSANConfig:
    """Configuration for QKSAN (Quantum-Kernel Self-Attention Network) model"""
    input_dim: int = 64
    n_qubits: int = 6
    encoding_type: str = 'amplitude'  # 'amplitude' or 'angle'
    ansatz_type: str = 'HE'  # 'HE' or 'QAOA'
    he_layers: int = 4
    qaoa_layers: int = 4
    num_classes: int = 2
    classical_hidden_dims: List[int] = None
    use_attention: bool = True
    dropout: float = 0.1
    use_classical_nn: bool = True
    device: str = 'cpu'
    
    def __post_init__(self):
        if self.classical_hidden_dims is None:
            self.classical_hidden_dims = [64, 32]

@dataclass
class QRWKVConfig:
    """Configuration for QRWKV (Quantum RWKV) model"""
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 6
    block_size: int = 128
    n_intermediate: int = 1024
    layer_norm_epsilon: float = 1e-5
    vocab_size: Optional[int] = None
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    n_qubits: int = 4
    q_depth: int = 2
    device: str = 'cpu'

@dataclass
class DataConfig:
    """Configuration for datasets"""
    dataset_name: str = 'simple_sentences'
    vocab_size: int = 1000
    max_length: int = 64
    batch_size: int = 16
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    min_length: int = 5
    max_samples: int = 10000

@dataclass
class TrainingConfig:
    """Configuration for training"""
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    save_every: int = 10
    eval_every: int = 5
    warmup_steps: int = 1000
    scheduler_type: str = 'cosine'  # 'cosine', 'linear', 'constant'
    
@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True

# Model factory mapping
MODEL_CONFIGS = {
    'qasa': QASAConfig,
    'qksan': QKSANConfig,
    'qrwkv': QRWKVConfig
}

def get_model_config(model_name: str, **kwargs) -> Any:
    """Get model configuration by name"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config_class = MODEL_CONFIGS[model_name]
    return config_class(**kwargs)