#!/usr/bin/env python3
"""
Quick start example for Quantum Text Generation
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import create_dataloader
from models.quantum_text_models import create_quantum_text_model
from generation.generator import QuantumTextGenerator_Module

def quick_demo():
    """Quick demonstration of quantum text generation"""
    
    print("Quantum Text Generation - Quick Demo")
    print("=" * 50)
    
    # 1. Load a small dataset
    print("1. Loading dataset...")
    train_loader, tokenizer = create_dataloader(
        dataset_name='simple_sentences',
        split='train',
        batch_size=4,
        max_length=32,
        vocab_size=300,
        max_samples=100,
        shuffle=True
    )
    
    print(f"   Dataset loaded: {len(train_loader.dataset)} samples")
    print(f"   Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # 2. Create a small quantum model
    print("\n2. Creating quantum model...")
    model = create_quantum_text_model(
        model_type='qasa',  # Start with QASA
        vocab_size=tokenizer.get_vocab_size(),
        device='cpu',
        hidden_dim=64,      # Small for demo
        n_qubits=4,         # Small number of qubits
        seq_len=16          # Short sequences
    )
    
    print(f"   Model created: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Test generation (without training)
    print("\n3. Testing text generation (untrained model)...")
    generator = QuantumTextGenerator_Module(model, tokenizer, 'cpu')
    
    test_prompts = [
        "the cat",
        "birds fly",
        "water flows"
    ]
    
    for prompt in test_prompts:
        try:
            generated = generator.generate(
                prompt=prompt,
                max_length=20,
                temperature=1.2,
                do_sample=True,
                num_return_sequences=1
            )
            print(f"   Prompt: '{prompt}' -> Generated: '{generated[0]}'")
        except Exception as e:
            print(f"   Prompt: '{prompt}' -> Error: {str(e)}")
    
    # 4. Show sample data
    print("\n4. Sample training data:")
    sample_batch = next(iter(train_loader))
    for i in range(min(3, len(sample_batch['input_ids']))):
        input_text = tokenizer.decode(sample_batch['input_ids'][i].tolist())
        target_text = tokenizer.decode(sample_batch['target_ids'][i].tolist())
        print(f"   Input:  {input_text}")
        print(f"   Target: {target_text}")
        print()
    
    print("Demo completed! To train the model, run:")
    print("python quantum_text_gen.py --mode train --model qasa --dataset simple_sentences --epochs 10 --batch_size 4 --max_samples 500")

if __name__ == '__main__':
    quick_demo()