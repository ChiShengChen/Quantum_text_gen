#!/usr/bin/env python3
"""
Fix generation functionality for QKSAN model
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qksan_fixed import create_fixed_qksan_model
from data.datasets import create_dataloader

def test_generation_fix():
    """Test the fixed generation functionality"""
    
    print("Testing QKSAN Generation Fix")
    print("=" * 50)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 50
    embedding_dim = 32
    n_qubits = 4
    
    # Create model
    model = create_fixed_qksan_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_qubits=n_qubits,
        device=device
    )
    
    # Create simple dataset
    train_loader, tokenizer = create_dataloader(
        dataset_name='simple_sentences',
        split='train',
        batch_size=4,
        max_length=16,
        vocab_size=vocab_size,
        max_samples=100,
        shuffle=True
    )
    
    # Quick training to get reasonable weights
    print("Quick training for generation test...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = target_ids.view(-1)
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()
        break  # Just one step
    
    # Test generation with different top_k values
    print("\nTesting generation with different top_k values...")
    model.eval()
    
    test_prompts = ["the", "birds", "water"]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Tokenize prompt
        prompt_tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_tokens], device=device)
        
        # Test different top_k values
        for top_k in [5, 10, 20, 50]:
            try:
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=input_ids,
                        max_length=8,
                        temperature=1.0,
                        top_k=top_k,
                        do_sample=True
                    )
                
                # Decode
                generated_text = tokenizer.decode(generated[0].tolist())
                print(f"  top_k={top_k}: '{generated_text}'")
                
            except Exception as e:
                print(f"  top_k={top_k}: Error - {e}")
    
    print("\n✅ Generation test completed!")

if __name__ == '__main__':
    test_generation_fix()
