#!/usr/bin/env python3
"""
Test script for fixed QKSAN model
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qksan_fixed import create_fixed_qksan_model, StableQKSANTextGenerator
from data.datasets import create_dataloader

def test_qksan_model():
    """Test the fixed QKSAN model"""
    
    print("Testing Fixed QKSAN Model")
    print("=" * 50)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 100
    embedding_dim = 64
    n_qubits = 4
    batch_size = 4
    seq_len = 16
    
    print(f"Device: {device}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of qubits: {n_qubits}")
    
    # Create model
    print("\n1. Creating fixed QKSAN model...")
    model = create_fixed_qksan_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_qubits=n_qubits,
        device=device
    )
    
    print(f"Model created: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    try:
        with torch.no_grad():
            logits = model(input_ids)
        print(f"Forward pass successful!")
        print(f"Output shape: {logits.shape}")
        print(f"Output range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return False
    
    # Test training step
    print("\n3. Testing training step...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    try:
        # Forward pass
        logits = model(input_ids)
        
        # Create targets (shift input_ids by 1 for language modeling)
        targets = input_ids[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        
        # Reshape for loss calculation
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # Calculate loss
        loss = criterion(logits_flat, targets_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        print(f"Training step successful!")
        print(f"Loss: {loss.item():.4f}")
        print(f"Gradient norm: {grad_norm:.4f}")
        
        # Check for NaN or inf
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Loss is NaN or Inf!")
            return False
            
    except Exception as e:
        print(f"Training step failed: {e}")
        return False
    
    # Test generation
    print("\n4. Testing text generation...")
    model.eval()
    
    try:
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids[:, :4],  # Start with first 4 tokens
                max_length=8,
                temperature=1.0,
                do_sample=True
            )
        print(f"Generation successful!")
        print(f"Generated shape: {generated.shape}")
        print(f"Generated tokens: {generated[0].tolist()}")
    except Exception as e:
        print(f"Generation failed: {e}")
        return False
    
    # Test with real data
    print("\n5. Testing with real dataset...")
    try:
        train_loader, tokenizer = create_dataloader(
            dataset_name='simple_sentences',
            split='train',
            batch_size=batch_size,
            max_length=seq_len,
            vocab_size=vocab_size,
            max_samples=100,
            shuffle=True
        )
        
        sample_batch = next(iter(train_loader))
        input_ids = sample_batch['input_ids'].to(device)
        target_ids = sample_batch['target_ids'].to(device)
        
        # Forward pass
        logits = model(input_ids)
        
        # Calculate loss
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = target_ids.view(-1)
        loss = criterion(logits_flat, targets_flat)
        
        print(f"Real data test successful!")
        print(f"Loss on real data: {loss.item():.4f}")
        
    except Exception as e:
        print(f"Real data test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! Fixed QKSAN model is working correctly.")
    return True

def test_training_stability():
    """Test training stability over multiple epochs"""
    
    print("\nTesting Training Stability")
    print("=" * 50)
    
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
        batch_size=8,
        max_length=16,
        vocab_size=vocab_size,
        max_samples=200,
        shuffle=True
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    
    print("Training for 5 epochs...")
    for epoch in range(5):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids)
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = target_ids.view(-1)
            
            loss = criterion(logits_flat, targets_flat)
            
            # Check for NaN or inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"❌ Epoch {epoch+1}: Loss is NaN or Inf!")
                return False
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Check if loss is decreasing
        if epoch > 0 and losses[-1] > losses[-2] * 1.5:
            print(f"⚠️  Warning: Loss increased significantly")
    
    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Loss trend: {'Decreasing' if losses[-1] < losses[0] else 'Not decreasing'}")
    
    if losses[-1] < float('inf') and not torch.isnan(torch.tensor(losses[-1])):
        print("✅ Training stability test passed!")
        return True
    else:
        print("❌ Training stability test failed!")
        return False

if __name__ == '__main__':
    print("QKSAN Model Fix Test")
    print("=" * 60)
    
    # Test basic functionality
    basic_test = test_qksan_model()
    
    # Test training stability
    stability_test = test_training_stability()
    
    print("\n" + "=" * 60)
    if basic_test and stability_test:
        print("🎉 All tests passed! QKSAN model is fixed and ready for use.")
    else:
        print("❌ Some tests failed. Further debugging needed.")
