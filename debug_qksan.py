#!/usr/bin/env python3

"""
Debug script for QKSAN model issues
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from models.quantum_text_models import create_quantum_text_model
from data.datasets import create_dataloader

def debug_qksan_model():
    """Debug QKSAN model architecture and output dimensions"""
    
    print("="*60)
    print("QKSAN MODEL DEBUG")
    print("="*60)
    
    # 1. Load the saved checkpoint
    checkpoint_path = "./experiments/qksan/simple_sentences/best_model.pth"
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✅ Checkpoint loaded successfully")
        
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check model state dict
        state_dict = checkpoint['model_state_dict']
        print(f"\nModel state dict keys:")
        for key, tensor in state_dict.items():
            print(f"  {key}: {tensor.shape}")
            
        # Get embedding dimensions
        if 'embedding.weight' in state_dict:
            vocab_size, embedding_dim = state_dict['embedding.weight'].shape
            print(f"\nInferred from embedding: vocab_size={vocab_size}, embedding_dim={embedding_dim}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    # 2. Create a new QKSAN model with correct parameters
    print(f"\n" + "="*40)
    print("CREATING NEW QKSAN MODEL")
    print("="*40)
    
    try:
        # Create model with inferred parameters
        model = create_quantum_text_model(
            model_type='qksan',
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            n_qubits=6,
            model_variant='PCAAmHE',
            device='cpu'
        )
        
        print("✅ QKSAN model created successfully")
        print(f"Model type: {type(model)}")
        print(f"Model parameters:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape}")
            
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Test forward pass with different input sizes
    print(f"\n" + "="*40)
    print("TESTING FORWARD PASS")
    print("="*40)
    
    # Test with single token
    test_inputs = [
        torch.tensor([[1]]),  # Single token
        torch.tensor([[1, 2, 3]]),  # Multiple tokens
        torch.tensor([[1, 2, 3, 4, 5]])  # Longer sequence
    ]
    
    for i, input_ids in enumerate(test_inputs):
        print(f"\nTest {i+1}: Input shape {input_ids.shape}")
        try:
            with torch.no_grad():
                output = model(input_ids)
                print(f"  Output shape: {output.shape}")
                print(f"  Output values (first few): {output.flatten()[:5]}")
                
                # Check if output is reasonable for vocab_size
                if output.shape[-1] != vocab_size:
                    print(f"  ❌ Output vocab dimension ({output.shape[-1]}) != expected vocab_size ({vocab_size})")
                else:
                    print(f"  ✅ Output vocab dimension matches expected vocab_size ({vocab_size})")
                    
        except Exception as e:
            print(f"  ❌ Error in forward pass: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. Analyze the QKSAN internal model
    print(f"\n" + "="*40)
    print("ANALYZING QKSAN INTERNAL MODEL")
    print("="*40)
    
    if hasattr(model, 'qksan_model'):
        qksan_internal = model.qksan_model
        print(f"QKSAN internal model type: {type(qksan_internal)}")
        print(f"QKSAN internal model attributes:")
        for attr in dir(qksan_internal):
            if not attr.startswith('_'):
                try:
                    value = getattr(qksan_internal, attr)
                    if hasattr(value, 'shape'):
                        print(f"  {attr}: {value.shape}")
                    elif isinstance(value, (int, float, str, bool)):
                        print(f"  {attr}: {value}")
                except:
                    print(f"  {attr}: <unable to access>")
        
        # Test the internal QKSAN model directly
        print(f"\nTesting QKSAN internal model:")
        try:
            # Test with embedding dimension input
            test_embedding = torch.randn(1, embedding_dim)
            internal_output = qksan_internal(test_embedding)
            print(f"  Internal input shape: {test_embedding.shape}")
            print(f"  Internal output shape: {internal_output.shape}")
            print(f"  Internal output values: {internal_output.flatten()}")
            
        except Exception as e:
            print(f"  ❌ Error testing internal model: {e}")
            import traceback
            traceback.print_exc()
    
    # 5. Load and test the original checkpoint
    print(f"\n" + "="*40)
    print("TESTING ORIGINAL CHECKPOINT")
    print("="*40)
    
    try:
        # Load state dict with flexibility for missing keys
        model_state_dict = checkpoint['model_state_dict']
        current_state_dict = model.state_dict()
        
        # Filter out keys that don't exist in the current model
        filtered_state_dict = {}
        missing_keys = []
        
        for key, value in model_state_dict.items():
            if key in current_state_dict:
                if current_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
        
        for key in current_state_dict:
            if key not in model_state_dict:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"Missing keys (will use random initialization): {missing_keys}")
        
        model.load_state_dict(filtered_state_dict, strict=False)
        print("✅ Checkpoint loaded into model (with missing keys handled)")
        
        # Test forward pass with loaded weights
        test_input = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            output = model(test_input)
            print(f"With loaded weights - Output shape: {output.shape}")
            print(f"With loaded weights - Output values (first few): {output.flatten()[:5]}")
            
    except Exception as e:
        print(f"❌ Error loading checkpoint into model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_qksan_model()