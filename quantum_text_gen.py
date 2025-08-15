#!/usr/bin/env python3
"""
Quantum Text Generation Project
Main entry point for training and generating text with quantum models

Usage:
    python quantum_text_gen.py --mode train --model qasa --dataset simple_sentences
    python quantum_text_gen.py --mode generate --model qrwkv --checkpoint ./checkpoints/best_model.pth
    python quantum_text_gen.py --mode interactive --model qksan
"""

import argparse
import os
import sys
import torch
import json
from typing import Dict, Any, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from config.model_configs import get_model_config, DataConfig, TrainingConfig, GenerationConfig
from config.data_configs import list_available_datasets, get_dataset_info
from data.datasets import create_dataloader
from models.quantum_text_models import create_quantum_text_model
from training.trainer import QuantumTextTrainer, create_optimizer, create_scheduler
from generation.generator import QuantumTextGenerator_Module
from utils.logger import Logger, ExperimentTracker
from utils.metrics import GenerationEvaluator

def setup_device() -> str:
    """Setup computation device"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    return device

def train_model(args) -> Dict[str, Any]:
    """Train a quantum text generation model"""
    
    # Setup
    device = setup_device()
    logger = Logger(args.log_dir, f"{args.model}_{args.dataset}")
    
    # Log configuration
    config = vars(args)
    logger.log_config(config)
    
    # Create data loaders
    logger.log("Loading dataset...")
    train_loader, tokenizer = create_dataloader(
        dataset_name=args.dataset,
        split='train',
        batch_size=args.batch_size,
        max_length=args.max_length,
        vocab_size=args.vocab_size,
        max_samples=args.max_samples,
        shuffle=True
    )
    
    val_loader, _ = create_dataloader(
        dataset_name=args.dataset,
        split='val',
        batch_size=args.batch_size,
        max_length=args.max_length,
        vocab_size=args.vocab_size,
        max_samples=args.max_samples // 10 if args.max_samples else None,
        shuffle=False
    )
    
    logger.log(f"Train samples: {len(train_loader.dataset)}")
    logger.log(f"Validation samples: {len(val_loader.dataset)}")
    logger.log(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Create model
    logger.log(f"Creating {args.model} model...")
    model_kwargs = {
        'device': device,
        'hidden_dim': args.hidden_dim,
        'n_qubits': args.n_qubits
    }
    
    if args.model == 'qrwkv':
        model_kwargs.update({
            'n_embd': args.hidden_dim,
            'n_layer': args.n_layers,
            'n_head': args.n_heads,
            'block_size': args.max_length
        })
    elif args.model == 'qksan':
        model_kwargs.update({
            'embedding_dim': args.hidden_dim,
            'model_variant': args.qksan_variant
        })
    
    model = create_quantum_text_model(
        model_type=args.model,
        vocab_size=tokenizer.get_vocab_size(),
        **model_kwargs
    )
    
    logger.log_model_info(model)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model, 
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        num_epochs=args.epochs
    )
    
    # Create trainer
    trainer = QuantumTextTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger
    )
    
    # Train
    logger.log("Starting training...")
    training_history = trainer.train(
        epochs=args.epochs,
        save_dir=args.checkpoint_dir,
        eval_every=args.eval_every,
        save_every=args.save_every,
        early_stopping_patience=args.patience
    )
    
    # Save tokenizer
    tokenizer_path = os.path.join(args.checkpoint_dir, 'tokenizer.json')
    with open(tokenizer_path, 'w') as f:
        json.dump({
            'word_to_id': tokenizer.word_to_id,
            'id_to_word': tokenizer.id_to_word,
            'vocab_size': tokenizer.get_vocab_size()
        }, f, indent=2)
    
    logger.log(f"Training completed. Best validation loss: {trainer.best_val_loss:.4f}")
    logger.save_metrics()
    
    return {
        'best_val_loss': trainer.best_val_loss,
        'training_history': training_history,
        'model_path': os.path.join(args.checkpoint_dir, 'best_model.pth'),
        'tokenizer_path': tokenizer_path
    }

def generate_text(args) -> None:
    """Generate text using a trained model"""
    
    device = setup_device()
    
    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(args.checkpoint), 'tokenizer.json')
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}")
        return
    
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    
    # Recreate tokenizer
    from data.datasets import SimpleTokenizer
    tokenizer = SimpleTokenizer(tokenizer_data['vocab_size'])
    tokenizer.word_to_id = tokenizer_data['word_to_id']
    tokenizer.id_to_word = {int(k): v for k, v in tokenizer_data['id_to_word'].items()}
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_type = checkpoint.get('model_type', args.model)
    
    # Create model
    model_kwargs = {
        'device': device,
        'hidden_dim': 128,  # Default values - should be saved in checkpoint
        'n_qubits': 6
    }
    
    model = create_quantum_text_model(
        model_type=model_type,
        vocab_size=tokenizer.get_vocab_size(),
        **model_kwargs
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded {model_type} model from {args.checkpoint}")
    
    # Create generator
    generator = QuantumTextGenerator_Module(model, tokenizer, device)
    
    # Generate text
    if args.prompt:
        print(f"\nGenerating text for prompt: '{args.prompt}'")
        generated_texts = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            num_return_sequences=args.num_sequences
        )
        
        print("\nGenerated text:")
        print("-" * 50)
        for i, text in enumerate(generated_texts):
            print(f"{i+1}: {text}")
            print()
    else:
        print("No prompt provided. Use --prompt to specify input text.")

def interactive_mode(args) -> None:
    """Interactive text generation mode"""
    
    device = setup_device()
    
    # Load or create a simple model for demo
    if args.checkpoint and os.path.exists(args.checkpoint):
        # Load trained model
        tokenizer_path = os.path.join(os.path.dirname(args.checkpoint), 'tokenizer.json')
        
        with open(tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)
        
        from data.datasets import SimpleTokenizer
        tokenizer = SimpleTokenizer(tokenizer_data['vocab_size'])
        tokenizer.word_to_id = tokenizer_data['word_to_id']
        tokenizer.id_to_word = {int(k): v for k, v in tokenizer_data['id_to_word'].items()}
        
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model_type = checkpoint.get('model_type', args.model)
        
        model = create_quantum_text_model(
            model_type=model_type,
            vocab_size=tokenizer.get_vocab_size(),
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained {model_type} model")
    else:
        # Create demo model with simple dataset
        print("No checkpoint provided. Creating demo model...")
        _, tokenizer = create_dataloader(
            dataset_name='simple_sentences',
            split='train',
            batch_size=1,
            max_samples=100
        )
        
        model = create_quantum_text_model(
            model_type=args.model,
            vocab_size=tokenizer.get_vocab_size(),
            device=device,
            hidden_dim=64,
            n_qubits=4
        )
        print(f"Created demo {args.model} model")
    
    # Create generator and start interactive session
    generator = QuantumTextGenerator_Module(model, tokenizer, device)
    generator.interactive_generation()

def evaluate_generation(args) -> None:
    """Evaluate model generation against ground truth"""
    
    device = setup_device()
    
    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(args.checkpoint), 'tokenizer.json')
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}")
        return
    
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    
    # Recreate tokenizer
    from data.datasets import SimpleTokenizer
    tokenizer = SimpleTokenizer(tokenizer_data['vocab_size'])
    tokenizer.word_to_id = tokenizer_data['word_to_id']
    tokenizer.id_to_word = {int(k): v for k, v in tokenizer_data['id_to_word'].items()}
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_type = checkpoint.get('model_type', args.model)
    
    # Create model
    model_kwargs = {
        'device': device,
        'hidden_dim': 128,
        'n_qubits': 6
    }
    
    model = create_quantum_text_model(
        model_type=model_type,
        vocab_size=tokenizer.get_vocab_size(),
        **model_kwargs
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load validation dataset for ground truth
    val_dataloader, _ = create_dataloader(
        dataset_name=args.dataset,
        split='val',
        batch_size=1,
        max_samples=50,  # Evaluate on 50 samples
        tokenizer=tokenizer
    )
    
    # Create generator
    generator = QuantumTextGenerator_Module(model, tokenizer, device)
    
    print(f"\nEvaluating {model_type} on {args.dataset} dataset")
    print("=" * 60)
    
    total_samples = 0
    total_bleu = 0.0
    
    for i, batch in enumerate(val_dataloader):
        if i >= 10:  # Limit to 10 samples for display
            break
            
        input_ids = batch['input_ids'][0]  # Remove batch dimension
        target_ids = batch['target_ids'][0]
        
        # Convert input to text (first few tokens as prompt)
        prompt_length = min(5, len(input_ids))
        prompt_tokens = input_ids[:prompt_length]
        prompt_text = ' '.join([tokenizer.id_to_word.get(int(token_id), '<UNK>') 
                               for token_id in prompt_tokens])
        
        # Ground truth (target text)
        ground_truth = ' '.join([tokenizer.id_to_word.get(int(token_id), '<UNK>') 
                                for token_id in target_ids])
        
        # Generate text
        generated_texts = generator.generate(
            prompt=prompt_text,
            max_length=len(target_ids) + prompt_length,
            temperature=0.8,
            do_sample=True,
            num_return_sequences=1
        )
        
        generated_text = generated_texts[0] if generated_texts else ""
        
        # Simple BLEU-like score (word overlap)
        ground_truth_words = set(ground_truth.lower().split())
        generated_words = set(generated_text.lower().split())
        
        if len(ground_truth_words) > 0:
            overlap = len(ground_truth_words.intersection(generated_words))
            bleu_like = overlap / len(ground_truth_words)
        else:
            bleu_like = 0.0
        
        total_bleu += bleu_like
        total_samples += 1
        
        # Display comparison
        print(f"\nSample {i+1}:")
        print(f"Prompt: {prompt_text}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Generated: {generated_text}")
        print(f"Word Overlap Score: {bleu_like:.3f}")
        print("-" * 60)
    
    # Calculate average scores
    avg_bleu = total_bleu / total_samples if total_samples > 0 else 0.0
    
    print(f"\nEvaluation Summary:")
    print(f"Average Word Overlap Score: {avg_bleu:.3f}")
    print(f"Samples evaluated: {total_samples}")
    
    # Save evaluation results
    eval_results = {
        'model': model_type,
        'dataset': args.dataset,
        'avg_word_overlap': avg_bleu,
        'samples_evaluated': total_samples
    }
    
    eval_output_path = os.path.join(os.path.dirname(args.checkpoint), 'evaluation_results.json')
    with open(eval_output_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"Results saved to: {eval_output_path}")

def evaluate_model(args) -> None:
    """Evaluate a trained model"""
    
    device = setup_device()
    
    # Load model and tokenizer (similar to generate_text)
    tokenizer_path = os.path.join(os.path.dirname(args.checkpoint), 'tokenizer.json')
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    
    from data.datasets import SimpleTokenizer
    tokenizer = SimpleTokenizer(tokenizer_data['vocab_size'])
    tokenizer.word_to_id = tokenizer_data['word_to_id']
    tokenizer.id_to_word = {int(k): v for k, v in tokenizer_data['id_to_word'].items()}
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_type = checkpoint.get('model_type', args.model)
    
    model = create_quantum_text_model(
        model_type=model_type,
        vocab_size=tokenizer.get_vocab_size(),
        device=device
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test data
    test_loader, _ = create_dataloader(
        dataset_name=args.dataset,
        split='test',
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Generate text samples
    generator = QuantumTextGenerator_Module(model, tokenizer, device)
    evaluator = GenerationEvaluator()
    
    # Generate samples from test prompts
    test_prompts = []
    for batch in test_loader:
        for input_ids in batch['input_ids']:
            # Use first few tokens as prompt
            prompt_ids = input_ids[:5]  # Use first 5 tokens as prompt
            prompt = tokenizer.decode(prompt_ids.tolist())
            test_prompts.append(prompt)
            
            if len(test_prompts) >= 50:  # Evaluate on 50 samples
                break
        if len(test_prompts) >= 50:
            break
    
    print(f"Generating {len(test_prompts)} samples for evaluation...")
    
    generated_texts = generator.batch_generate(
        test_prompts,
        batch_size=4,
        max_length=args.max_length,
        temperature=args.temperature,
        do_sample=args.do_sample
    )
    
    # Evaluate
    metrics = evaluator.evaluate(generated_texts)
    
    print("\nEvaluation Results:")
    print("-" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Quantum Text Generation")
    
    # Main arguments
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'generate', 'interactive', 'evaluate'],
                       help='Mode of operation')
    parser.add_argument('--model', type=str, default='qasa',
                       choices=['qasa', 'qksan', 'qrwkv'],
                       help='Quantum model to use')
    parser.add_argument('--dataset', type=str, default='simple_sentences',
                       choices=list_available_datasets(),
                       help='Dataset to use')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension size')
    parser.add_argument('--n_qubits', type=int, default=6,
                       help='Number of qubits')
    parser.add_argument('--n_layers', type=int, default=4,
                       help='Number of layers (for QRWKV)')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads (for QRWKV)')
    parser.add_argument('--qksan_variant', type=str, default='PCAAmHE',
                       help='QKSAN model variant')
    
    # Data arguments
    parser.add_argument('--vocab_size', type=int, default=1000,
                       help='Vocabulary size')
    parser.add_argument('--max_length', type=int, default=64,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'linear', 'constant'],
                       help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--eval_every', type=int, default=5,
                       help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Generation arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='',
                       help='Text prompt for generation')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Nucleus sampling')
    parser.add_argument('--do_sample', action='store_true',
                       help='Use sampling instead of greedy decoding')
    parser.add_argument('--num_sequences', type=int, default=1,
                       help='Number of sequences to generate')
    
    # Directory arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory to save logs')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Execute based on mode
    try:
        if args.mode == 'train':
            results = train_model(args)
            print(f"\nTraining completed successfully!")
            print(f"Best model saved at: {results['model_path']}")
            
        elif args.mode == 'generate':
            if not args.checkpoint:
                print("Error: --checkpoint required for generation mode")
                return
            generate_text(args)
            
        elif args.mode == 'interactive':
            interactive_mode(args)
            
        elif args.mode == 'evaluate':
            if not args.checkpoint:
                print("Error: --checkpoint required for evaluation mode")
                return
            if not args.dataset:
                print("Error: --dataset required for evaluation mode")
                return
            evaluate_generation(args)
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
