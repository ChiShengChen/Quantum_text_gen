#!/usr/bin/env python3
"""
QKSAN Model Fix Experiment
Run comprehensive experiments with the fixed QKSAN model
"""

import os
import sys
import torch
import torch.nn as nn
import json
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qksan_fixed import create_fixed_qksan_model
from data.datasets import create_dataloader
from training.trainer import QuantumTextTrainer, create_optimizer, create_scheduler
from utils.logger import Logger

def run_qksan_fix_experiment(dataset_name='simple_sentences', 
                           epochs=10, 
                           batch_size=16,
                           learning_rate=0.001,
                           device='cpu'):
    """
    Run QKSAN fix experiment
    """
    
    print(f"QKSAN Fix Experiment - {dataset_name}")
    print("=" * 60)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"./experiments/qksan_fixed/{dataset_name}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Setup logger
    logger = Logger(exp_dir, f"qksan_fixed_{dataset_name}")
    
    # Load dataset
    print("Loading dataset...")
    train_loader, tokenizer = create_dataloader(
        dataset_name=dataset_name,
        split='train',
        batch_size=batch_size,
        max_length=32,
        vocab_size=500,
        max_samples=1000,
        shuffle=True
    )
    
    val_loader, _ = create_dataloader(
        dataset_name=dataset_name,
        split='val',
        batch_size=batch_size,
        max_length=32,
        vocab_size=500,
        max_samples=200,
        shuffle=False
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Create fixed QKSAN model
    print("Creating fixed QKSAN model...")
    model = create_fixed_qksan_model(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=128,
        n_qubits=6,
        device=device
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model, 
        optimizer_type='adam',
        learning_rate=learning_rate,
        weight_decay=1e-4
    )
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_type='cosine',
        num_epochs=epochs
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
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    
    training_history = trainer.train(
        epochs=epochs,
        save_dir=exp_dir,
        eval_every=1,
        save_every=5,
        early_stopping_patience=5
    )
    
    training_time = time.time() - start_time
    
    # Save results
    results = {
        'dataset': dataset_name,
        'model': 'qksan_fixed',
        'epochs_completed': len(training_history['train_loss']),
        'best_val_loss': trainer.best_val_loss,
        'final_train_loss': training_history['train_loss'][-1] if training_history['train_loss'] else float('inf'),
        'final_val_loss': training_history['val_loss'][-1] if training_history['val_loss'] else float('inf'),
        'training_time_seconds': training_time,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'vocab_size': tokenizer.get_vocab_size(),
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'status': 'SUCCESS' if trainer.best_val_loss < float('inf') else 'FAILED',
        'training_history': training_history
    }
    
    # Save results
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save tokenizer
    tokenizer_path = os.path.join(exp_dir, 'tokenizer.json')
    with open(tokenizer_path, 'w') as f:
        json.dump({
            'word_to_id': tokenizer.word_to_id,
            'id_to_word': tokenizer.id_to_word,
            'vocab_size': tokenizer.get_vocab_size()
        }, f, indent=2)
    
    # Test generation
    print("Testing text generation...")
    try:
        model.eval()
        test_prompts = ["the", "birds", "water"]
        
        generated_texts = []
        for prompt in test_prompts:
            # Tokenize prompt
            prompt_tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([prompt_tokens], device=device)
            
            # Generate
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    max_length=10,
                    temperature=1.0,
                    do_sample=True
                )
            
            # Decode
            generated_text = tokenizer.decode(generated[0].tolist())
            generated_texts.append(f"Prompt: '{prompt}' -> Generated: '{generated_text}'")
        
        # Save generation results
        with open(os.path.join(exp_dir, 'generation_test.txt'), 'w') as f:
            f.write("QKSAN Fixed Model Generation Test\n")
            f.write("=" * 40 + "\n\n")
            for text in generated_texts:
                f.write(text + "\n")
        
        print("Generation test completed!")
        
    except Exception as e:
        print(f"Generation test failed: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Model: QKSAN Fixed")
    print(f"Status: {results['status']}")
    print(f"Epochs completed: {results['epochs_completed']}")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Final training loss: {results['final_train_loss']:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Total parameters: {results['total_parameters']:,}")
    print(f"Experiment directory: {exp_dir}")
    
    return results

def run_multiple_datasets():
    """Run experiments on multiple datasets"""
    
    datasets = ['simple_sentences', 'short_stories', 'quantum_phrases', 'haiku', 'proverbs']
    
    all_results = []
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Running experiment on {dataset}")
        print(f"{'='*80}")
        
        try:
            results = run_qksan_fix_experiment(
                dataset_name=dataset,
                epochs=10,
                batch_size=16,
                learning_rate=0.001,
                device='cpu'
            )
            all_results.append(results)
        except Exception as e:
            print(f"Experiment failed for {dataset}: {e}")
            all_results.append({
                'dataset': dataset,
                'status': 'FAILED',
                'error': str(e)
            })
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_results_path = f"./experiments/qksan_fixed_combined_{timestamp}.json"
    
    with open(combined_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    
    successful = [r for r in all_results if r.get('status') == 'SUCCESS']
    failed = [r for r in all_results if r.get('status') != 'SUCCESS']
    
    print(f"Successful experiments: {len(successful)}/{len(all_results)}")
    print(f"Failed experiments: {len(failed)}/{len(all_results)}")
    
    if successful:
        print("\nSuccessful experiments:")
        for result in successful:
            print(f"  - {result['dataset']}: Best val loss = {result['best_val_loss']:.4f}")
    
    if failed:
        print("\nFailed experiments:")
        for result in failed:
            print(f"  - {result['dataset']}: {result.get('error', 'Unknown error')}")
    
    print(f"\nCombined results saved to: {combined_results_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run QKSAN fix experiments')
    parser.add_argument('--dataset', type=str, default='simple_sentences',
                       help='Dataset to use for single experiment')
    parser.add_argument('--all_datasets', action='store_true',
                       help='Run experiments on all datasets')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    if args.all_datasets:
        run_multiple_datasets()
    else:
        run_qksan_fix_experiment(
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device
        )
