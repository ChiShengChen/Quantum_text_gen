#!/usr/bin/env python3

import os
import re
import json
import csv

def extract_metrics_from_log(log_file):
    """Extract metrics from output.log file"""
    metrics = {
        'epochs_completed': 0,
        'training_time': 'N/A',
        'total_params': 'N/A',
        'vocab_size': 'N/A',
        'train_samples': 'N/A',
        'val_samples': 'N/A',
        'best_val_loss': 'N/A',
        'final_train_loss': 'N/A',
        'final_val_loss': 'N/A',
        'final_train_ppl': 'N/A',
        'final_val_ppl': 'N/A'
    }
    
    if not os.path.exists(log_file):
        return metrics
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Count epochs
    epoch_matches = re.findall(r'INFO - Epoch \d+:', content)
    metrics['epochs_completed'] = len(epoch_matches)
    
    # Training time
    time_match = re.search(r'Training completed in ([\d.]+) seconds', content)
    if time_match:
        metrics['training_time'] = time_match.group(1)
    
    # Total parameters
    params_match = re.search(r'Total parameters: ([\d,]+)', content)
    if params_match:
        metrics['total_params'] = params_match.group(1).replace(',', '')
    
    # Vocabulary size
    vocab_match = re.search(r'Vocabulary size: (\d+)', content)
    if vocab_match:
        metrics['vocab_size'] = vocab_match.group(1)
    
    # Train samples
    train_match = re.search(r'Train samples: (\d+)', content)
    if train_match:
        metrics['train_samples'] = train_match.group(1)
    
    # Validation samples
    val_match = re.search(r'Validation samples: (\d+)', content)
    if val_match:
        metrics['val_samples'] = val_match.group(1)
    
    # Best validation loss
    best_val_match = re.search(r'Best validation loss: ([\d.]+)', content)
    if best_val_match:
        metrics['best_val_loss'] = best_val_match.group(1)
    
    # Final train loss and PPL
    train_matches = re.findall(r'Train Loss: ([\d.]+), Train PPL: ([\d.]+)', content)
    if train_matches:
        final_train = train_matches[-1]
        metrics['final_train_loss'] = final_train[0]
        metrics['final_train_ppl'] = final_train[1]
    
    # Final validation loss and PPL
    val_matches = re.findall(r'Val Loss: ([\d.]+), Val PPL: ([\d.]+)', content)
    if val_matches:
        final_val = val_matches[-1]
        metrics['final_val_loss'] = final_val[0]
        metrics['final_val_ppl'] = final_val[1]
    
    return metrics

def extract_generation_sample(gen_file):
    """Extract generation sample from generation_test.txt"""
    if not os.path.exists(gen_file):
        return "N/A"
    
    with open(gen_file, 'r') as f:
        content = f.read()
    
    # Look for generated text patterns
    patterns = [
        r'^(\d+): (.+)$',  # "1: generated text"
        r'Generated: (.+)$',  # "Generated: text"
        r'Generation \d+: (.+)$'  # "Generation 1: text"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            # Get the last group from the match
            groups = match.groups()
            if groups:
                text = groups[-1].strip()
                return text[:100].replace(',', ';')  # Limit to 100 chars and escape commas
    
    return "N/A"

def extract_word_overlap(eval_file):
    """Extract word overlap score from evaluation_results.json"""
    if not os.path.exists(eval_file):
        return "N/A"
    
    try:
        with open(eval_file, 'r') as f:
            data = json.load(f)
        return str(data.get('avg_word_overlap', 'N/A'))
    except:
        return "N/A"

def main():
    base_dir = "./experiments"
    output_csv = os.path.join(base_dir, "experiment_results.csv")
    
    models = ["qasa", "qksan", "qrwkv"]
    datasets = ["simple_sentences", "short_stories", "quantum_phrases", "haiku", "proverbs"]
    
    # CSV header
    header = [
        "Model", "Dataset", "Status", "Epochs_Completed", "Best_Epoch", 
        "Best_Val_Loss", "Best_Train_Loss", "Best_Train_PPL", "Best_Val_PPL",
        "Final_Train_Loss", "Final_Val_Loss", "Training_Time_Seconds",
        "Total_Parameters", "Vocab_Size", "Train_Samples", "Val_Samples",
        "Generation_Sample", "Word_Overlap_Score", "Experiment_Dir"
    ]
    
    rows = []
    
    for model in models:
        for dataset in datasets:
            exp_dir = os.path.join(base_dir, model, dataset)
            
            # Check if experiment exists
            if not os.path.exists(exp_dir):
                continue
            
            # Check status
            status_file = os.path.join(exp_dir, "status.txt")
            status = "FAILED"
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status_content = f.read().strip()
                    if status_content == "SUCCESS":
                        status = "SUCCESS"
            
            # Extract metrics
            log_file = os.path.join(exp_dir, "output.log")
            metrics = extract_metrics_from_log(log_file)
            
            # Extract generation sample
            gen_file = os.path.join(exp_dir, "generation_test.txt")
            generation_sample = extract_generation_sample(gen_file)
            
            # Extract word overlap
            eval_file = os.path.join(exp_dir, "evaluation_results.json")
            word_overlap = extract_word_overlap(eval_file)
            
            # Calculate best epoch (approximate as final epoch)
            best_epoch = max(0, metrics['epochs_completed'] - 1) if metrics['epochs_completed'] > 0 else 0
            
            row = [
                model,
                dataset,
                status,
                metrics['epochs_completed'],
                best_epoch,
                metrics['best_val_loss'],
                metrics['final_train_loss'],  # Use final as best approximation
                metrics['final_train_ppl'],
                metrics['final_val_ppl'],
                metrics['final_train_loss'],
                metrics['final_val_loss'],
                metrics['training_time'],
                metrics['total_params'],
                metrics['vocab_size'],
                metrics['train_samples'],
                metrics['val_samples'],
                generation_sample,
                word_overlap,
                exp_dir
            ]
            
            rows.append(row)
            print(f"Processed: {model} + {dataset} (Status: {status}, Epochs: {metrics['epochs_completed']}, Val Loss: {metrics['best_val_loss']})")
    
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"\nFixed CSV saved to: {output_csv}")
    print(f"Total experiments processed: {len(rows)}")

if __name__ == "__main__":
    main()