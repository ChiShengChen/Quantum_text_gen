#!/bin/bash

# Quantum Text Generation - Enhanced Experiments Runner with CSV logging and plotting
# This script runs training experiments for all model-dataset combinations and generates analysis

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
EPOCHS=30
BATCH_SIZE=8
MAX_SAMPLES=300
LEARNING_RATE=0.005
BASE_DIR="./experiments"
RESULTS_CSV="$BASE_DIR/experiment_results.csv"
PLOTS_DIR="$BASE_DIR/plots"

# Arrays of models and datasets
MODELS=("qasa" "qksan" "qrwkv")
DATASETS=("simple_sentences" "short_stories" "quantum_phrases" "haiku" "proverbs")

# Model-specific parameters
declare -A MODEL_PARAMS
MODEL_PARAMS[qasa]="--hidden_dim 128 --n_qubits 6"
MODEL_PARAMS[qksan]="--hidden_dim 64 --n_qubits 6 --qksan_variant PCAAmHE"
MODEL_PARAMS[qrwkv]="--hidden_dim 256 --n_qubits 4 --n_layers 4 --n_heads 8"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to initialize CSV file
initialize_csv() {
    mkdir -p "$BASE_DIR"
    mkdir -p "$PLOTS_DIR"
    
    # Create CSV header
    cat > "$RESULTS_CSV" << EOF
Model,Dataset,Status,Epochs_Completed,Best_Epoch,Best_Val_Loss,Best_Train_Loss,Best_Train_PPL,Best_Val_PPL,Final_Train_Loss,Final_Val_Loss,Training_Time_Seconds,Total_Parameters,Vocab_Size,Train_Samples,Val_Samples,Generation_Sample,Word_Overlap_Score,Experiment_Dir
EOF
    
    print_status "Initialized results CSV: $RESULTS_CSV"
}

# Function to extract metrics from training history
extract_metrics() {
    local exp_dir=$1
    local model=$2
    local dataset=$3
    
    local training_history="$exp_dir/training_history.json"
    local output_log="$exp_dir/output.log"
    
    # Default values
    local status="FAILED"
    local epochs_completed=0
    local best_epoch=0
    local best_val_loss="N/A"
    local best_train_loss="N/A"
    local best_train_ppl="N/A"
    local best_val_ppl="N/A"
    local final_train_loss="N/A"
    local final_val_loss="N/A"
    local training_time=0
    local total_params="N/A"
    local vocab_size="N/A"
    local train_samples="N/A"
    local val_samples="N/A"
    local generation_sample="N/A"
    
    # Check if training completed successfully
    if [ -f "$exp_dir/status.txt" ] && [ "$(cat $exp_dir/status.txt)" = "SUCCESS" ]; then
        status="SUCCESS"
        
        # Extract from output log
        if [ -f "$output_log" ]; then
            epochs_completed=$(grep -c "INFO - Epoch [0-9]*:" "$output_log" || echo "0")
            training_time=$(grep "Training completed in" "$output_log" | awk '{print $10}' || echo "0")
            total_params=$(grep "Total parameters:" "$output_log" | awk '{print $10}' | tr -d ',' || echo "N/A")
            vocab_size=$(grep "Vocabulary size:" "$output_log" | awk '{print $9}' || echo "N/A")
            train_samples=$(grep "Train samples:" "$output_log" | awk '{print $9}' || echo "N/A")
            val_samples=$(grep "Validation samples:" "$output_log" | awk '{print $9}' || echo "N/A")
            
            # Get best validation loss (field 13: "Best validation loss: 4.3764")
            best_val_loss=$(grep "Best validation loss:" "$output_log" | tail -1 | awk '{print $13}' || echo "N/A")
            
            # Get final losses and PPL
            final_train_loss=$(grep "Train Loss:" "$output_log" | tail -1 | awk '{print $12}' | tr -d ',' || echo "N/A")
            final_val_loss=$(grep "Val Loss:" "$output_log" | tail -1 | awk '{print $12}' | tr -d ',' || echo "N/A")
            final_train_ppl=$(grep "Train PPL:" "$output_log" | tail -1 | awk '{print $15}' || echo "N/A")
            final_val_ppl=$(grep "Val PPL:" "$output_log" | tail -1 | awk '{print $15}' || echo "N/A")
            
            # For best metrics, use the final values as approximation
            if [ "$best_val_loss" != "N/A" ]; then
                best_train_loss="$final_train_loss"
                best_train_ppl="$final_train_ppl"
                best_val_ppl="$final_val_ppl"
                # Calculate best epoch (final epoch - 1)
                best_epoch=$((epochs_completed - 1))
            fi
        fi
        
        # Extract generation sample
        if [ -f "$exp_dir/generation_test.txt" ]; then
            generation_sample=$(grep "Generated:" "$exp_dir/generation_test.txt" | head -1 | cut -d':' -f2- | cut -c1-100 | tr -d '\n' | tr ',' ';' || echo "N/A")
        fi
        
        # Extract evaluation metrics
        local word_overlap="N/A"
        if [ -f "$exp_dir/evaluation_results.json" ]; then
            word_overlap=$(grep "avg_word_overlap" "$exp_dir/evaluation_results.json" | awk -F':' '{print $2}' | tr -d ' ,' || echo "N/A")
        fi
    fi
    
    # Write to CSV
    echo "$model,$dataset,$status,$epochs_completed,$best_epoch,$best_val_loss,$best_train_loss,$best_train_ppl,$best_val_ppl,$final_train_loss,$final_val_loss,$training_time,$total_params,$vocab_size,$train_samples,$val_samples,\"$generation_sample\",$word_overlap,$exp_dir" >> "$RESULTS_CSV"
}

# Function to create training plots
create_training_plots() {
    local exp_dir=$1
    local model=$2
    local dataset=$3
    
    local training_history="$exp_dir/training_history.json"
    local plot_script="$exp_dir/create_plots.py"
    
    if [ ! -f "$training_history" ]; then
        print_warning "No training history found for $model + $dataset"
        return 1
    fi
    
    # Create Python script to generate plots
    cat > "$plot_script" << 'EOF'
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def create_training_plots(history_file, model_name, dataset_name, output_dir):
    """Create training loss and perplexity plots"""
    
    # Load training history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    if not history.get('train_losses') or not history.get('val_losses'):
        print(f"No training data found in {history_file}")
        return
    
    # Extract data
    epochs = list(range(len(history['train_losses'])))
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    train_ppls = history.get('train_ppls', [])
    val_ppls = history.get('val_ppls', [])
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name.upper()} on {dataset_name.replace("_", " ").title()}', fontsize=16)
    
    # Plot 1: Training and Validation Loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses and any(v != 0 for v in val_losses):
        val_epochs = [i for i, v in enumerate(val_losses) if v != 0]
        val_losses_clean = [v for v in val_losses if v != 0]
        ax1.plot(val_epochs, val_losses_clean, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss (detailed)
    ax2.plot(epochs, train_losses, 'b-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss Progress')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Perplexity (if available)
    if train_ppls:
        ax3.plot(epochs, train_ppls, 'g-', label='Train PPL', linewidth=2)
        if val_ppls and any(v != 0 for v in val_ppls):
            val_ppl_epochs = [i for i, v in enumerate(val_ppls) if v != 0]
            val_ppls_clean = [v for v in val_ppls if v != 0]
            ax3.plot(val_ppl_epochs, val_ppls_clean, 'orange', label='Val PPL', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Perplexity')
        ax3.set_title('Training and Validation Perplexity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')  # Log scale for better visualization
    else:
        ax3.text(0.5, 0.5, 'No Perplexity Data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Perplexity (No Data)')
    
    # Plot 4: Loss improvement rate
    if len(train_losses) > 1:
        loss_improvements = [train_losses[i-1] - train_losses[i] for i in range(1, len(train_losses))]
        ax4.plot(epochs[1:], loss_improvements, 'purple', linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Improvement')
        ax4.set_title('Training Loss Improvement per Epoch')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Loss Improvement (Insufficient Data)')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save individual plot
    plot_filename = f'{model_name}_{dataset_name}_training_plots.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    
    # Also save in experiment directory
    exp_plot_path = os.path.join(os.path.dirname(history_file), 'training_plots.png')
    plt.savefig(exp_plot_path, dpi=300, bbox_inches='tight')
    
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python create_plots.py <history_file> <model_name> <dataset_name> <output_dir>")
        sys.exit(1)
    
    history_file = sys.argv[1]
    model_name = sys.argv[2]
    dataset_name = sys.argv[3]
    output_dir = sys.argv[4]
    
    create_training_plots(history_file, model_name, dataset_name, output_dir)
EOF

    # Run the plotting script
    if python "$plot_script" "$training_history" "$model" "$dataset" "$PLOTS_DIR"; then
        print_success "Created training plots for $model + $dataset"
    else
        print_warning "Failed to create plots for $model + $dataset"
    fi
    
    # Clean up the temporary script
    rm -f "$plot_script"
}

# Function to run a single experiment
run_experiment() {
    local model=$1
    local dataset=$2
    local exp_dir="${BASE_DIR}/${model}/${dataset}"
    
    print_status "Starting experiment: ${model} on ${dataset}"
    
    # Create experiment directory
    mkdir -p "$exp_dir"
    
    # Build command
    local cmd="python quantum_text_gen.py \
        --mode train \
        --model $model \
        --dataset $dataset \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --max_samples $MAX_SAMPLES \
        --learning_rate $LEARNING_RATE \
        --checkpoint_dir $exp_dir \
        --log_dir $exp_dir \
        ${MODEL_PARAMS[$model]}"
    
    # Log the command
    echo "Command: $cmd" > "$exp_dir/command.log"
    
    # Run the experiment
    print_status "Running: $model + $dataset"
    echo "Starting training - you will see epoch progress below:"
    if eval $cmd 2>&1 | tee "$exp_dir/output.log"; then
        print_success "Completed: $model + $dataset"
        echo "SUCCESS" > "$exp_dir/status.txt"
        
        # Test generation with the trained model
        print_status "Testing generation for: $model + $dataset"
        local test_prompt="the"
        if [[ $dataset == "quantum_phrases" ]]; then
            test_prompt="quantum"
        elif [[ $dataset == "haiku" ]]; then
            test_prompt="spring"
        elif [[ $dataset == "proverbs" ]]; then
            test_prompt="time"
        fi
        
        echo "Testing text generation..."
        python quantum_text_gen.py \
            --mode generate \
            --checkpoint "$exp_dir/best_model.pth" \
            --prompt "$test_prompt" \
            --max_length 30 \
            --temperature 1.0 \
            --do_sample \
            --num_sequences 3 2>&1 | tee "$exp_dir/generation_test.txt"
            
        # Evaluate generation against ground truth
        echo "Evaluating generation quality..."
        python quantum_text_gen.py \
            --mode evaluate \
            --checkpoint "$exp_dir/best_model.pth" \
            --dataset "$dataset" 2>&1 | tee "$exp_dir/evaluation_results.txt"
            
        if [ $? -eq 0 ]; then
            print_success "Generation test completed for: $model + $dataset"
        else
            print_warning "Generation test failed for: $model + $dataset"
        fi
        
        # Extract metrics and add to CSV
        extract_metrics "$exp_dir" "$model" "$dataset"
        
        # Create training plots
        create_training_plots "$exp_dir" "$model" "$dataset"
        
    else
        print_error "Failed: $model + $dataset"
        echo "FAILED" > "$exp_dir/status.txt"
        # Still extract what we can for failed experiments
        extract_metrics "$exp_dir" "$model" "$dataset"
        return 1
    fi
    
    return 0
}

# Function to create summary plots
create_summary_plots() {
    print_status "Creating summary analysis plots..."
    
    local summary_script="$BASE_DIR/create_summary_plots.py"
    
    cat > "$summary_script" << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

def create_summary_plots(csv_file, plots_dir):
    """Create summary plots from experiment results"""
    
    # Load results
    df = pd.read_csv(csv_file)
    
    # Filter successful experiments
    df_success = df[df['Status'] == 'SUCCESS'].copy()
    
    if len(df_success) == 0:
        print("No successful experiments found!")
        return
    
    # Convert numeric columns
    numeric_cols = ['Best_Val_Loss', 'Best_Train_Loss', 'Best_Train_PPL', 'Best_Val_PPL', 
                   'Training_Time_Seconds', 'Total_Parameters', 'Vocab_Size']
    
    for col in numeric_cols:
        df_success[col] = pd.to_numeric(df_success[col], errors='coerce')
    
    # Create summary plots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Best Validation Loss by Model and Dataset
    plt.subplot(3, 3, 1)
    if not df_success['Best_Val_Loss'].isna().all():
        pivot_val_loss = df_success.pivot(index='Dataset', columns='Model', values='Best_Val_Loss')
        sns.heatmap(pivot_val_loss, annot=True, fmt='.3f', cmap='viridis_r')
        plt.title('Best Validation Loss\n(Lower is Better)')
        plt.ylabel('Dataset')
    else:
        plt.text(0.5, 0.5, 'No Validation Loss Data', ha='center', va='center')
        plt.title('Best Validation Loss (No Data)')
    
    # Plot 2: Training Time by Model and Dataset
    plt.subplot(3, 3, 2)
    if not df_success['Training_Time_Seconds'].isna().all():
        pivot_time = df_success.pivot(index='Dataset', columns='Model', values='Training_Time_Seconds')
        sns.heatmap(pivot_time, annot=True, fmt='.0f', cmap='plasma')
        plt.title('Training Time (Seconds)')
        plt.ylabel('Dataset')
    else:
        plt.text(0.5, 0.5, 'No Training Time Data', ha='center', va='center')
        plt.title('Training Time (No Data)')
    
    # Plot 3: Model Parameters
    plt.subplot(3, 3, 3)
    if not df_success['Total_Parameters'].isna().all():
        model_params = df_success.groupby('Model')['Total_Parameters'].first()
        bars = plt.bar(model_params.index, model_params.values)
        plt.title('Model Size (Total Parameters)')
        plt.ylabel('Parameters')
        plt.xticks(rotation=45)
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'No Parameters Data', ha='center', va='center')
        plt.title('Model Size (No Data)')
    
    # Plot 4: Best Train PPL by Model
    plt.subplot(3, 3, 4)
    if not df_success['Best_Train_PPL'].isna().all():
        df_success.boxplot(column='Best_Train_PPL', by='Model', ax=plt.gca())
        plt.title('Training Perplexity Distribution by Model')
        plt.suptitle('')  # Remove automatic title
        plt.ylabel('Perplexity (log scale)')
        plt.yscale('log')
    else:
        plt.text(0.5, 0.5, 'No PPL Data', ha='center', va='center')
        plt.title('Training Perplexity (No Data)')
    
    # Plot 5: Success Rate by Model
    plt.subplot(3, 3, 5)
    success_rate = df.groupby('Model')['Status'].apply(lambda x: (x == 'SUCCESS').mean() * 100)
    bars = plt.bar(success_rate.index, success_rate.values)
    plt.title('Success Rate by Model')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Plot 6: Success Rate by Dataset
    plt.subplot(3, 3, 6)
    success_rate_dataset = df.groupby('Dataset')['Status'].apply(lambda x: (x == 'SUCCESS').mean() * 100)
    bars = plt.bar(range(len(success_rate_dataset)), success_rate_dataset.values)
    plt.title('Success Rate by Dataset')
    plt.ylabel('Success Rate (%)')
    plt.xticks(range(len(success_rate_dataset)), success_rate_dataset.index, rotation=45)
    plt.ylim(0, 100)
    # Add percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Plot 7: Training vs Validation Loss Correlation
    plt.subplot(3, 3, 7)
    if not (df_success['Best_Train_Loss'].isna().all() or df_success['Best_Val_Loss'].isna().all()):
        plt.scatter(df_success['Best_Train_Loss'], df_success['Best_Val_Loss'], 
                   c=df_success['Model'].astype('category').cat.codes, alpha=0.7)
        plt.xlabel('Best Training Loss')
        plt.ylabel('Best Validation Loss')
        plt.title('Training vs Validation Loss')
        
        # Add model legend
        models = df_success['Model'].unique()
        for i, model in enumerate(models):
            plt.scatter([], [], c=plt.cm.tab10(i), label=model)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Insufficient Loss Data', ha='center', va='center')
        plt.title('Training vs Validation Loss (No Data)')
    
    # Plot 8: Dataset Vocabulary Size
    plt.subplot(3, 3, 8)
    if not df_success['Vocab_Size'].isna().all():
        vocab_by_dataset = df_success.groupby('Dataset')['Vocab_Size'].first()
        bars = plt.bar(range(len(vocab_by_dataset)), vocab_by_dataset.values)
        plt.title('Vocabulary Size by Dataset')
        plt.ylabel('Vocabulary Size')
        plt.xticks(range(len(vocab_by_dataset)), vocab_by_dataset.index, rotation=45)
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'No Vocab Data', ha='center', va='center')
        plt.title('Vocabulary Size (No Data)')
    
    # Plot 9: Training Samples by Dataset
    plt.subplot(3, 3, 9)
    if not df_success['Train_Samples'].isna().all():
        samples_by_dataset = df_success.groupby('Dataset')['Train_Samples'].first()
        bars = plt.bar(range(len(samples_by_dataset)), samples_by_dataset.values)
        plt.title('Training Samples by Dataset')
        plt.ylabel('Number of Samples')
        plt.xticks(range(len(samples_by_dataset)), samples_by_dataset.index, rotation=45)
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'No Sample Data', ha='center', va='center')
        plt.title('Training Samples (No Data)')
    
    plt.tight_layout()
    
    # Save summary plot
    summary_plot_path = os.path.join(plots_dir, 'experiment_summary.png')
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {summary_plot_path}")
    
    plt.close()
    
    # Create a detailed comparison table
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    if len(df_success) > 0:
        print(f"\nSuccessful Experiments: {len(df_success)}/{len(df)}")
        print("\nBest Results by Model:")
        for model in df_success['Model'].unique():
            model_data = df_success[df_success['Model'] == model]
            if not model_data['Best_Val_Loss'].isna().all():
                best_exp = model_data.loc[model_data['Best_Val_Loss'].idxmin()]
                print(f"  {model.upper()}: {best_exp['Dataset']} (Val Loss: {best_exp['Best_Val_Loss']:.4f})")
            else:
                print(f"  {model.upper()}: No validation loss data")
        
        print(f"\nDetailed Results:")
        print(df_success[['Model', 'Dataset', 'Best_Val_Loss', 'Best_Train_Loss', 'Training_Time_Seconds']].to_string(index=False))
    else:
        print("No successful experiments found!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_summary_plots.py <csv_file> <plots_dir>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    plots_dir = sys.argv[2]
    
    create_summary_plots(csv_file, plots_dir)
EOF

    # Run the summary plotting script
    if python "$summary_script" "$RESULTS_CSV" "$PLOTS_DIR"; then
        print_success "Created summary analysis plots"
    else
        print_warning "Failed to create summary plots"
    fi
    
    # Clean up the temporary script
    rm -f "$summary_script"
}

# Function to show progress
show_progress() {
    local current=$1
    local total=$2
    local model=$3
    local dataset=$4
    
    local percentage=$((current * 100 / total))
    local completed=$((current * 50 / total))
    local remaining=$((50 - completed))
    
    printf "\r${BLUE}Progress: [${GREEN}"
    printf "%${completed}s" | tr ' ' '='
    printf "${NC}${BLUE}"
    printf "%${remaining}s" | tr ' ' '-'
    printf "] ${percentage}%% (${current}/${total}) - ${model}+${dataset}${NC}"
}

# Main execution
main() {
    print_status "Starting Enhanced Quantum Text Generation Experiments"
    print_status "Models: ${MODELS[*]}"
    print_status "Datasets: ${DATASETS[*]}"
    print_status "Total experiments: $((${#MODELS[@]} * ${#DATASETS[@]}))"
    
    # Activate conda environment
    print_status "Activating qcl conda environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate qcl
    
    if [ $? -ne 0 ]; then
        print_error "Failed to activate qcl environment"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [ ! -f "quantum_text_gen.py" ]; then
        print_error "quantum_text_gen.py not found. Please run from the correct directory."
        exit 1
    fi
    
    # Initialize CSV and directories
    initialize_csv
    
    # Initialize counters
    local total_experiments=$((${#MODELS[@]} * ${#DATASETS[@]}))
    local current_experiment=0
    local successful_experiments=0
    local failed_experiments=0
    
    # Record start time
    local start_time=$(date +%s)
    
    # Run all experiments
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            current_experiment=$((current_experiment + 1))
            
            show_progress $current_experiment $total_experiments $model $dataset
            echo ""
            echo "========================================="
            echo "Experiment $current_experiment/$total_experiments: $model + $dataset"
            echo "========================================="
            
            if run_experiment "$model" "$dataset"; then
                successful_experiments=$((successful_experiments + 1))
            else
                failed_experiments=$((failed_experiments + 1))
            fi
            
            # Clear progress line and show status
            echo ""
        done
    done
    
    # Calculate total time
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    local hours=$((total_time / 3600))
    local minutes=$(((total_time % 3600) / 60))
    local seconds=$((total_time % 60))
    
    # Create summary analysis
    create_summary_plots
    
    # Print final summary
    echo ""
    print_status "All experiments completed!"
    print_success "Successful: $successful_experiments/$total_experiments"
    if [ $failed_experiments -gt 0 ]; then
        print_error "Failed: $failed_experiments/$total_experiments"
    fi
    print_status "Total time: ${hours}h ${minutes}m ${seconds}s"
    
    print_status "Results saved to:"
    print_status "  CSV: $RESULTS_CSV"
    print_status "  Plots: $PLOTS_DIR/"
    print_status "  Individual plots: In each experiment directory"
    
    print_status "Enhanced experiment runner finished!"
}

# Handle interruption
trap 'print_warning "Experiment interrupted by user"; exit 130' INT

# Check if script is run with arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Enhanced Quantum Text Generation Experiment Runner"
    echo ""
    echo "Features:"
    echo "  - Runs all model-dataset combinations"
    echo "  - Records results in CSV format"
    echo "  - Creates training plots for each experiment"
    echo "  - Generates summary analysis plots"
    echo "  - Tests text generation for each trained model"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --dry-run      Show what would be executed without running"
    echo ""
    echo "Output:"
    echo "  ./experiments/experiment_results.csv    - Detailed results"
    echo "  ./experiments/plots/                    - Summary plots"
    echo "  ./experiments/[model]/[dataset]/        - Individual results"
    exit 0
fi

if [ "$1" = "--dry-run" ]; then
    print_status "DRY RUN - Enhanced experiment runner would:"
    echo "1. Run training for all model-dataset combinations"
    echo "2. Record results in CSV: $RESULTS_CSV"
    echo "3. Create individual training plots for each experiment"
    echo "4. Generate summary analysis plots in: $PLOTS_DIR"
    echo "5. Test text generation for each trained model"
    echo ""
    print_status "Experiments that would be run:"
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            echo "  - $model + $dataset"
        done
    done
    exit 0
fi

# Run main function
main "$@"