#!/bin/bash

# Test script for analysis features
# This tests CSV logging and plotting with existing experiment data

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test with existing experiment data
BASE_DIR="./experiments"
RESULTS_CSV="$BASE_DIR/test_experiment_results.csv"
PLOTS_DIR="$BASE_DIR/test_plots"

print_status "Testing analysis features with existing data..."

# Create test directories
mkdir -p "$BASE_DIR"
mkdir -p "$PLOTS_DIR"

# Initialize CSV
cat > "$RESULTS_CSV" << EOF
Model,Dataset,Status,Epochs_Completed,Best_Epoch,Best_Val_Loss,Best_Train_Loss,Best_Train_PPL,Best_Val_PPL,Final_Train_Loss,Final_Val_Loss,Training_Time_Seconds,Total_Parameters,Vocab_Size,Train_Samples,Val_Samples,Generation_Sample,Experiment_Dir
EOF

print_status "Initialized test CSV: $RESULTS_CSV"

# Extract metrics from existing experiment (if available)
if [ -d "./experiments/qasa/simple_sentences" ]; then
    exp_dir="./experiments/qasa/simple_sentences"
    print_status "Found existing experiment data in $exp_dir"
    
    # Extract basic metrics
    if [ -f "$exp_dir/output.log" ]; then
        epochs_completed=$(grep -c "Epoch [0-9]*:" "$exp_dir/output.log" || echo "20")
        training_time=$(grep "Training completed in" "$exp_dir/output.log" | awk '{print $4}' || echo "8516")
        total_params=$(grep "Total parameters:" "$exp_dir/output.log" | awk '{print $3}' | tr -d ',' || echo "288298")
        vocab_size=$(grep "Vocabulary size:" "$exp_dir/output.log" | awk '{print $3}' || echo "45")
        train_samples=$(grep "Train samples:" "$exp_dir/output.log" | awk '{print $3}' || echo "160")
        val_samples=$(grep "Validation samples:" "$exp_dir/output.log" | awk '{print $3}' || echo "20")
        best_val_loss=$(grep "Best validation loss:" "$exp_dir/output.log" | tail -1 | awk '{print $4}' || echo "6.6039")
        
        # Add to CSV
        echo "qasa,simple_sentences,SUCCESS,$epochs_completed,5,$best_val_loss,1.3082,3.70,737.99,1.3082,10.4269,$training_time,$total_params,$vocab_size,$train_samples,$val_samples,\"The cat sat on the mat and looked around\",$exp_dir" >> "$RESULTS_CSV"
        
        print_success "Extracted metrics from existing experiment"
    fi
    
    # Test plotting if training history exists
    if [ -f "$exp_dir/training_history.json" ]; then
        print_status "Testing plot creation with existing training history..."
        
        # Create test plotting script
        cat > "$exp_dir/test_plots.py" << 'EOF'
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def create_test_plots(history_file, output_dir):
    """Create test training plots"""
    
    # Load training history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    if not history.get('train_losses'):
        print(f"No training data found in {history_file}")
        return False
    
    # Extract data
    epochs = list(range(len(history['train_losses'])))
    train_losses = history['train_losses']
    val_losses = history.get('val_losses', [])
    train_ppls = history.get('train_ppls', [])
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('QASA on Simple Sentences (Test)', fontsize=16)
    
    # Plot 1: Training Loss
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
    
    # Plot 3: Perplexity
    if train_ppls:
        ax3.plot(epochs, train_ppls, 'g-', label='Train PPL', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Perplexity')
        ax3.set_title('Training Perplexity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
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
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'test_training_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved test plot: {plot_path}")
    
    plt.close()
    return True

if __name__ == "__main__":
    history_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    success = create_test_plots(history_file, output_dir)
    sys.exit(0 if success else 1)
EOF

        # Run test plotting
        if python "$exp_dir/test_plots.py" "$exp_dir/training_history.json" "$PLOTS_DIR"; then
            print_success "Successfully created test training plots"
        else
            print_error "Failed to create test training plots"
        fi
        
        rm -f "$exp_dir/test_plots.py"
    else
        print_warning "No training history found for plotting test"
    fi
else
    print_warning "No existing experiment data found. Adding mock data for testing..."
    
    # Add mock data for testing
    echo "qasa,simple_sentences,SUCCESS,20,5,6.6039,1.3082,3.70,737.99,1.3082,10.4269,8516,288298,45,160,20,\"The cat sat on the mat\",./experiments/qasa/simple_sentences" >> "$RESULTS_CSV"
    echo "qksan,quantum_phrases,SUCCESS,15,8,4.2156,1.1234,3.08,67.23,1.1234,5.8901,6234,156789,65,80,10,\"Quantum entanglement creates superposition\",./experiments/qksan/quantum_phrases" >> "$RESULTS_CSV"
    echo "qrwkv,haiku,FAILED,3,0,N/A,2.4567,N/A,N/A,2.4567,N/A,1234,512000,35,50,5,\"N/A\",./experiments/qrwkv/haiku" >> "$RESULTS_CSV"
fi

# Test summary plots creation
print_status "Testing summary plots creation..."

cat > "$BASE_DIR/test_summary_plots.py" << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

def create_test_summary_plots(csv_file, plots_dir):
    """Create test summary plots"""
    
    # Load results
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} experiments from CSV")
    
    # Filter successful experiments
    df_success = df[df['Status'] == 'SUCCESS'].copy()
    print(f"Found {len(df_success)} successful experiments")
    
    if len(df_success) == 0:
        print("No successful experiments found for plotting!")
        return False
    
    # Convert numeric columns
    numeric_cols = ['Best_Val_Loss', 'Best_Train_Loss', 'Best_Train_PPL', 'Best_Val_PPL', 
                   'Training_Time_Seconds', 'Total_Parameters', 'Vocab_Size']
    
    for col in numeric_cols:
        df_success[col] = pd.to_numeric(df_success[col], errors='coerce')
    
    # Create summary plots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Best Validation Loss by Model and Dataset
    plt.subplot(2, 3, 1)
    if not df_success['Best_Val_Loss'].isna().all():
        models = df_success['Model'].unique()
        datasets = df_success['Dataset'].unique()
        
        # Create a simple bar plot if pivot doesn't work well
        model_val_loss = df_success.groupby('Model')['Best_Val_Loss'].mean()
        bars = plt.bar(model_val_loss.index, model_val_loss.values)
        plt.title('Average Validation Loss by Model')
        plt.ylabel('Validation Loss')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'No Validation Loss Data', ha='center', va='center')
        plt.title('Validation Loss (No Data)')
    
    # Plot 2: Training Time by Model
    plt.subplot(2, 3, 2)
    if not df_success['Training_Time_Seconds'].isna().all():
        model_time = df_success.groupby('Model')['Training_Time_Seconds'].mean()
        bars = plt.bar(model_time.index, model_time.values)
        plt.title('Average Training Time by Model')
        plt.ylabel('Time (Seconds)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}s', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'No Training Time Data', ha='center', va='center')
        plt.title('Training Time (No Data)')
    
    # Plot 3: Model Parameters
    plt.subplot(2, 3, 3)
    if not df_success['Total_Parameters'].isna().all():
        model_params = df_success.groupby('Model')['Total_Parameters'].first()
        bars = plt.bar(model_params.index, model_params.values)
        plt.title('Model Size (Parameters)')
        plt.ylabel('Parameters')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=8)
    else:
        plt.text(0.5, 0.5, 'No Parameters Data', ha='center', va='center')
        plt.title('Model Size (No Data)')
    
    # Plot 4: Success Rate by Model
    plt.subplot(2, 3, 4)
    success_rate = df.groupby('Model')['Status'].apply(lambda x: (x == 'SUCCESS').mean() * 100)
    bars = plt.bar(success_rate.index, success_rate.values)
    plt.title('Success Rate by Model')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Plot 5: Dataset Vocabulary Size
    plt.subplot(2, 3, 5)
    if not df_success['Vocab_Size'].isna().all():
        vocab_by_dataset = df_success.groupby('Dataset')['Vocab_Size'].first()
        bars = plt.bar(range(len(vocab_by_dataset)), vocab_by_dataset.values)
        plt.title('Vocabulary Size by Dataset')
        plt.ylabel('Vocabulary Size')
        plt.xticks(range(len(vocab_by_dataset)), vocab_by_dataset.index, rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'No Vocab Data', ha='center', va='center')
        plt.title('Vocabulary Size (No Data)')
    
    # Plot 6: Experiment Status Overview
    plt.subplot(2, 3, 6)
    status_counts = df['Status'].value_counts()
    colors = ['green' if status == 'SUCCESS' else 'red' for status in status_counts.index]
    bars = plt.bar(status_counts.index, status_counts.values, color=colors, alpha=0.7)
    plt.title('Experiment Status Overview')
    plt.ylabel('Number of Experiments')
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save summary plot
    summary_plot_path = os.path.join(plots_dir, 'test_experiment_summary.png')
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved test summary plot: {summary_plot_path}")
    
    plt.close()
    
    # Print summary table
    print("\n" + "="*60)
    print("TEST EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    print(f"Total Experiments: {len(df)}")
    print(f"Successful: {len(df_success)}")
    print(f"Failed: {len(df) - len(df_success)}")
    
    if len(df_success) > 0:
        print("\nSuccessful Experiments:")
        print(df_success[['Model', 'Dataset', 'Best_Val_Loss', 'Training_Time_Seconds']].to_string(index=False))
    
    return True

if __name__ == "__main__":
    csv_file = sys.argv[1]
    plots_dir = sys.argv[2]
    
    success = create_test_summary_plots(csv_file, plots_dir)
    sys.exit(0 if success else 1)
EOF

# Run test summary plotting
if python "$BASE_DIR/test_summary_plots.py" "$RESULTS_CSV" "$PLOTS_DIR"; then
    print_success "Successfully created test summary plots"
else
    print_error "Failed to create test summary plots"
fi

# Clean up
rm -f "$BASE_DIR/test_summary_plots.py"

print_status "Analysis features test completed!"
print_status "Check the following files:"
print_status "  CSV: $RESULTS_CSV"
print_status "  Plots: $PLOTS_DIR/"

# Show CSV content
if [ -f "$RESULTS_CSV" ]; then
    print_status "CSV content preview:"
    echo "----------------------------------------"
    head -3 "$RESULTS_CSV"
    echo "----------------------------------------"
fi

# List generated plots
if [ -d "$PLOTS_DIR" ]; then
    print_status "Generated plots:"
    ls -la "$PLOTS_DIR"/*.png 2>/dev/null || echo "No plots found"
fi