#!/bin/bash

# Quantum Text Generation - All Experiments Runner
# This script runs training experiments for all model-dataset combinations

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
EPOCHS=50
BATCH_SIZE=8
MAX_SAMPLES=300
LEARNING_RATE=0.005
BASE_DIR="./experiments"

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
    if eval $cmd > "$exp_dir/output.log" 2>&1; then
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
        
        python quantum_text_gen.py \
            --mode generate \
            --checkpoint "$exp_dir/best_model.pth" \
            --prompt "$test_prompt" \
            --max_length 30 \
            --temperature 1.0 \
            --do_sample \
            --num_sequences 3 > "$exp_dir/generation_test.txt" 2>&1
            
        if [ $? -eq 0 ]; then
            print_success "Generation test completed for: $model + $dataset"
        else
            print_warning "Generation test failed for: $model + $dataset"
        fi
        
    else
        print_error "Failed: $model + $dataset"
        echo "FAILED" > "$exp_dir/status.txt"
        return 1
    fi
    
    return 0
}

# Function to generate summary report
generate_report() {
    local report_file="$BASE_DIR/experiment_summary.md"
    
    print_status "Generating experiment summary report..."
    
    cat > "$report_file" << EOF
# Quantum Text Generation Experiment Summary

**Generated on:** $(date)
**Total Experiments:** $((${#MODELS[@]} * ${#DATASETS[@]}))

## Configuration
- Epochs: $EPOCHS
- Batch Size: $BATCH_SIZE  
- Max Samples: $MAX_SAMPLES
- Learning Rate: $LEARNING_RATE

## Results Overview

| Model | Dataset | Status | Best Val Loss | Train PPL | Generation Sample |
|-------|---------|--------|---------------|-----------|-------------------|
EOF

    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            local exp_dir="${BASE_DIR}/${model}/${dataset}"
            local status="❌ Not Run"
            local val_loss="N/A"
            local train_ppl="N/A"
            local generation="N/A"
            
            if [ -f "$exp_dir/status.txt" ]; then
                local status_text=$(cat "$exp_dir/status.txt")
                if [ "$status_text" = "SUCCESS" ]; then
                    status="✅ Success"
                    
                    # Extract metrics from log file
                    if [ -f "$exp_dir/output.log" ]; then
                        val_loss=$(grep "Best validation loss:" "$exp_dir/output.log" | tail -1 | awk '{print $4}' || echo "N/A")
                        train_ppl=$(grep "Train PPL:" "$exp_dir/output.log" | tail -1 | awk '{print $6}' || echo "N/A")
                    fi
                    
                    # Extract generation sample
                    if [ -f "$exp_dir/generation_test.txt" ]; then
                        generation=$(grep "Generated:" "$exp_dir/generation_test.txt" | head -1 | cut -d':' -f2 | cut -c1-50 || echo "N/A")
                    fi
                else
                    status="❌ Failed"
                fi
            fi
            
            echo "| $model | $dataset | $status | $val_loss | $train_ppl | $generation |" >> "$report_file"
        done
    done
    
    cat >> "$report_file" << EOF

## Detailed Results

EOF

    # Add detailed results for each experiment
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            local exp_dir="${BASE_DIR}/${model}/${dataset}"
            
            echo "### $model + $dataset" >> "$report_file"
            echo "" >> "$report_file"
            
            if [ -f "$exp_dir/status.txt" ] && [ "$(cat $exp_dir/status.txt)" = "SUCCESS" ]; then
                echo "**Status:** ✅ Success" >> "$report_file"
                
                if [ -f "$exp_dir/training_history.json" ]; then
                    echo "**Training completed successfully**" >> "$report_file"
                fi
                
                if [ -f "$exp_dir/generation_test.txt" ]; then
                    echo "" >> "$report_file"
                    echo "**Generation Test:**" >> "$report_file"
                    echo '```' >> "$report_file"
                    head -20 "$exp_dir/generation_test.txt" >> "$report_file"
                    echo '```' >> "$report_file"
                fi
                
            else
                echo "**Status:** ❌ Failed or Not Run" >> "$report_file"
                
                if [ -f "$exp_dir/output.log" ]; then
                    echo "" >> "$report_file"
                    echo "**Error Log (last 10 lines):**" >> "$report_file"
                    echo '```' >> "$report_file"
                    tail -10 "$exp_dir/output.log" >> "$report_file"
                    echo '```' >> "$report_file"
                fi
            fi
            
            echo "" >> "$report_file"
            echo "---" >> "$report_file"
            echo "" >> "$report_file"
        done
    done
    
    print_success "Report generated: $report_file"
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
    print_status "Starting Quantum Text Generation Experiments"
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
    
    # Create base experiment directory
    mkdir -p "$BASE_DIR"
    
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
    
    # Print final summary
    echo ""
    print_status "All experiments completed!"
    print_success "Successful: $successful_experiments/$total_experiments"
    if [ $failed_experiments -gt 0 ]; then
        print_error "Failed: $failed_experiments/$total_experiments"
    fi
    print_status "Total time: ${hours}h ${minutes}m ${seconds}s"
    
    # Generate report
    generate_report
    
    print_status "Experiment runner finished. Check $BASE_DIR for results."
}

# Handle interruption
trap 'print_warning "Experiment interrupted by user"; exit 130' INT

# Check if script is run with arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Quantum Text Generation Experiment Runner"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --dry-run      Show what would be executed without running"
    echo "  --models       Specify models to run (comma-separated)"
    echo "  --datasets     Specify datasets to run (comma-separated)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all experiments"
    echo "  $0 --models qasa,qksan               # Run only QASA and QKSAN"
    echo "  $0 --datasets simple_sentences,haiku # Run only specified datasets"
    echo "  $0 --dry-run                         # Preview commands"
    exit 0
fi

if [ "$1" = "--dry-run" ]; then
    print_status "DRY RUN - Commands that would be executed:"
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            echo ""
            echo "Experiment: $model + $dataset"
            echo "python quantum_text_gen.py --mode train --model $model --dataset $dataset --epochs $EPOCHS --batch_size $BATCH_SIZE --max_samples $MAX_SAMPLES --learning_rate $LEARNING_RATE --checkpoint_dir ${BASE_DIR}/${model}/${dataset} --log_dir ${BASE_DIR}/${model}/${dataset} ${MODEL_PARAMS[$model]}"
        done
    done
    exit 0
fi

# Parse custom models/datasets if provided
if [ "$1" = "--models" ] && [ -n "$2" ]; then
    IFS=',' read -ra MODELS <<< "$2"
    shift 2
fi

if [ "$1" = "--datasets" ] && [ -n "$2" ]; then
    IFS=',' read -ra DATASETS <<< "$2"
    shift 2
fi

# Run main function
main "$@"