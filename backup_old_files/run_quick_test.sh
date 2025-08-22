#!/bin/bash

# Quick test version - runs a subset of experiments for testing
# This script runs only 2 model-dataset combinations for quick verification

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration for quick test
EPOCHS=5
BATCH_SIZE=4
MAX_SAMPLES=100
LEARNING_RATE=0.01
BASE_DIR="./test_experiments"

# Quick test: only test 2 combinations
MODELS=("qasa" "qksan")
DATASETS=("simple_sentences" "quantum_phrases")

# Model-specific parameters
declare -A MODEL_PARAMS
MODEL_PARAMS[qasa]="--hidden_dim 64 --n_qubits 4"
MODEL_PARAMS[qksan]="--hidden_dim 64 --n_qubits 4 --qksan_variant PCAAmHE"

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

# Function to run a single experiment
run_experiment() {
    local model=$1
    local dataset=$2
    local exp_dir="${BASE_DIR}/${model}/${dataset}"
    
    print_status "Starting quick test: ${model} on ${dataset}"
    
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
    if eval $cmd; then
        print_success "Completed: $model + $dataset"
        
        # Test generation
        print_status "Testing generation..."
        python quantum_text_gen.py \
            --mode generate \
            --checkpoint "$exp_dir/best_model.pth" \
            --prompt "the" \
            --max_length 20 \
            --temperature 1.0 \
            --do_sample
            
    else
        print_error "Failed: $model + $dataset"
        return 1
    fi
    
    return 0
}

# Main execution
main() {
    print_status "Starting Quick Test of Experiment Runner"
    print_status "This will run 2 quick experiments to test the system"
    
    # Create base experiment directory
    mkdir -p "$BASE_DIR"
    
    local successful=0
    local failed=0
    
    # Run quick tests
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            if run_experiment "$model" "$dataset"; then
                successful=$((successful + 1))
            else
                failed=$((failed + 1))
            fi
        done
    done
    
    # Print summary
    echo ""
    print_status "Quick test completed!"
    print_success "Successful: $successful"
    if [ $failed -gt 0 ]; then
        print_error "Failed: $failed"
    fi
    
    if [ $successful -gt 0 ]; then
        print_success "System is working! You can now run the full experiment script:"
        echo "  ./run_all_experiments.sh"
    fi
}

# Run main function
main "$@"