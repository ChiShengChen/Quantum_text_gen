#!/bin/bash

# Script to run evaluation for all available models with proper error handling

echo "Running evaluation for all available models..."

MODELS=("qasa" "qksan")
DATASETS=("simple_sentences" "short_stories" "quantum_phrases" "haiku" "proverbs")

total_success=0
total_failed=0

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        exp_dir="./experiments/$model/$dataset"
        best_model="$exp_dir/best_model.pth"
        
        if [ -f "$best_model" ]; then
            echo "Evaluating $model + $dataset..."
            
            # Run evaluation with timeout to avoid hanging
            timeout 300 python quantum_text_gen.py \
                --mode evaluate \
                --checkpoint "$best_model" \
                --dataset "$dataset" \
                > "$exp_dir/evaluation_results.txt" 2>&1
            
            exit_code=$?
            
            if [ $exit_code -eq 0 ] && [ -f "$exp_dir/evaluation_results.json" ]; then
                echo "✅ Success: $model + $dataset"
                total_success=$((total_success + 1))
            else
                echo "❌ Failed: $model + $dataset (exit code: $exit_code)"
                total_failed=$((total_failed + 1))
                
                # Add failure info to the results file
                echo "Evaluation failed with exit code: $exit_code" >> "$exp_dir/evaluation_results.txt"
            fi
        else
            echo "⚠️  No model file: $model + $dataset"
        fi
    done
done

echo ""
echo "Evaluation Summary:"
echo "✅ Successful: $total_success"
echo "❌ Failed: $total_failed"

# Count evaluation_results.json files
json_count=$(find ./experiments -name "evaluation_results.json" | wc -l)
echo "📊 Total evaluation_results.json files: $json_count"

echo ""
echo "Listing all evaluation_results.json files:"
find ./experiments -name "evaluation_results.json" | sort