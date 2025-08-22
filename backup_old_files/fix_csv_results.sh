#!/bin/bash

# Script to fix the CSV results by re-extracting metrics from existing experiments

BASE_DIR="./experiments"
RESULTS_CSV="$BASE_DIR/experiment_results_fixed.csv"

# Create new CSV header
cat > "$RESULTS_CSV" << EOF
Model,Dataset,Status,Epochs_Completed,Best_Epoch,Best_Val_Loss,Best_Train_Loss,Best_Train_PPL,Best_Val_PPL,Final_Train_Loss,Final_Val_Loss,Training_Time_Seconds,Total_Parameters,Vocab_Size,Train_Samples,Val_Samples,Generation_Sample,Word_Overlap_Score,Experiment_Dir
EOF

# Function to extract metrics from training history
extract_metrics_fixed() {
    local exp_dir=$1
    local model=$2
    local dataset=$3
    
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
    local word_overlap="N/A"
    
    # Check if training completed successfully
    if [ -f "$exp_dir/status.txt" ] && [ "$(cat $exp_dir/status.txt)" = "SUCCESS" ]; then
        status="SUCCESS"
        
        # Extract from output log
        if [ -f "$output_log" ]; then
            epochs_completed=$(grep -c "INFO - Epoch [0-9]*:" "$output_log" || echo "0")
            training_time=$(grep "Training completed in" "$output_log" | awk '{print $10}' || echo "0")
            total_params=$(grep "Total parameters:" "$output_log" | awk '{print $9}' | tr -d ',' || echo "N/A")
            vocab_size=$(grep "Vocabulary size:" "$output_log" | awk '{print $9}' || echo "N/A")
            train_samples=$(grep "Train samples:" "$output_log" | awk '{print $9}' || echo "N/A")
            val_samples=$(grep "Validation samples:" "$output_log" | awk '{print $9}' || echo "N/A")
            
            # Get best validation loss
            best_val_loss=$(grep "Best validation loss:" "$output_log" | tail -1 | awk '{print $10}' || echo "N/A")
            
            # Get final losses
            final_train_loss=$(grep "Train Loss:" "$output_log" | tail -1 | awk '{print $12}' | tr -d ',' || echo "N/A")
            final_val_loss=$(grep "Val Loss:" "$output_log" | tail -1 | awk '{print $12}' | tr -d ',' || echo "N/A")
            
            # Get final PPL
            final_train_ppl=$(grep "Train PPL:" "$output_log" | tail -1 | awk '{print $15}' || echo "N/A")
            final_val_ppl=$(grep "Val Loss:" "$output_log" | tail -1 | awk '{print $15}' || echo "N/A")
            
            # For best metrics, use the final values as approximation
            if [ "$best_val_loss" != "N/A" ]; then
                best_train_loss="$final_train_loss"
                best_train_ppl="$final_train_ppl"
                best_val_ppl="$final_val_ppl"
                # Extract epoch number from best validation loss line
                best_epoch=$(grep -n "Best validation loss:" "$output_log" | tail -1 | cut -d':' -f1)
            fi
        fi
        
        # Extract generation sample
        if [ -f "$exp_dir/generation_test.txt" ]; then
            generation_sample=$(grep "Generated:" "$exp_dir/generation_test.txt" | head -1 | cut -d':' -f2- | cut -c1-100 | tr -d '\n' | tr ',' ';' || echo "N/A")
        fi
        
        # Extract evaluation metrics
        if [ -f "$exp_dir/evaluation_results.json" ]; then
            word_overlap=$(grep "avg_word_overlap" "$exp_dir/evaluation_results.json" | awk -F':' '{print $2}' | tr -d ' ,' || echo "N/A")
        fi
    fi
    
    # Write to CSV
    echo "$model,$dataset,$status,$epochs_completed,$best_epoch,$best_val_loss,$best_train_loss,$best_train_ppl,$best_val_ppl,$final_train_loss,$final_val_loss,$training_time,$total_params,$vocab_size,$train_samples,$val_samples,\"$generation_sample\",$word_overlap,$exp_dir" >> "$RESULTS_CSV"
    
    echo "Processed: $model + $dataset"
}

# Process all experiments
MODELS=("qasa" "qksan" "qrwkv")
DATASETS=("simple_sentences" "short_stories" "quantum_phrases" "haiku" "proverbs")

echo "Re-extracting metrics from completed experiments..."

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        exp_dir="${BASE_DIR}/${model}/${dataset}"
        if [ -d "$exp_dir" ]; then
            extract_metrics_fixed "$exp_dir" "$model" "$dataset"
        fi
    done
done

echo "Fixed CSV saved to: $RESULTS_CSV"
echo "You can replace the original with: mv $RESULTS_CSV $BASE_DIR/experiment_results.csv"