#!/bin/bash

# Script to create best_model.pth files from checkpoints for all experiments

echo "Creating best_model.pth files from existing checkpoints..."

MODELS=("qasa" "qksan" "qrwkv")
DATASETS=("simple_sentences" "short_stories" "quantum_phrases" "haiku" "proverbs")

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        exp_dir="./experiments/$model/$dataset"
        best_model="$exp_dir/best_model.pth"
        
        if [ -d "$exp_dir" ]; then
            if [ ! -f "$best_model" ]; then
                # Find the highest numbered checkpoint
                latest_checkpoint=$(ls "$exp_dir"/checkpoint_epoch_*.pth 2>/dev/null | sort -V | tail -1)
                
                if [ -n "$latest_checkpoint" ]; then
                    echo "Creating best_model.pth for $model + $dataset from $(basename "$latest_checkpoint")"
                    cp "$latest_checkpoint" "$best_model"
                else
                    echo "No checkpoints found for $model + $dataset"
                fi
            else
                echo "best_model.pth already exists for $model + $dataset"
            fi
        else
            echo "Directory not found: $exp_dir"
        fi
    done
done

echo "Done creating best_model.pth files!"