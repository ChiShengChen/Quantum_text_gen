#!/usr/bin/env python3
"""
Transformer Benchmark Script
運行傳統Transformer模型在相同數據集上的benchmark
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any

sys.path.append('.')

from data.datasets import create_dataloader
from transformer_model import TransformerModel, TransformerConfig
from models.transformer_text_models import create_transformer_generator
from comprehensive_evaluation import (
    calculate_perplexity,
    calculate_bleu_score,
    calculate_diversity_metrics,
    calculate_repetition_rate,
    calculate_fluency_metrics,
    generate_texts_for_evaluation,
    get_reference_texts
)

def create_transformer_model_for_benchmark(vocab_size, device, **kwargs):
    """創建用於benchmark的Transformer模型"""
    config = TransformerConfig(
        vocab_size=vocab_size,
        n_embd=kwargs.get('n_embd', 256),  # 較小的嵌入維度
        n_head=kwargs.get('n_head', 8),
        n_layer=kwargs.get('n_layer', 4),   # 較少的層數
        block_size=kwargs.get('block_size', 256),
        n_intermediate=kwargs.get('n_intermediate', 1024),
        dropout=kwargs.get('dropout', 0.1)
    )
    return TransformerModel(config)

def run_transformer_experiment(dataset_name, device, epochs=8, batch_size=16, max_length=32, 
                             vocab_size=500, max_samples=1000, learning_rate=0.001):
    """運行Transformer實驗"""
    
    print(f"🚀 開始Transformer實驗: {dataset_name}")
    print(f"📊 配置: epochs={epochs}, batch_size={batch_size}, max_length={max_length}")
    print(f"📊 配置: vocab_size={vocab_size}, max_samples={max_samples}, lr={learning_rate}")
    
    # 創建數據加載器
    print(f"\n📊 創建數據加載器...")
    train_loader, tokenizer = create_dataloader(
        dataset_name=dataset_name,
        split='train',
        batch_size=batch_size,
        max_length=max_length,
        vocab_size=vocab_size,
        max_samples=max_samples,
        shuffle=True
    )
    
    val_loader, _ = create_dataloader(
        dataset_name=dataset_name,
        split='val',
        batch_size=batch_size,
        max_length=max_length,
        vocab_size=vocab_size,
        max_samples=max_samples // 4,
        shuffle=False
    )
    
    test_loader, _ = create_dataloader(
        dataset_name=dataset_name,
        split='test',
        batch_size=batch_size,
        max_length=max_length,
        vocab_size=vocab_size,
        max_samples=max_samples // 4,
        shuffle=False
    )
    
    print(f"✅ 數據加載器創建成功")
    print(f"📝 詞彙大小: {tokenizer.get_vocab_size()}")
    
    # 創建Transformer模型
    print(f"\n🤖 創建Transformer模型...")
    vocab_size = tokenizer.get_vocab_size()
    model = create_transformer_model_for_benchmark(
        vocab_size=vocab_size,
        device=device,
        n_embd=256,
        n_head=8,
        n_layer=4,
        block_size=256,
        n_intermediate=1024,
        dropout=0.1
    )
    
    print(f"✅ 模型創建成功")
    print(f"📊 模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 訓練模型
    print(f"\n🎯 開始訓練...")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            targets = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            logits, loss = model(input_ids, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                targets = batch['target_ids'].to(device)
                logits, loss = model(input_ids, targets)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'transformer_best_{dataset_name}.pth')
    
    print(f"✅ 訓練完成")
    
    # 評估模型
    print(f"\n📊 開始評估...")
    
    # 加載最佳模型
    model.load_state_dict(torch.load(f'transformer_best_{dataset_name}.pth'))
    model.eval()
    
    # 計算困惑度
    print(f"  🔍 計算困惑度...")
    perplexity = calculate_transformer_perplexity(model, test_loader, device)
    print(f"  📊 困惑度: {perplexity:.4f}")
    
    # 生成文本進行評估
    print(f"  ✍️ 生成文本...")
    reference_texts = get_reference_texts(dataset_name, num_samples=50)
    generated_texts = generate_texts_for_evaluation(model, tokenizer, reference_texts, device, max_length=32)
    
    # 計算BLEU分數
    print(f"  📊 計算BLEU分數...")
    bleu_scores = calculate_bleu_score(reference_texts, generated_texts)
    print(f"  📊 BLEU分數: {bleu_scores}")
    
    # 計算多樣性指標
    print(f"  🌈 計算多樣性指標...")
    diversity_metrics = calculate_diversity_metrics(generated_texts)
    print(f"  📊 多樣性指標: {diversity_metrics}")
    
    # 計算重複率
    print(f"  🔄 計算重複率...")
    repetition_rate = calculate_repetition_rate(generated_texts)
    print(f"  📊 重複率: {repetition_rate:.4f}")
    
    # 計算流暢度指標
    print(f"  📝 計算流暢度指標...")
    fluency_metrics = calculate_fluency_metrics(generated_texts)
    print(f"  📊 流暢度指標: {fluency_metrics}")
    
    # 創建結果目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"experiments/comprehensive/transformer/{dataset_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存comprehensive_results.json
    comprehensive_results = {
        'model_type': 'transformer',
        'dataset_name': dataset_name,
        'training_results': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None
        },
        'evaluation_results': {
            'perplexity': perplexity,
            'generated_texts': generated_texts[:10],  # 只保存前10個
            'reference_texts': reference_texts[:10],
            'bleu_scores': bleu_scores,
            'diversity_metrics': diversity_metrics,
            'repetition_rate': repetition_rate,
            'fluency_metrics': fluency_metrics
        },
        'vocab_size': vocab_size,
        'model_dir': output_dir,
        'model_config': {
            'n_embd': 256,
            'n_head': 8,
            'n_layer': 4,
            'block_size': 256,
            'n_intermediate': 1024,
            'dropout': 0.1
        }
    }
    
    comprehensive_file = f"{output_dir}/comprehensive_results.json"
    with open(comprehensive_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 綜合結果已保存到: {comprehensive_file}")
    
    # 生成文本比較文件
    comparison_text = f"""Text Generation Comparison for Transformer on {dataset_name}
{'='*60}

"""
    
    for i in range(min(5, len(generated_texts))):
        ref = reference_texts[i] if i < len(reference_texts) else "N/A"
        gen = generated_texts[i] if i < len(generated_texts) else "N/A"
        comparison_text += f"Example {i+1}:\n"
        comparison_text += f"Reference: {ref}\n"
        comparison_text += f"Generated: {gen}\n\n"
    
    comparison_file = f"{output_dir}/text_comparison.txt"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write(comparison_text)
    
    print(f"📝 文本比較已保存到: {comparison_file}")
    
    return comprehensive_results

def calculate_transformer_perplexity(model, data_loader, device):
    """計算Transformer模型的困惑度"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            try:
                logits, loss = model(input_ids, target_ids)
                
                # 計算token數量
                batch_tokens = target_ids.numel()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                
            except Exception as e:
                print(f"計算困惑度時跳過batch: {e}")
                continue
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def evaluate_transformer_comprehensive(datasets, device, epochs=8, batch_size=16, max_length=32, 
                                     vocab_size=500, max_samples=1000, learning_rate=0.001):
    """綜合評估Transformer模型"""
    
    print("🔬 Transformer模型綜合評估")
    print("=" * 50)
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*20} {dataset_name} {'='*20}")
        
        try:
            results = run_transformer_experiment(
                dataset_name=dataset_name,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                max_length=max_length,
                vocab_size=vocab_size,
                max_samples=max_samples,
                learning_rate=learning_rate
            )
            
            all_results[dataset_name] = results
            
        except Exception as e:
            print(f"❌ 評估 {dataset_name} 時發生錯誤: {e}")
            all_results[dataset_name] = {
                'model_type': 'transformer',
                'dataset_name': dataset_name,
                'status': 'failed',
                'error': str(e)
            }
    
    print(f"\n🎉 Transformer benchmark評估完成！")
    print(f"📁 每個數據集的結果已保存到各自的目錄中")
    print(f"📁 目錄結構: experiments/comprehensive/transformer/{dataset_name}_timestamp/")
    
    return all_results

if __name__ == "__main__":
    # 檢查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用設備: {device}")
    
    # 定義數據集
    datasets = [
        'simple_sentences',
        'short_stories', 
        'quantum_phrases',
        'haiku',
        'proverbs'
    ]
    
    # 運行綜合評估
    results = evaluate_transformer_comprehensive(
        datasets=datasets,
        device=device,
        epochs=8,
        batch_size=16,
        max_length=32,
        vocab_size=500,
        max_samples=1000,
        learning_rate=0.001
    )
    
    print("\n🎉 Transformer benchmark評估完成！")
