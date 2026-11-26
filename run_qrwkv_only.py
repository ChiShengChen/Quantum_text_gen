#!/usr/bin/env python3
"""
只運行QRWKV模型的評估腳本
"""

import os
import sys
import torch
import json
import time
import numpy as np
from collections import Counter
import re
from datetime import datetime

sys.path.append('.')

from data.datasets import create_dataloader
from qrwkv_model import QuantumRWKVModel, ModelConfig
from comprehensive_evaluation import (
    calculate_perplexity, 
    calculate_bleu_score, 
    calculate_diversity_metrics,
    calculate_repetition_rate,
    calculate_fluency_metrics,
    generate_texts_for_evaluation,
    get_reference_texts
)

def create_qrwkv_model(vocab_size, device, **kwargs):
    """創建QRWKV模型"""
    config = ModelConfig(
        vocab_size=vocab_size,
        n_embd=kwargs.get('embedding_dim', 64),
        n_head=8,
        n_layer=kwargs.get('n_layers', 2),
        n_qubits=kwargs.get('n_qubits', 2),
        block_size=256
    )
    return QuantumRWKVModel(config)

def run_qrwkv_experiment(dataset_name, device, epochs=8, batch_size=16, max_length=32, 
                        vocab_size=500, max_samples=1000, learning_rate=0.001):
    """運行QRWKV實驗"""
    
    print(f"🚀 開始QRWKV實驗: {dataset_name}")
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
    
    # 創建QRWKV模型
    print(f"\n🤖 創建QRWKV模型...")
    vocab_size = tokenizer.get_vocab_size()
    model = create_qrwkv_model(
        vocab_size=vocab_size,
        device=device,
        embedding_dim=64,
        n_layers=2,
        n_qubits=2
    )
    
    print(f"✅ 模型創建成功")
    print(f"📊 模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 訓練模型
    print(f"\n🎯 開始訓練...")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            logits = outputs.view(-1, outputs.size(-1))
            targets = target_ids.view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                
                outputs = model(input_ids)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                logits = outputs.view(-1, outputs.size(-1))
                targets = target_ids.view(-1)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    print(f"✅ 訓練完成! 最佳驗證損失: {best_val_loss:.4f}")
    
    # 全面評估
    print(f"\n📊 開始全面評估...")
    evaluation_results = evaluate_model_comprehensive(model, tokenizer, test_loader, device, dataset_name)
    
    # 保存結果
    print(f"\n💾 保存評估結果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"./experiments/comprehensive/qrwkv/{dataset_name}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    result = {
        'model_type': 'qrwkv',
        'dataset_name': dataset_name,
        'training_results': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        },
        'evaluation_results': evaluation_results,
        'vocab_size': vocab_size,
        'model_dir': result_dir
    }
    
    # 保存詳細結果
    with open(os.path.join(result_dir, 'comprehensive_results.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    # 保存生成文本對比
    with open(os.path.join(result_dir, 'text_comparison.txt'), 'w') as f:
        f.write(f"Text Generation Comparison for QRWKV on {dataset_name}\n")
        f.write("=" * 60 + "\n\n")
        
        for i, (ref, gen) in enumerate(zip(evaluation_results['reference_texts'], evaluation_results['generated_texts'])):
            f.write(f"Example {i+1}:\n")
            f.write(f"Reference: {ref}\n")
            f.write(f"Generated: {gen}\n\n")
    
    print(f"✅ QRWKV評估完成: {dataset_name}")
    print(f"📁 結果保存到: {result_dir}")
    
    return result

def evaluate_model_comprehensive(model, tokenizer, test_loader, device, dataset_name):
    """全面評估模型"""
    print("📊 開始全面評估...")
    
    # 1. 計算困惑度
    perplexity = calculate_perplexity(model, test_loader, device)
    print(f"  📈 困惑度: {perplexity:.4f}")
    
    # 2. 生成文本
    test_prompts = ["the", "birds", "water", "spring", "stars"]
    generated_texts = generate_texts_for_evaluation(model, tokenizer, test_prompts, device)
    
    print(f"  📝 生成文本示例:")
    for i, text in enumerate(generated_texts):
        print(f"    {i+1}. {text}")
    
    # 3. 獲取參考文本
    reference_texts = get_reference_texts(dataset_name, len(generated_texts))
    
    # 4. 計算BLEU分數
    bleu_scores = calculate_bleu_score(reference_texts, generated_texts)
    print(f"  📊 BLEU分數: {bleu_scores}")
    
    # 5. 計算多樣性指標
    diversity_metrics = calculate_diversity_metrics(generated_texts)
    print(f"  🌈 多樣性指標: {diversity_metrics}")
    
    # 6. 計算重複率
    repetition_rate = calculate_repetition_rate(generated_texts)
    print(f"  🔄 重複率: {repetition_rate:.4f}")
    
    # 7. 計算流暢度
    fluency_metrics = calculate_fluency_metrics(generated_texts)
    print(f"  📝 流暢度指標: {fluency_metrics}")
    
    return {
        'perplexity': perplexity,
        'bleu_scores': bleu_scores,
        'diversity_metrics': diversity_metrics,
        'repetition_rate': repetition_rate,
        'fluency_metrics': fluency_metrics,
        'generated_texts': generated_texts,
        'reference_texts': reference_texts
    }

def main():
    """主函數"""
    print("🚀 QRWKV模型專用評估系統")
    print("=" * 60)
    
    # 設置設備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用設備: {device}")
    
    # 實驗配置
    datasets = ['simple_sentences', 'short_stories', 'quantum_phrases', 'haiku', 'proverbs']
    
    # 從環境變量讀取配置
    epochs = int(os.environ.get('EPOCHS', 8))
    batch_size = int(os.environ.get('BATCH_SIZE', 16))
    max_length = int(os.environ.get('MAX_LENGTH', 32))
    vocab_size = int(os.environ.get('VOCAB_SIZE', 500))
    max_samples = int(os.environ.get('MAX_SAMPLES', 1000))
    learning_rate = float(os.environ.get('LEARNING_RATE', 0.001))
    
    print(f"📊 數據集: {datasets}")
    print(f"🔄 訓練輪數: {epochs} epochs")
    print(f"📈 評估指標: 困惑度、BLEU、多樣性、重複率、流暢度")
    
    # 運行實驗
    all_results = {}
    
    for dataset_name in datasets:
        try:
            result = run_qrwkv_experiment(
                dataset_name=dataset_name,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                max_length=max_length,
                vocab_size=vocab_size,
                max_samples=max_samples,
                learning_rate=learning_rate
            )
            all_results[dataset_name] = result
        except Exception as e:
            print(f"❌ 實驗失敗 {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成綜合報告
    print("\n📊 生成QRWKV綜合評估報告...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"./experiments/comprehensive/qrwkv/qrwkv_report_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    # 保存所有結果
    with open(os.path.join(report_dir, 'all_qrwkv_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # 生成報告
    report = f"""
# QRWKV模型評估報告

## 實驗概覽
- 模型: QRWKV (Quantum RWKV)
- 數據集: {datasets}
- 訓練輪數: {epochs}
- 批次大小: {batch_size}
- 最大長度: {max_length}
- 詞彙大小: {vocab_size}
- 最大樣本: {max_samples}
- 學習率: {learning_rate}

## 結果總結
"""
    
    # 添加每個數據集的結果
    for dataset_name, result in all_results.items():
        if 'evaluation_results' in result:
            eval_results = result['evaluation_results']
            bleu_info = eval_results.get('bleu_scores', {})
            bleu_str = f"BLEU-1: {bleu_info.get('BLEU-1', 'N/A'):.4f}" if isinstance(bleu_info, dict) else str(bleu_info)
            
            report += f"""
### {dataset_name}
- 困惑度: {eval_results.get('perplexity', 'N/A'):.4f}
- BLEU分數: {bleu_str}
- 詞彙多樣性: {eval_results.get('diversity_metrics', {}).get('distinct_1', 'N/A'):.4f}
- 重複率: {eval_results.get('repetition_rate', 'N/A'):.4f}
"""
    
    # 保存報告
    with open(os.path.join(report_dir, 'qrwkv_report.md'), 'w') as f:
        f.write(report)
    
    print(f"✅ QRWKV評估完成!")
    print(f"📁 報告保存到: {report_dir}")

if __name__ == "__main__":
    main()
