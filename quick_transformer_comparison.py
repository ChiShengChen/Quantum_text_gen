#!/usr/bin/env python3
"""
快速比較Transformer Baseline與量子模型
只測試simple_sentences數據集
"""

import os
import sys
import torch
import json
import time
import numpy as np
from datetime import datetime

# 設置環境變量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def main():
    print("🚀 快速比較Transformer Baseline與量子模型")
    print("="*60)
    
    # 檢查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用設備: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # 實驗配置 - 使用較小的配置進行快速測試
    datasets = ['simple_sentences']  # 只測試一個數據集
    models = ['qksan', 'qrwkv', 'qasa', 'transformer']
    
    # 較小的配置以加快測試
    epochs = 20  # 減少訓練輪數
    batch_size = 8
    max_length = 16
    vocab_size = 500
    max_samples = 50
    learning_rate = 0.001
    
    print(f"⚙️  快速測試配置:")
    print(f"   - 訓練輪數: {epochs}")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - 最大長度: {max_length}")
    print(f"   - 詞彙大小: {vocab_size}")
    print(f"   - 最大樣本: {max_samples}")
    print(f"   - 學習率: {learning_rate}")
    
    print(f"\n📊 評估模型: {', '.join(models)}")
    print(f"📚 數據集: {', '.join(datasets)}")
    
    # 導入評估函數
    from comprehensive_evaluation import run_comprehensive_experiment
    
    # 運行實驗
    all_results = {}
    
    for model_type in models:
        for dataset_name in datasets:
            try:
                print(f"\n🔬 開始實驗: {model_type.upper()} on {dataset_name}")
                print("-" * 50)
                
                result = run_comprehensive_experiment(
                    model_type, dataset_name, device, epochs, batch_size, 
                    max_length, vocab_size, max_samples, learning_rate
                )
                
                experiment_key = f"{model_type}_{dataset_name}"
                all_results[experiment_key] = result
                
                print(f"✅ 實驗完成: {model_type.upper()} on {dataset_name}")
                
            except Exception as e:
                print(f"❌ 實驗失敗 {model_type} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 生成快速比較報告
    print("\n📊 生成快速比較報告...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"./experiments/quick_comparison_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    # 保存結果
    with open(os.path.join(report_dir, 'quick_comparison_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # 生成報告
    report = f"""
# 快速比較報告: Transformer Baseline vs 量子模型

## 實驗概覽
- **實驗時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **訓練輪數**: {epochs} epochs
- **數據集**: simple_sentences
- **模型數量**: {len(models)}個 (包含Transformer Baseline)
- **配置**: 快速測試配置

## 實驗結果

### 性能比較表
| 模型 | 困惑度 | BLEU-1 | BLEU-2 | Distinct-1 | Distinct-2 | 重複率 |
|------|--------|--------|--------|------------|------------|--------|
"""
    
    for model in models:
        model_data = [v for k, v in all_results.items() if v['model_type'] == model]
        if model_data:
            data = model_data[0]  # 只有一個數據集
            eval_results = data['evaluation_results']
            report += f"| {model.upper()} | {eval_results['perplexity']:.2f} | {eval_results['bleu_scores']['BLEU-1']:.3f} | {eval_results['bleu_scores']['BLEU-2']:.3f} | {eval_results['diversity_metrics']['distinct_1']:.3f} | {eval_results['diversity_metrics']['distinct_2']:.3f} | {eval_results['repetition_rate']:.3f} |\n"
    
    # 性能分析
    report += "\n## 性能分析\n\n"
    
    # 找出最佳模型
    best_perplexity = float('inf')
    best_bleu = 0
    best_distinct = 0
    best_models = {}
    
    for model in models:
        model_data = [v for k, v in all_results.items() if v['model_type'] == model]
        if model_data:
            data = model_data[0]
            eval_results = data['evaluation_results']
            
            perplexity = eval_results['perplexity']
            bleu1 = eval_results['bleu_scores']['BLEU-1']
            distinct1 = eval_results['diversity_metrics']['distinct_1']
            
            if perplexity < best_perplexity and perplexity != float('inf'):
                best_perplexity = perplexity
                best_models['perplexity'] = model
            
            if bleu1 > best_bleu:
                best_bleu = bleu1
                best_models['bleu'] = model
            
            if distinct1 > best_distinct:
                best_distinct = distinct1
                best_models['distinct'] = model
    
    report += "### 最佳性能模型\n"
    if 'perplexity' in best_models:
        report += f"- **最低困惑度**: {best_models['perplexity'].upper()} ({best_perplexity:.2f})\n"
    if 'bleu' in best_models:
        report += f"- **最高BLEU-1**: {best_models['bleu'].upper()} ({best_bleu:.3f})\n"
    if 'distinct' in best_models:
        report += f"- **最高Distinct-1**: {best_models['distinct'].upper()} ({best_distinct:.3f})\n"
    
    # 與Transformer的比較
    transformer_data = [v for k, v in all_results.items() if v['model_type'] == 'transformer']
    if transformer_data:
        transformer_results = transformer_data[0]['evaluation_results']
        report += f"\n### Transformer Baseline 基準\n"
        report += f"- 困惑度: {transformer_results['perplexity']:.2f}\n"
        report += f"- BLEU-1: {transformer_results['bleu_scores']['BLEU-1']:.3f}\n"
        report += f"- Distinct-1: {transformer_results['diversity_metrics']['distinct_1']:.3f}\n"
        report += f"- 重複率: {transformer_results['repetition_rate']:.3f}\n"
        
        # 比較量子模型與Transformer
        report += f"\n### 量子模型 vs Transformer\n"
        quantum_models = ['qksan', 'qrwkv', 'qasa']
        for model in quantum_models:
            model_data = [v for k, v in all_results.items() if v['model_type'] == model]
            if model_data:
                model_results = model_data[0]['evaluation_results']
                
                perplexity_diff = ((model_results['perplexity'] - transformer_results['perplexity']) / transformer_results['perplexity']) * 100 if transformer_results['perplexity'] != float('inf') else 0
                bleu_diff = ((model_results['bleu_scores']['BLEU-1'] - transformer_results['bleu_scores']['BLEU-1']) / transformer_results['bleu_scores']['BLEU-1']) * 100 if transformer_results['bleu_scores']['BLEU-1'] > 0 else 0
                distinct_diff = ((model_results['diversity_metrics']['distinct_1'] - transformer_results['diversity_metrics']['distinct_1']) / transformer_results['diversity_metrics']['distinct_1']) * 100 if transformer_results['diversity_metrics']['distinct_1'] > 0 else 0
                
                report += f"\n**{model.upper()}**:\n"
                report += f"- 困惑度差異: {perplexity_diff:+.1f}% ({'更好' if perplexity_diff < 0 else '更差'})\n"
                report += f"- BLEU-1差異: {bleu_diff:+.1f}% ({'更好' if bleu_diff > 0 else '更差'})\n"
                report += f"- Distinct-1差異: {distinct_diff:+.1f}% ({'更好' if distinct_diff > 0 else '更差'})\n"
    
    # 結論
    report += f"\n## 結論\n\n"
    report += f"這是一個快速比較測試，使用較小的配置來驗證Transformer Baseline模型的效果。\n"
    report += f"完整評估建議運行 `run_transformer_baseline_evaluation.py` 腳本。\n"
    
    with open(os.path.join(report_dir, 'quick_comparison_report.md'), 'w') as f:
        f.write(report)
    
    print(f"📄 快速比較報告已保存到: {report_dir}")
    print(f"\n{'='*60}")
    print("🎉 快速比較完成!")
    print(f"📊 共完成了 {len(all_results)} 個實驗")
    print(f"💡 查看報告了解Transformer Baseline的效果")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
