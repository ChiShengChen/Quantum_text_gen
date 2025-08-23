#!/usr/bin/env python3
"""
運行包含Transformer Baseline的全面評估
"""

import os
import sys
import torch
import json
import time
import numpy as np
from datetime import datetime

# 設置環境變量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用GPU 0
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def main():
    print("🚀 開始運行包含Transformer Baseline的全面評估")
    print("="*80)
    
    # 檢查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用設備: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 實驗配置
    datasets = ['simple_sentences', 'short_stories', 'quantum_phrases', 'haiku', 'proverbs']
    models = ['qksan', 'qrwkv', 'qasa', 'transformer']  # 包含Transformer baseline
    
    # 從環境變量讀取配置
    epochs = int(os.environ.get('EPOCHS', '50'))
    batch_size = int(os.environ.get('BATCH_SIZE', '16'))
    max_length = int(os.environ.get('MAX_LENGTH', '32'))
    vocab_size = int(os.environ.get('VOCAB_SIZE', '1000'))
    max_samples = int(os.environ.get('MAX_SAMPLES', '100'))
    learning_rate = float(os.environ.get('LEARNING_RATE', '0.001'))
    
    print(f"⚙️  實驗配置:")
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
                print("-" * 60)
                
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
    
    # 生成綜合報告
    print("\n📊 生成綜合評估報告...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"./experiments/comprehensive/comprehensive_report_with_transformer_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    # 保存所有結果
    with open(os.path.join(report_dir, 'all_comprehensive_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # 生成報告
    report = f"""
# 量子文字生成模型全面評估報告 (含Transformer Baseline)

## 實驗概覽
- **實驗時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **訓練輪數**: {epochs} epochs
- **數據集數量**: {len(datasets)}
- **模型數量**: {len(models)}個 (包含Transformer Baseline)
- **評估指標**: 困惑度、BLEU-1/2/3/4、多樣性(Distinct-1/2)、重複率、流暢度

## 評估指標說明

### 1. 困惑度 (Perplexity)
- 衡量語言模型預測下一個詞的能力
- 越低越好

### 2. BLEU分數 (BLEU-1/2/3/4)
- 衡量生成文本與參考文本的n-gram重疊度
- 越高越好

### 3. 多樣性指標
- **Distinct-1/2**: 衡量詞彙使用的多樣性
- **Entropy**: 詞彙分布的熵
- **Vocabulary Size**: 使用的詞彙大小

### 4. 重複率 (Repetition Rate)
- 衡量文本中連續重複詞的比例
- 越低越好

### 5. 流暢度指標
- **平均句子長度**: 句子的平均詞數
- **句子長度標準差**: 句子長度的變化
- **句子長度變異係數**: 標準差/平均值

## 實驗結果

### 按模型類型比較
"""
    
    for model in models:
        model_data = [v for k, v in all_results.items() if v['model_type'] == model]
        if model_data:
            report += f"\n#### {model.upper()} 模型\n"
            report += "| 數據集 | 困惑度 | BLEU-1 | BLEU-2 | Distinct-1 | Distinct-2 | 重複率 |\n"
            report += "|--------|--------|--------|--------|------------|------------|--------|\n"
            
            for data in model_data:
                eval_results = data['evaluation_results']
                report += f"| {data['dataset_name']} | {eval_results['perplexity']:.2f} | {eval_results['bleu_scores']['BLEU-1']:.3f} | {eval_results['bleu_scores']['BLEU-2']:.3f} | {eval_results['diversity_metrics']['distinct_1']:.3f} | {eval_results['diversity_metrics']['distinct_2']:.3f} | {eval_results['repetition_rate']:.3f} |\n"
    
    # 性能總結
    report += "\n## 性能總結\n\n"
    
    for model in models:
        model_data = [v for k, v in all_results.items() if v['model_type'] == model]
        if model_data:
            avg_perplexity = np.mean([d['evaluation_results']['perplexity'] for d in model_data if d['evaluation_results']['perplexity'] != float('inf')])
            avg_bleu1 = np.mean([d['evaluation_results']['bleu_scores']['BLEU-1'] for d in model_data])
            avg_distinct1 = np.mean([d['evaluation_results']['diversity_metrics']['distinct_1'] for d in model_data])
            avg_repetition = np.mean([d['evaluation_results']['repetition_rate'] for d in model_data])
            
            report += f"- **{model.upper()}**: 平均困惑度={avg_perplexity:.2f}, 平均BLEU-1={avg_bleu1:.3f}, 平均Distinct-1={avg_distinct1:.3f}, 平均重複率={avg_repetition:.3f}\n"
    
    # 與Transformer Baseline的比較分析
    report += "\n## 與Transformer Baseline的比較分析\n\n"
    
    transformer_data = [v for k, v in all_results.items() if v['model_type'] == 'transformer']
    if transformer_data:
        transformer_avg_perplexity = np.mean([d['evaluation_results']['perplexity'] for d in transformer_data if d['evaluation_results']['perplexity'] != float('inf')])
        transformer_avg_bleu1 = np.mean([d['evaluation_results']['bleu_scores']['BLEU-1'] for d in transformer_data])
        transformer_avg_distinct1 = np.mean([d['evaluation_results']['diversity_metrics']['distinct_1'] for d in transformer_data])
        transformer_avg_repetition = np.mean([d['evaluation_results']['repetition_rate'] for d in transformer_data])
        
        report += f"### Transformer Baseline 基準性能\n"
        report += f"- 平均困惑度: {transformer_avg_perplexity:.2f}\n"
        report += f"- 平均BLEU-1: {transformer_avg_bleu1:.3f}\n"
        report += f"- 平均Distinct-1: {transformer_avg_distinct1:.3f}\n"
        report += f"- 平均重複率: {transformer_avg_repetition:.3f}\n\n"
        
        # 比較各量子模型與Transformer的差異
        quantum_models = ['qksan', 'qrwkv', 'qasa']
        for model in quantum_models:
            model_data = [v for k, v in all_results.items() if v['model_type'] == model]
            if model_data:
                model_avg_perplexity = np.mean([d['evaluation_results']['perplexity'] for d in model_data if d['evaluation_results']['perplexity'] != float('inf')])
                model_avg_bleu1 = np.mean([d['evaluation_results']['bleu_scores']['BLEU-1'] for d in model_data])
                model_avg_distinct1 = np.mean([d['evaluation_results']['diversity_metrics']['distinct_1'] for d in model_data])
                model_avg_repetition = np.mean([d['evaluation_results']['repetition_rate'] for d in model_data])
                
                perplexity_diff = ((model_avg_perplexity - transformer_avg_perplexity) / transformer_avg_perplexity) * 100
                bleu_diff = ((model_avg_bleu1 - transformer_avg_bleu1) / transformer_avg_bleu1) * 100 if transformer_avg_bleu1 > 0 else 0
                distinct_diff = ((model_avg_distinct1 - transformer_avg_distinct1) / transformer_avg_distinct1) * 100
                
                report += f"### {model.upper()} vs Transformer\n"
                report += f"- 困惑度差異: {perplexity_diff:+.1f}% ({'更好' if perplexity_diff < 0 else '更差'})\n"
                report += f"- BLEU-1差異: {bleu_diff:+.1f}% ({'更好' if bleu_diff > 0 else '更差'})\n"
                report += f"- Distinct-1差異: {distinct_diff:+.1f}% ({'更好' if distinct_diff > 0 else '更差'})\n\n"
    
    with open(os.path.join(report_dir, 'comprehensive_report.md'), 'w') as f:
        f.write(report)
    
    print(f"📄 綜合報告已保存到: {report_dir}")
    print(f"\n{'='*80}")
    print("🎉 包含Transformer Baseline的全面評估完成!")
    print(f"📊 共完成了 {len(all_results)} 個實驗")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
