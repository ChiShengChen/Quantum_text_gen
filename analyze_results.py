#!/usr/bin/env python3
"""
量子文字生成模型實驗結果分析腳本
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

def load_experiment_results(file_path):
    """加載實驗結果"""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_performance_comparison(results):
    """創建性能比較圖表"""
    models = ['qksan', 'qrwkv']
    datasets = ['simple_sentences', 'short_stories', 'quantum_phrases', 'haiku', 'proverbs']
    
    # 提取困惑度數據
    perplexity_data = {}
    for model in models:
        perplexity_data[model] = []
        for dataset in datasets:
            key = f"{model}_{dataset}"
            if key in results:
                perplexity = results[key]['evaluation_results']['perplexity']
                if perplexity == float('inf'):
                    perplexity = 200  # 設置一個最大值用於顯示
                perplexity_data[model].append(perplexity)
    
    # 創建圖表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 困惑度比較
    x = np.arange(len(datasets))
    width = 0.35
    
    ax1.bar(x - width/2, perplexity_data['qksan'], width, label='QKSAN', alpha=0.8)
    ax1.bar(x + width/2, perplexity_data['qrwkv'], width, label='QRWKV', alpha=0.8)
    ax1.set_xlabel('數據集')
    ax1.set_ylabel('困惑度')
    ax1.set_title('模型困惑度比較')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. BLEU-1分數比較
    bleu1_data = {}
    for model in models:
        bleu1_data[model] = []
        for dataset in datasets:
            key = f"{model}_{dataset}"
            if key in results:
                bleu1 = results[key]['evaluation_results']['bleu_scores']['BLEU-1']
                bleu1_data[model].append(bleu1)
    
    ax2.bar(x - width/2, bleu1_data['qksan'], width, label='QKSAN', alpha=0.8)
    ax2.bar(x + width/2, bleu1_data['qrwkv'], width, label='QRWKV', alpha=0.8)
    ax2.set_xlabel('數據集')
    ax2.set_ylabel('BLEU-1分數')
    ax2.set_title('模型BLEU-1分數比較')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distinct-1分數比較
    distinct1_data = {}
    for model in models:
        distinct1_data[model] = []
        for dataset in datasets:
            key = f"{model}_{dataset}"
            if key in results:
                distinct1 = results[key]['evaluation_results']['diversity_metrics']['distinct_1']
                distinct1_data[model].append(distinct1)
    
    ax3.bar(x - width/2, distinct1_data['qksan'], width, label='QKSAN', alpha=0.8)
    ax3.bar(x + width/2, distinct1_data['qrwkv'], width, label='QRWKV', alpha=0.8)
    ax3.set_xlabel('數據集')
    ax3.set_ylabel('Distinct-1分數')
    ax3.set_title('模型多樣性(Distinct-1)比較')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 重複率比較
    repetition_data = {}
    for model in models:
        repetition_data[model] = []
        for dataset in datasets:
            key = f"{model}_{dataset}"
            if key in results:
                repetition = results[key]['evaluation_results']['repetition_rate']
                repetition_data[model].append(repetition)
    
    ax4.bar(x - width/2, repetition_data['qksan'], width, label='QKSAN', alpha=0.8)
    ax4.bar(x + width/2, repetition_data['qrwkv'], width, label='QRWKV', alpha=0.8)
    ax4.set_xlabel('數據集')
    ax4.set_ylabel('重複率')
    ax4.set_title('模型重複率比較')
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_training_curves(results):
    """創建訓練曲線"""
    models = ['qksan', 'qrwkv']
    datasets = ['short_stories', 'proverbs']  # 選擇兩個代表性數據集
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            key = f"{model}_{dataset}"
            if key in results:
                train_losses = results[key]['training_results']['train_losses']
                val_losses = results[key]['training_results']['val_losses']
                
                epochs = range(1, len(train_losses) + 1)
                axes[i, j].plot(epochs, train_losses, 'b-', label='訓練損失', linewidth=2)
                axes[i, j].plot(epochs, val_losses, 'r-', label='驗證損失', linewidth=2)
                axes[i, j].set_xlabel('Epoch')
                axes[i, j].set_ylabel('損失')
                axes[i, j].set_title(f'{model.upper()} on {dataset}')
                axes[i, j].legend()
                axes[i, j].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_summary_table(results):
    """創建總結表格"""
    models = ['qksan', 'qrwkv']
    datasets = ['simple_sentences', 'short_stories', 'quantum_phrases', 'haiku', 'proverbs']
    
    # 計算平均指標
    summary_data = []
    
    for model in models:
        model_perplexities = []
        model_bleu1 = []
        model_distinct1 = []
        model_repetition = []
        
        for dataset in datasets:
            key = f"{model}_{dataset}"
            if key in results:
                eval_results = results[key]['evaluation_results']
                perplexity = eval_results['perplexity']
                if perplexity == float('inf'):
                    perplexity = 200  # 設置最大值
                model_perplexities.append(perplexity)
                model_bleu1.append(eval_results['bleu_scores']['BLEU-1'])
                model_distinct1.append(eval_results['diversity_metrics']['distinct_1'])
                model_repetition.append(eval_results['repetition_rate'])
        
        if model_perplexities:
            summary_data.append({
                '模型': model.upper(),
                '平均困惑度': f"{np.mean(model_perplexities):.2f}",
                '平均BLEU-1': f"{np.mean(model_bleu1):.3f}",
                '平均Distinct-1': f"{np.mean(model_distinct1):.3f}",
                '平均重複率': f"{np.mean(model_repetition):.3f}",
                '成功實驗數': len(model_perplexities)
            })
    
    return pd.DataFrame(summary_data)

def main():
    """主函數"""
    print("🔍 開始分析量子文字生成模型實驗結果...")
    
    # 加載結果
    results_file = "./experiments/comprehensive/comprehensive_report_20250823_041331/all_comprehensive_results.json"
    results = load_experiment_results(results_file)
    
    print(f"✅ 成功加載 {len(results)} 個實驗結果")
    
    # 創建性能比較圖表
    print("📊 生成性能比較圖表...")
    fig1 = create_performance_comparison(results)
    fig1.savefig('./experiment_analysis_performance.png', dpi=300, bbox_inches='tight')
    print("✅ 性能比較圖表已保存: experiment_analysis_performance.png")
    
    # 創建訓練曲線
    print("📈 生成訓練曲線...")
    fig2 = create_training_curves(results)
    fig2.savefig('./experiment_analysis_training.png', dpi=300, bbox_inches='tight')
    print("✅ 訓練曲線已保存: experiment_analysis_training.png")
    
    # 創建總結表格
    print("📋 生成總結表格...")
    summary_df = create_summary_table(results)
    print("\n📊 實驗結果總結:")
    print(summary_df.to_string(index=False))
    
    # 保存總結表格
    summary_df.to_csv('./experiment_summary.csv', index=False)
    print("✅ 總結表格已保存: experiment_summary.csv")
    
    # 生成文本示例分析
    print("\n📝 生成文本示例分析:")
    for model in ['qksan', 'qrwkv']:
        for dataset in ['short_stories', 'proverbs']:
            key = f"{model}_{dataset}"
            if key in results:
                generated_texts = results[key]['evaluation_results']['generated_texts']
                print(f"\n{model.upper()} on {dataset}:")
                for i, text in enumerate(generated_texts[:2]):  # 只顯示前2個
                    print(f"  {i+1}. {text[:100]}...")  # 截斷長文本
    
    print("\n🎉 分析完成!")
    print("📁 生成的文件:")
    print("  - experiment_analysis_performance.png")
    print("  - experiment_analysis_training.png") 
    print("  - experiment_summary.csv")

if __name__ == '__main__':
    main()
