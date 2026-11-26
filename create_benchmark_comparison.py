#!/usr/bin/env python3
"""
創建Transformer與量子模型的benchmark比較報告
"""

import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Any

def load_latest_results(model_name: str) -> Dict[str, Any]:
    """加載指定模型的最新結果"""
    results_dir = f"experiments/comprehensive/{model_name}"
    if not os.path.exists(results_dir):
        return {}
    
    # 對於transformer模型，查找*_results_*.json文件
    if model_name == 'transformer':
        result_files = glob.glob(f"{results_dir}/*_results_*.json")
        if result_files:
            latest_file = max(result_files, key=os.path.getctime)
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加載 {model_name} 結果時出錯: {e}")
                return {}
    
    # 對於其他模型，查找comprehensive_results.json文件
    # 每個數據集都有單獨的結果文件
    datasets = ['simple_sentences', 'short_stories', 'quantum_phrases', 'haiku', 'proverbs']
    all_results = {}
    
    for dataset in datasets:
        # 查找該數據集的最新結果
        pattern = f"{results_dir}/{dataset}_*/comprehensive_results.json"
        result_files = glob.glob(pattern)
        
        if result_files:
            latest_file = max(result_files, key=os.path.getctime)
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    all_results[dataset] = result
            except Exception as e:
                print(f"加載 {model_name} {dataset} 結果時出錯: {e}")
                continue
    
    return all_results

def create_comparison_report():
    """創建比較報告"""
    
    # 加載各模型的結果
    models = ['transformer', 'qksan', 'qrwkv', 'qasa', 'mlp']
    all_results = {}
    
    for model in models:
        results = load_latest_results(model)
        if results:
            all_results[model] = results
            print(f"✅ 已加載 {model} 的結果")
        else:
            print(f"❌ 未找到 {model} 的結果")
    
    if not all_results:
        print("❌ 沒有找到任何模型結果")
        return
    
    # 創建比較報告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"experiments/comprehensive/benchmark_comparison_{timestamp}.md"
    
    report = f"""# 量子模型 vs Transformer Benchmark比較報告

## 概述
- **比較時間**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **參與模型**: {', '.join(all_results.keys())}
- **數據集**: simple_sentences, short_stories, quantum_phrases, haiku, proverbs

## 模型說明
- **Transformer**: 傳統Transformer模型 (baseline)
- **QKSAN**: 量子Kernel Self-Attention Network (真量子)
- **QRWKV**: 量子RWKV模型 (真量子)
- **QASA**: 量子Attention Sequence Architecture (真量子)
- **MLP**: 多層感知機 (假量子)

## 詳細比較

"""
    
    # 獲取所有數據集
    all_datasets = set()
    for model_results in all_results.values():
        all_datasets.update(model_results.keys())
    
    datasets = ['simple_sentences', 'short_stories', 'quantum_phrases', 'haiku', 'proverbs']
    
    for dataset in datasets:
        if dataset not in all_datasets:
            continue
            
        report += f"\n### {dataset}\n\n"
        report += "| 模型 | 困惑度 | BLEU-1 | 詞彙多樣性 | 重複率 |\n"
        report += "|------|--------|--------|------------|--------|\n"
        
        for model in models:
            if model not in all_results or dataset not in all_results[model]:
                report += f"| {model.upper()} | N/A | N/A | N/A | N/A |\n"
                continue
            
            result = all_results[model][dataset]
            
            if result.get('status') == 'failed':
                report += f"| {model.upper()} | ❌失敗 | ❌失敗 | ❌失敗 | ❌失敗 |\n"
                continue
            
            # 對於transformer模型
            if model == 'transformer':
                perplexity = result.get('perplexity', 'N/A')
                bleu_scores = result.get('bleu_scores', {})
                bleu_1 = bleu_scores.get('BLEU-1', 'N/A') if isinstance(bleu_scores, dict) else 'N/A'
                diversity = result.get('diversity_metrics', {}).get('distinct_1', 'N/A')
                repetition = result.get('repetition_rate', 'N/A')
            else:
                # 對於其他模型，數據在evaluation_results中
                eval_results = result.get('evaluation_results', {})
                perplexity = eval_results.get('perplexity', 'N/A')
                bleu_scores = eval_results.get('bleu_scores', {})
                bleu_1 = bleu_scores.get('BLEU-1', 'N/A') if isinstance(bleu_scores, dict) else 'N/A'
                diversity = eval_results.get('diversity_metrics', {}).get('distinct_1', 'N/A')
                repetition = eval_results.get('repetition_rate', 'N/A')
            
            # 格式化數值
            if isinstance(perplexity, (int, float)):
                perplexity = f"{perplexity:.4f}"
            if isinstance(bleu_1, (int, float)):
                bleu_1 = f"{bleu_1:.4f}"
            if isinstance(diversity, (int, float)):
                diversity = f"{diversity:.4f}"
            if isinstance(repetition, (int, float)):
                repetition = f"{repetition:.4f}"
            
            report += f"| {model.upper()} | {perplexity} | {bleu_1} | {diversity} | {repetition} |\n"
    
    # 添加總結
    report += f"""

## 總結分析

### 性能排名 (基於平均BLEU-1分數)
"""
    
    # 計算平均BLEU分數
    model_avg_bleu = {}
    for model in models:
        if model not in all_results:
            continue
        
        bleu_scores = []
        for dataset in datasets:
            if dataset in all_results[model]:
                result = all_results[model][dataset]
                if result.get('status') != 'failed':
                    # 對於transformer模型
                    if model == 'transformer':
                        bleu_1 = result.get('bleu_scores', {}).get('BLEU-1', 0)
                    else:
                        # 對於其他模型，數據在evaluation_results中
                        eval_results = result.get('evaluation_results', {})
                        bleu_1 = eval_results.get('bleu_scores', {}).get('BLEU-1', 0)
                    
                    if isinstance(bleu_1, (int, float)) and bleu_1 > 0:
                        bleu_scores.append(bleu_1)
        
        if bleu_scores:
            model_avg_bleu[model] = sum(bleu_scores) / len(bleu_scores)
        else:
            model_avg_bleu[model] = 0.0
    
    # 排序
    sorted_models = sorted(model_avg_bleu.items(), key=lambda x: x[1], reverse=True)
    
    for i, (model, avg_bleu) in enumerate(sorted_models, 1):
        report += f"{i}. **{model.upper()}**: {avg_bleu:.4f}\n"
    
    report += f"""

### 關鍵發現

1. **量子模型表現**: 量子模型在特定任務上可能表現優於傳統Transformer
2. **數據集適應性**: 不同模型在不同數據集上表現各異
3. **模型複雜度**: 量子模型雖然參數較少，但可能具有更強的表示能力
4. **訓練穩定性**: 需要進一步分析訓練過程的穩定性

### 建議

1. **進一步優化**: 針對表現較差的模型進行超參數調優
2. **擴大評估**: 在更多數據集和指標上進行評估
3. **深入分析**: 分析量子模型在不同類型文本上的優勢
4. **實際應用**: 考慮在實際應用場景中的表現

---
*報告生成時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    # 保存報告
    os.makedirs("experiments/comprehensive", exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 比較報告已保存到: {report_file}")
    
    # 同時創建JSON格式的比較數據
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'models': list(all_results.keys()),
        'datasets': datasets,
        'results': all_results,
        'summary': {
            'model_avg_bleu': model_avg_bleu,
            'best_model': sorted_models[0][0] if sorted_models else None,
            'best_score': sorted_models[0][1] if sorted_models else None
        }
    }
    
    json_file = f"experiments/comprehensive/benchmark_comparison_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 比較數據已保存到: {json_file}")
    
    return report_file

if __name__ == "__main__":
    create_comparison_report()
