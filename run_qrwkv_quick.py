#!/usr/bin/env python3
"""
QRWKV模型快速測試腳本 - 只運行一個數據集
"""

import os
import sys
import torch
import json
from datetime import datetime

sys.path.append('.')

from run_qrwkv_only import run_qrwkv_experiment

def main():
    """主函數 - 快速測試"""
    print("🚀 QRWKV模型快速測試")
    print("=" * 40)
    
    # 設置設備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用設備: {device}")
    
    # 只測試一個數據集
    dataset_name = 'simple_sentences'
    
    # 快速配置
    epochs = 3
    batch_size = 8
    max_length = 16
    vocab_size = 200
    max_samples = 100
    learning_rate = 0.001
    
    print(f"📊 數據集: {dataset_name}")
    print(f"🔄 訓練輪數: {epochs} epochs")
    print(f"📈 快速測試配置")
    
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
        
        print(f"\n✅ QRWKV快速測試完成!")
        print(f"📁 結果保存到: {result['model_dir']}")
        
        # 顯示生成結果
        if 'evaluation_results' in result:
            eval_results = result['evaluation_results']
            print(f"\n📝 生成文本示例:")
            for i, text in enumerate(eval_results['generated_texts']):
                print(f"  {i+1}. {text}")
            
            print(f"\n📊 評估指標:")
            print(f"  困惑度: {eval_results.get('perplexity', 'N/A'):.4f}")
            bleu_scores = eval_results.get('bleu_scores', {})
            if isinstance(bleu_scores, dict):
                print(f"  BLEU分數: BLEU-1: {bleu_scores.get('BLEU-1', 'N/A'):.4f}")
            else:
                print(f"  BLEU分數: {bleu_scores}")
            
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
