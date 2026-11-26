#!/usr/bin/env python3
"""
文本生成原理和Top-K採樣詳細解釋
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

def explain_text_generation():
    """詳細解釋文本生成原理"""
    
    print("🎯 文本生成原理詳細解釋")
    print("=" * 60)
    
    # 1. 語言模型的基本概念
    print("\n📚 1. 語言模型的基本概念")
    print("-" * 40)
    print("語言模型學習預測：P(下一個詞 | 前面的詞序列)")
    print("例如：給定 'the cat'，預測下一個詞可能是什麼？")
    print("可能的概率分布：")
    print("  - sat: 0.3")
    print("  - runs: 0.2") 
    print("  - sleeps: 0.15")
    print("  - jumps: 0.1")
    print("  - ... 其他詞: 0.25")
    
    # 2. 生成過程演示
    print("\n🔄 2. 文本生成過程演示")
    print("-" * 40)
    
    # 模擬詞彙表
    vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', 'the', 'cat', 'sat', 'on', 'mat', 'runs', 'sleeps', 'jumps', 'dog', 'bird', 'tree']
    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for i, word in enumerate(vocab)}
    
    print(f"詞彙表: {vocab}")
    print(f"詞彙大小: {len(vocab)}")
    
    # 模擬模型輸出（logits）
    def simulate_model_output(context: str) -> torch.Tensor:
        """模擬模型對給定上下文的輸出"""
        # 這裡我們模擬一個簡單的概率分布
        if context == "the":
            # 給定 "the"，下一個詞的概率分布
            logits = torch.tensor([-10, -10, -10, -10, -10, 2.0, 1.5, 1.0, 0.8, 1.2, 0.9, 0.7, 1.1, 1.3, 0.6])
        elif context == "the cat":
            # 給定 "the cat"，下一個詞的概率分布
            logits = torch.tensor([-10, -10, -10, -10, -10, -10, 2.5, 1.8, 1.2, 1.5, 1.0, 0.8, -10, -10, -10])
        else:
            # 默認分布
            logits = torch.tensor([-10, -10, -10, -10, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        return logits
    
    # 3. 貪心解碼 vs 採樣解碼
    print("\n🎲 3. 貪心解碼 vs 採樣解碼")
    print("-" * 40)
    
    context = "the"
    logits = simulate_model_output(context)
    
    print(f"上下文: '{context}'")
    print(f"原始logits: {logits}")
    
    # 貪心解碼
    greedy_token = torch.argmax(logits).item()
    greedy_word = id_to_word[greedy_token]
    print(f"貪心解碼結果: '{greedy_word}' (token_id: {greedy_token})")
    
    # 採樣解碼
    probs = F.softmax(logits, dim=-1)
    sampled_token = torch.multinomial(probs, 1).item()
    sampled_word = id_to_word[sampled_token]
    print(f"採樣解碼結果: '{sampled_word}' (token_id: {sampled_token})")
    
    # 4. Top-K 採樣詳細解釋
    print("\n🔝 4. Top-K 採樣詳細解釋")
    print("-" * 40)
    
    def demonstrate_top_k_sampling(logits: torch.Tensor, k: int = 3):
        """演示Top-K採樣過程"""
        print(f"Top-K採樣 (k={k}) 過程:")
        print(f"原始logits: {logits}")
        
        # 1. 找到top-k個最大值的索引和值
        top_k_values, top_k_indices = torch.topk(logits, k)
        print(f"Top-{k} 值: {top_k_values}")
        print(f"Top-{k} 索引: {top_k_indices}")
        
        # 2. 創建新的logits，將非top-k的值設為負無窮
        new_logits = torch.full_like(logits, float('-inf'))
        new_logits[top_k_indices] = top_k_values
        print(f"過濾後的logits: {new_logits}")
        
        # 3. 計算softmax概率
        probs = F.softmax(new_logits, dim=-1)
        print(f"概率分布: {probs}")
        
        # 4. 顯示每個詞的概率
        print("各詞概率:")
        for i, prob in enumerate(probs):
            if prob > 0:
                word = id_to_word[i]
                print(f"  '{word}': {prob:.4f}")
        
        # 5. 採樣
        sampled_token = torch.multinomial(probs, 1).item()
        sampled_word = id_to_word[sampled_token]
        print(f"採樣結果: '{sampled_word}' (概率: {probs[sampled_token]:.4f})")
        
        return sampled_word
    
    # 演示不同k值的效果
    print("\n演示不同k值的效果:")
    for k in [1, 3, 5]:
        print(f"\n--- k={k} ---")
        demonstrate_top_k_sampling(logits, k)
    
    # 5. 溫度參數的作用
    print("\n🌡️ 5. 溫度參數的作用")
    print("-" * 40)
    
    def demonstrate_temperature(logits: torch.Tensor, temperature: float):
        """演示溫度參數的作用"""
        print(f"溫度 = {temperature}")
        print(f"原始logits: {logits}")
        
        # 應用溫度
        scaled_logits = logits / temperature
        print(f"縮放後logits: {scaled_logits}")
        
        # 計算概率
        probs = F.softmax(scaled_logits, dim=-1)
        print(f"概率分布: {probs}")
        
        # 顯示前3個最高概率的詞
        top_probs, top_indices = torch.topk(probs, 3)
        print("前3個最高概率的詞:")
        for i in range(3):
            word = id_to_word[top_indices[i].item()]
            prob = top_probs[i].item()
            print(f"  '{word}': {prob:.4f}")
        
        return probs
    
    # 演示不同溫度的效果
    temperatures = [0.5, 1.0, 2.0]
    for temp in temperatures:
        print(f"\n--- 溫度 = {temp} ---")
        demonstrate_temperature(logits, temp)
    
    # 6. 完整生成過程演示
    print("\n🎬 6. 完整生成過程演示")
    print("-" * 40)
    
    def generate_text_step_by_step(prompt: str, max_length: int = 5):
        """逐步演示文本生成過程"""
        print(f"生成文本: '{prompt}' -> 最大長度: {max_length}")
        
        current_text = prompt
        generated_tokens = []
        
        for step in range(max_length):
            print(f"\n步驟 {step + 1}:")
            print(f"當前文本: '{current_text}'")
            
            # 模擬模型輸出
            logits = simulate_model_output(current_text)
            
            # 應用Top-K和溫度
            temperature = 0.7
            k = 3
            
            # Top-K過濾
            top_k_values, top_k_indices = torch.topk(logits, k)
            filtered_logits = torch.full_like(logits, float('-inf'))
            filtered_logits[top_k_indices] = top_k_values
            
            # 應用溫度
            scaled_logits = filtered_logits / temperature
            
            # 計算概率
            probs = F.softmax(scaled_logits, dim=-1)
            
            # 採樣
            sampled_token = torch.multinomial(probs, 1).item()
            sampled_word = id_to_word[sampled_token]
            
            print(f"Top-{k} 候選詞:")
            for i in range(k):
                word = id_to_word[top_k_indices[i].item()]
                prob = probs[top_k_indices[i].item()].item()
                print(f"  '{word}': {prob:.4f}")
            
            print(f"選擇: '{sampled_word}'")
            
            # 更新文本
            current_text += " " + sampled_word
            generated_tokens.append(sampled_word)
            
            # 檢查結束條件
            if sampled_word == '<EOS>' or sampled_word == '<PAD>':
                print("遇到結束標記，停止生成")
                break
        
        print(f"\n最終生成文本: '{current_text}'")
        return current_text
    
    # 演示完整生成過程
    generate_text_step_by_step("the", 4)
    
    # 7. 總結
    print("\n📋 7. 總結")
    print("-" * 40)
    print("文本生成的核心步驟:")
    print("1. 輸入上下文 -> 模型 -> 輸出logits")
    print("2. 應用Top-K過濾 -> 保留前K個最可能的詞")
    print("3. 應用溫度縮放 -> 控制隨機性")
    print("4. Softmax -> 轉換為概率分布")
    print("5. 採樣 -> 選擇下一個詞")
    print("6. 重複直到結束")
    
    print("\n關鍵參數:")
    print("- Top-K: 控制詞彙選擇範圍 (k越大，選擇越多)")
    print("- 溫度: 控制隨機性 (溫度越高，越隨機)")
    print("- 最大長度: 控制生成文本長度")

if __name__ == "__main__":
    explain_text_generation()
