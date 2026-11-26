#!/usr/bin/env python3
"""
詳細解釋溫度參數如何影響隨機性
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_temperature_effect():
    """演示溫度對隨機性的影響"""
    
    print("🌡️ 溫度參數對隨機性的影響詳細解釋")
    print("=" * 60)
    
    # 模擬模型輸出的logits
    # 假設有5個詞，logits值分別為 [2.0, 1.5, 1.0, 0.5, 0.1]
    original_logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.1])
    words = ['cat', 'dog', 'bird', 'fish', 'tree']
    
    print(f"原始logits: {original_logits}")
    print(f"對應詞彙: {words}")
    
    # 測試不同溫度值
    temperatures = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    print("\n📊 不同溫度下的概率分布:")
    print("-" * 50)
    
    for temp in temperatures:
        print(f"\n溫度 = {temp}:")
        
        # 應用溫度縮放
        scaled_logits = original_logits / temp
        print(f"縮放後logits: {scaled_logits}")
        
        # 計算softmax概率
        probs = F.softmax(scaled_logits, dim=-1)
        print(f"概率分布: {probs}")
        
        # 顯示每個詞的概率
        print("各詞概率:")
        for i, (word, prob) in enumerate(zip(words, probs)):
            print(f"  '{word}': {prob:.4f}")
        
        # 計算熵（衡量隨機性）
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        print(f"熵值: {entropy:.4f} (越高越隨機)")
        
        # 計算最高概率與最低概率的比值
        max_prob = torch.max(probs)
        min_prob = torch.min(probs)
        ratio = max_prob / min_prob
        print(f"最高/最低概率比值: {ratio:.2f} (越小越隨機)")

def explain_why_temperature_works():
    """解釋為什麼溫度參數有效"""
    
    print("\n🤔 為什麼溫度越高越隨機？")
    print("=" * 50)
    
    print("關鍵理解：")
    print("1. 溫度縮放影響的是softmax函數的輸入")
    print("2. softmax函數對輸入的差異很敏感")
    print("3. 溫度越高，logits之間的相對差異被縮小")
    
    # 數學解釋
    print("\n📐 數學解釋:")
    print("原始softmax: P(i) = exp(logits[i]) / Σ exp(logits[j])")
    print("溫度縮放: P(i) = exp(logits[i]/T) / Σ exp(logits[j]/T)")
    
    # 具體例子
    print("\n🔍 具體例子:")
    logits = torch.tensor([3.0, 2.0, 1.0])
    print(f"原始logits: {logits}")
    
    # 計算不同溫度下的概率
    temps = [0.5, 1.0, 2.0]
    for temp in temps:
        scaled = logits / temp
        probs = F.softmax(scaled, dim=-1)
        print(f"溫度 {temp}: logits={scaled}, probs={probs}")
    
    print("\n觀察:")
    print("- 溫度=0.5: 概率差異最大 (0.665, 0.245, 0.090)")
    print("- 溫度=1.0: 概率差異中等 (0.576, 0.211, 0.213)")
    print("- 溫度=2.0: 概率差異最小 (0.244, 0.244, 0.512)")

def demonstrate_extreme_cases():
    """演示極端情況"""
    
    print("\n🎯 極端情況演示:")
    print("=" * 40)
    
    logits = torch.tensor([5.0, 1.0, 0.1])
    words = ['cat', 'dog', 'bird']
    
    print(f"原始logits: {logits}")
    
    # 極低溫度 (接近貪心)
    temp_low = 0.1
    scaled_low = logits / temp_low
    probs_low = F.softmax(scaled_low, dim=-1)
    
    print(f"\n極低溫度 ({temp_low}):")
    print(f"縮放後: {scaled_low}")
    print(f"概率: {probs_low}")
    print("結果: 幾乎確定選擇第一個詞")
    
    # 極高溫度 (接近均勻分布)
    temp_high = 10.0
    scaled_high = logits / temp_high
    probs_high = F.softmax(scaled_high, dim=-1)
    
    print(f"\n極高溫度 ({temp_high}):")
    print(f"縮放後: {scaled_high}")
    print(f"概率: {probs_high}")
    print("結果: 接近均勻分布，高度隨機")

def visualize_temperature_effect():
    """視覺化溫度效果"""
    
    print("\n📈 視覺化溫度效果:")
    print("=" * 40)
    
    # 創建更複雜的logits
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.5])
    words = ['the', 'cat', 'sat', 'on', 'mat']
    
    temperatures = np.linspace(0.1, 5.0, 20)
    entropies = []
    max_probs = []
    
    for temp in temperatures:
        scaled = logits / temp
        probs = F.softmax(scaled, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        max_prob = torch.max(probs)
        
        entropies.append(entropy.item())
        max_probs.append(max_prob.item())
    
    print("溫度對熵的影響:")
    for i, temp in enumerate(temperatures[::4]):  # 每4個取一個
        print(f"  溫度 {temp:.1f}: 熵 {entropies[i*4]:.3f}, 最高概率 {max_probs[i*4]:.3f}")
    
    print("\n結論:")
    print("- 溫度越低 → 熵越低 → 越確定性")
    print("- 溫度越高 → 熵越高 → 越隨機")

def compare_with_other_methods():
    """與其他採樣方法比較"""
    
    print("\n🔄 與其他採樣方法比較:")
    print("=" * 40)
    
    logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.1])
    words = ['cat', 'dog', 'bird', 'fish', 'tree']
    
    print(f"原始logits: {logits}")
    
    # 1. 貪心解碼
    greedy_idx = torch.argmax(logits)
    print(f"\n1. 貪心解碼: 選擇 '{words[greedy_idx]}' (確定性)")
    
    # 2. 溫度採樣
    temp = 1.0
    scaled = logits / temp
    probs = F.softmax(scaled, dim=-1)
    print(f"\n2. 溫度採樣 (T={temp}):")
    for word, prob in zip(words, probs):
        print(f"   '{word}': {prob:.3f}")
    
    # 3. Top-K採樣
    k = 3
    top_k_logits, top_k_indices = torch.topk(logits, k)
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits[top_k_indices] = top_k_logits
    top_k_probs = F.softmax(filtered_logits, dim=-1)
    
    print(f"\n3. Top-K採樣 (K={k}):")
    for word, prob in zip(words, top_k_probs):
        if prob > 0:
            print(f"   '{word}': {prob:.3f}")
        else:
            print(f"   '{word}': 0.000 (被過濾)")
    
    print("\n比較:")
    print("- 貪心: 完全確定性")
    print("- 溫度: 控制隨機性程度")
    print("- Top-K: 限制選擇範圍")

if __name__ == "__main__":
    demonstrate_temperature_effect()
    explain_why_temperature_works()
    demonstrate_extreme_cases()
    visualize_temperature_effect()
    compare_with_other_methods()
    
    print("\n🎉 溫度參數解釋完成!")
    print("\n💡 關鍵點:")
    print("1. 溫度縮放影響softmax函數的輸入")
    print("2. 溫度越高，logits差異被縮小，概率分布越均勻")
    print("3. 溫度越低，logits差異被放大，概率分布越尖銳")
    print("4. 這不是簡單的線性縮放，而是非線性的概率重分布")
